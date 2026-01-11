from copy import deepcopy
import cv2
import numpy as np
import torch
import torch.nn as nn
import roma
from copy import deepcopy
import tqdm
import os
import matplotlib.pyplot as plt

from cloud_opt.utils import *
from cloud_opt.utils import _check_edges, _compute_img_conf
import cloud_opt.init_all as init_fun


class BaseOptimizer(nn.Module):
    """Optimize a global scene, given a graph-organized observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2), pred2 is in pred1's coordinate
    """

    def __init__(self, *args, **kwargs):
        pass

    def _init_from_views(
        self,
        view1s,
        view2s,
        pred1s,
        pred2s,  # whatever predictions, they should be organized into pairwise for graph optimization
        dist="l1",
        conf="log",
        min_conf_thr=3,
        thr_for_init_conf=False,
        base_scale=0.5,
        allow_pw_adaptors=False,
        pw_break=20,
        rand_pose=torch.randn,
        empty_cache=False,
        verbose=True,
    ):
        super().__init__()
        self.edges = [
            (int(view1["idx"]), int(view2["idx"]))
            for view1, view2 in zip(view1s, view2s)
        ]
        self.dist = ALL_DISTS[dist]
        self.n_imgs = _check_edges(self.edges)

        self.edge2pts_i = NoGradParamDict(
            {ij: pred1s[n]["pts3d_is_self_view"] for n, ij in enumerate(self.str_edges)}
        )  # ij: the name of the edge
        self.edge2pts_j = NoGradParamDict(
            {
                ij: pred2s[n]["pts3d_in_other_view"]
                for n, ij in enumerate(self.str_edges)
            }
        )
        self.edge2conf_i = NoGradParamDict(
            {ij: pred1s[n]["conf_self"] for n, ij in enumerate(self.str_edges)}
        )
        self.edge2conf_j = NoGradParamDict(
            {ij: pred2s[n]["conf"] for n, ij in enumerate(self.str_edges)}
        )

        self.imshapes = get_imshapes(self.edges, pred1s, pred2s)
        self.min_conf_thr = min_conf_thr
        self.thr_for_init_conf = thr_for_init_conf
        self.conf_trf = get_conf_trf(conf)

        self.im_conf = _compute_img_conf(
            self.imshapes, self.device, self.edges, self.edge2conf_i, self.edge2conf_j
        )
        for i in range(len(self.im_conf)):
            self.im_conf[i].requires_grad = False

        self.init_conf_maps = [c.clone() for c in self.im_conf]

        self.base_scale = base_scale
        self.norm_pw_scale = True
        self.pw_break = pw_break
        self.POSE_DIM = 7
        self.pw_poses = nn.Parameter(
            rand_pose((self.n_edges, 1 + self.POSE_DIM))
        )  # pairwise poses
        self.pw_adaptors = nn.Parameter(
            torch.zeros((self.n_edges, 2))
        )  # slight xy/z adaptation
        self.pw_adaptors.requires_grad_(allow_pw_adaptors)
        self.has_im_poses = False
        self.rand_pose = rand_pose

    def get_known_poses(self):
        if self.has_im_poses:
            known_poses_msk = torch.tensor(
                [not (p.requires_grad) for p in self.im_poses]
            )
            known_poses = self.get_im_poses()
            return known_poses_msk.sum(), known_poses_msk, known_poses
        else:
            return 0, None, None

    def get_pw_norm_scale_factor(self):
        if self.norm_pw_scale:
            # normalize scales so that things cannot go south
            # we want that exp(scale) ~= self.base_scale
            return (np.log(self.base_scale) - self.pw_poses[:, -1].mean()).exp()
        else:
            return 1  # don't norm scale for known poses

    def _set_pose(self, poses, idx, R, T=None, scale=None, force=False):
        # all poses == cam-to-world
        pose = poses[idx]
        if not (pose.requires_grad or force):
            return pose

        if R.shape == (4, 4):
            assert T is None
            T = R[:3, 3]
            R = R[:3, :3]

        if R is not None:
            pose.data[0:4] = roma.rotmat_to_unitquat(R)
        if T is not None:
            pose.data[4:7] = signed_log1p(
                T / (scale or 1)
            )  # translation is function of scale

        if scale is not None:
            assert poses.shape[-1] in (8, 13)
            pose.data[-1] = np.log(float(scale))
        return pose

    def forward(self, ret_details=False):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors()
        proj_pts3d = self.get_pts3d()
        # pre-compute pixel weights
        weight_i = {i_j: self.conf_trf(c) for i_j, c in self.conf_i.items()}
        weight_j = {i_j: self.conf_trf(c) for i_j, c in self.conf_j.items()}

        loss = 0
        if ret_details:
            details = -torch.ones((self.n_imgs, self.n_imgs))

        for e, (i, j) in enumerate(self.edges):
            i_j = edge_str(i, j)
            # distance in image i and j
            aligned_pred_i = geotrf(pw_poses[e], pw_adapt[e] * self.pred_i[i_j])
            aligned_pred_j = geotrf(pw_poses[e], pw_adapt[e] * self.pred_j[i_j])
            li = self.dist(proj_pts3d[i], aligned_pred_i, weight=weight_i[i_j]).mean()
            lj = self.dist(proj_pts3d[j], aligned_pred_j, weight=weight_j[i_j]).mean()
            loss = loss + li + lj

            if ret_details:
                details[i, j] = li + lj
        loss /= self.n_edges  # average over all pairs

        if ret_details:
            return loss, details
        return loss

    @torch.cuda.amp.autocast(enabled=False)
    def compute_global_alignment(self, init=None, niter_PnP=10, **kw):
        if init is None:
            pass
        elif init == "msp" or init == "mst":
            init_fun.init_minimum_spanning_tree(self, niter_PnP=niter_PnP)
        elif init == "known_poses":
            raise NotImplementedError
            self.preset_pose(known_poses=self.camera_poses, requires_grad=True)
            init_fun.init_from_known_poses(
                self, min_conf_thr=self.min_conf_thr, niter_PnP=niter_PnP
            )
        else:
            raise ValueError(f"bad value for {init=}")

        return global_alignment_loop(self, **kw)

    @property
    def str_edges(self):
        return [edge_str(i, j) for i, j in self.edges]

    @property
    def n_edges(self):
        return len(self.edges)


def global_alignment_loop(
    net,
    lr=0.01,
    niter=300,
    schedule="cosine",
    lr_min=1e-3,
    temporal_smoothing_weight=0,
    depth_map_save_dir=None,
):
    params = [p for p in net.parameters() if p.requires_grad]
    if not params:
        return net

    verbose = net.verbose
    if verbose:
        print("Global alignement - optimizing for:")
        print([name for name, value in net.named_parameters() if value.requires_grad])

    lr_base = lr
    optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9))

    loss = float("inf")
    if verbose:
        with tqdm.tqdm(total=niter) as bar:
            while bar.n < bar.total:
                if bar.n % 500 == 0 and depth_map_save_dir is not None:
                    if not os.path.exists(depth_map_save_dir):
                        os.makedirs(depth_map_save_dir)
                    # visualize the depthmaps
                    depth_maps = net.get_depthmaps()
                    for i, depth_map in enumerate(depth_maps):
                        depth_map_save_path = os.path.join(
                            depth_map_save_dir, f"depthmaps_{i}_iter_{bar.n}.png"
                        )
                        plt.imsave(
                            depth_map_save_path,
                            depth_map.detach().cpu().numpy(),
                            cmap="jet",
                        )
                    print(
                        f"Saved depthmaps at iteration {bar.n} to {depth_map_save_dir}"
                    )
                loss, lr = global_alignment_iter(
                    net,
                    bar.n,
                    niter,
                    lr_base,
                    lr_min,
                    optimizer,
                    schedule,
                    temporal_smoothing_weight=temporal_smoothing_weight,
                )
                bar.set_postfix_str(f"{lr=:g} loss={loss:g}")
                bar.update()
    else:
        for n in range(niter):
            loss, _ = global_alignment_iter(
                net,
                n,
                niter,
                lr_base,
                lr_min,
                optimizer,
                schedule,
                temporal_smoothing_weight=temporal_smoothing_weight,
            )
    return loss


def global_alignment_iter(
    net,
    cur_iter,
    niter,
    lr_base,
    lr_min,
    optimizer,
    schedule,
    temporal_smoothing_weight=0,
):
    t = cur_iter / niter
    if schedule == "cosine":
        lr = cosine_schedule(t, lr_base, lr_min)
    elif schedule == "linear":
        lr = linear_schedule(t, lr_base, lr_min)
    elif schedule.startswith("cycle"):
        try:
            num_cycles = int(schedule[5:])
        except ValueError:
            num_cycles = 2
        lr = cycled_linear_schedule(t, lr_base, lr_min, num_cycles=num_cycles)
    else:
        raise ValueError(f"bad lr {schedule=}")

    adjust_learning_rate_by_lr(optimizer, lr)
    optimizer.zero_grad()

    if net.empty_cache:
        torch.cuda.empty_cache()

    loss = net(epoch=cur_iter)

    if net.empty_cache:
        torch.cuda.empty_cache()

    loss.backward()

    if net.empty_cache:
        torch.cuda.empty_cache()

    optimizer.step()

    return float(loss), lr
