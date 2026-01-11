from functools import cache
import numpy as np
import scipy.sparse as sp
import torch
import cv2
import roma
from tqdm import tqdm

from cloud_opt.utils import *


def compute_edge_scores(edges, edge2conf_i, edge2conf_j):
    """
    edges: 'i_j', (i,j)
    """
    score_dict = {
        (i, j): edge_conf(edge2conf_i[e], edge2conf_j[e]) for e, (i, j) in edges
    }
    return score_dict


def dict_to_sparse_graph(dic):
    n_imgs = max(max(e) for e in dic) + 1
    res = sp.dok_array((n_imgs, n_imgs))
    for edge, value in dic.items():
        res[edge] = value
    return res


@torch.no_grad()
def init_minimum_spanning_tree(self, **kw):
    """Init all camera poses (image-wise and pairwise poses) given
    an initial set of pairwise estimations.
    """
    device = self.device
    pts3d, _, im_focals, im_poses = minimum_spanning_tree(
        self.imshapes,
        self.edges,
        self.edge2pts_i,
        self.edge2pts_j,
        self.edge2conf_i,
        self.edge2conf_j,
        self.im_conf,
        self.min_conf_thr,
        device,
        has_im_poses=self.has_im_poses,
        verbose=self.verbose,
        **kw,
    )

    return init_from_pts3d(self, pts3d, im_focals, im_poses)


def minimum_spanning_tree(
    imshapes,
    edges,
    edge2pred_i,
    edge2pred_j,
    edge2conf_i,
    edge2conf_j,
    im_conf,
    min_conf_thr,
    device,
    has_im_poses=True,
    niter_PnP=10,
    verbose=True,
    save_score_path=None,
):
    n_imgs = len(imshapes)
    eadge_and_scores = compute_edge_scores(map(i_j_ij, edges), edge2conf_i, edge2conf_j)
    sparse_graph = -dict_to_sparse_graph(eadge_and_scores)
    msp = sp.csgraph.minimum_spanning_tree(sparse_graph).tocoo()

    # temp variable to store 3d points
    pts3d = [None] * len(imshapes)

    todo = sorted(zip(-msp.data, msp.row, msp.col))  # sorted edges
    im_poses = [None] * n_imgs
    im_focals = [None] * n_imgs

    # init with strongest edge
    score, i, j = todo.pop()
    if verbose:
        print(f" init edge ({i}*,{j}*) {score=}")
    i_j = edge_str(i, j)

    pts3d[i] = edge2pred_i[i_j].clone()
    pts3d[j] = edge2pred_j[i_j].clone()
    done = {i, j}
    if has_im_poses:
        im_poses[i] = torch.eye(4, device=device)
        im_focals[i] = estimate_focal(edge2pred_i[i_j])

    # set initial pointcloud based on pairwise graph
    msp_edges = [(i, j)]
    while todo:
        # each time, predict the next one
        score, i, j = todo.pop()

        if im_focals[i] is None:
            im_focals[i] = estimate_focal(edge2pred_i[i_j])

        if i in done:
            if verbose:
                print(f" init edge ({i},{j}*) {score=}")
            assert j not in done
            # align pred[i] with pts3d[i], and then set j accordingly
            i_j = edge_str(i, j)
            s, R, T = rigid_points_registration(
                edge2pred_i[i_j], pts3d[i], conf=edge2conf_i[i_j]
            )
            trf = sRT_to_4x4(s, R, T, device)
            pts3d[j] = geotrf(trf, edge2pred_j[i_j])
            done.add(j)
            msp_edges.append((i, j))

            if has_im_poses and im_poses[i] is None:
                im_poses[i] = sRT_to_4x4(1, R, T, device)

        elif j in done:
            if verbose:
                print(f" init edge ({i}*,{j}) {score=}")
            assert i not in done
            i_j = edge_str(i, j)
            s, R, T = rigid_points_registration(
                edge2pred_j[i_j], pts3d[j], conf=edge2conf_j[i_j]
            )
            trf = sRT_to_4x4(s, R, T, device)
            pts3d[i] = geotrf(trf, edge2pred_i[i_j])
            done.add(i)
            msp_edges.append((i, j))

            if has_im_poses and im_poses[i] is None:
                im_poses[i] = sRT_to_4x4(1, R, T, device)
        else:
            # let's try again later
            todo.insert(0, (score, i, j))

    if has_im_poses:
        # complete all missing informations
        pair_scores = list(
            sparse_graph.values()
        )  # already negative scores: less is best
        edges_from_best_to_worse = np.array(list(sparse_graph.keys()))[
            np.argsort(pair_scores)
        ]
        for i, j in edges_from_best_to_worse.tolist():
            if im_focals[i] is None:
                im_focals[i] = estimate_focal(edge2pred_i[edge_str(i, j)])

        for i in range(n_imgs):
            if im_poses[i] is None:
                msk = im_conf[i] > min_conf_thr
                res = fast_pnp(
                    pts3d[i], im_focals[i], msk=msk, device=device, niter_PnP=niter_PnP
                )
                if res:
                    im_focals[i], im_poses[i] = res
            if im_poses[i] is None:
                im_poses[i] = torch.eye(4, device=device)
        im_poses = torch.stack(im_poses)
    else:
        im_poses = im_focals = None

    return pts3d, msp_edges, im_focals, im_poses


def init_from_pts3d(self, pts3d, im_focals, im_poses):
    # init poses
    nkp, known_poses_msk, known_poses = self.get_known_poses()
    if nkp == 1:
        raise NotImplementedError(
            "Would be simpler to just align everything afterwards on the single known pose"
        )
    elif nkp > 1:
        # global rigid SE3 alignment
        s, R, T = align_multiple_poses(
            im_poses[known_poses_msk], known_poses[known_poses_msk]
        )
        trf = sRT_to_4x4(s, R, T, device=known_poses.device)

        # rotate everything
        im_poses = trf @ im_poses
        im_poses[:, :3, :3] /= s  # undo scaling on the rotation part
        for img_pts3d in pts3d:
            img_pts3d[:] = geotrf(trf, img_pts3d)
    else:
        pass  # no known poses

    # set all pairwise poses
    for e, (i, j) in enumerate(self.edges):
        i_j = edge_str(i, j)
        # compute transform that goes from cam to world
        s, R, T = rigid_points_registration(
            self.pred_i[i_j], pts3d[i], conf=self.conf_i[i_j]
        )
        self._set_pose(self.pw_poses, e, R, T, scale=s)

    # take into account the scale normalization
    s_factor = self.get_pw_norm_scale_factor()
    im_poses[:, :3, 3] *= s_factor  # apply downscaling factor
    for img_pts3d in pts3d:
        img_pts3d *= s_factor

    # init all image poses
    if self.has_im_poses:
        for i in range(self.n_imgs):
            cam2world = im_poses[i]
            depth = geotrf(inv(cam2world), pts3d[i])[..., 2]
            self._set_depthmap(i, depth)
            self._set_pose(self.im_poses, i, cam2world)
            if im_focals[i] is not None:
                if not self.shared_focal:
                    self._set_focal(i, im_focals[i])
        if self.shared_focal:
            self._set_focal(0, sum(im_focals) / self.n_imgs)
        if self.n_imgs > 2:
            self._set_init_depthmap()

    if self.verbose:
        with torch.no_grad():
            print(" init loss =", float(self()))
