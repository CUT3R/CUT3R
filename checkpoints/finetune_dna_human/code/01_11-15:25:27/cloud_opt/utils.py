import torch.nn as nn
import torch
import roma
import numpy as np
import cv2
from functools import cache


def todevice(batch, device, callback=None, non_blocking=False):
    """Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    """
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == "numpy":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


to_device = todevice  # alias


def to_numpy(x):
    return todevice(x, "numpy")


def to_cpu(x):
    return todevice(x, "cpu")


def to_cuda(x):
    return todevice(x, "cuda")


def signed_log1p(x):
    sign = torch.sign(x)
    return sign * torch.log1p(torch.abs(x))


def l2_dist(a, b, weight):
    return (a - b).square().sum(dim=-1) * weight


def l1_dist(a, b, weight):
    return (a - b).norm(dim=-1) * weight


ALL_DISTS = dict(l1=l1_dist, l2=l2_dist)


def _check_edges(edges):
    indices = sorted({i for edge in edges for i in edge})
    assert indices == list(range(len(indices))), "bad pair indices: missing values "
    return len(indices)


def NoGradParamDict(x):
    assert isinstance(x, dict)
    return nn.ParameterDict(x).requires_grad_(False)


def edge_str(i, j):
    return f"{i}_{j}"


def i_j_ij(ij):
    # inputs are (i, j)
    return edge_str(*ij), ij


def edge_conf(conf_i, conf_j):
    score = float(conf_i.mean() * conf_j.mean())
    return score


def get_imshapes(edges, pred_i, pred_j):
    n_imgs = max(max(e) for e in edges) + 1
    imshapes = [None] * n_imgs
    for e, (i, j) in enumerate(edges):
        shape_i = tuple(pred_i[e]["pts3d_is_self_view"].shape[0:2])
        shape_j = tuple(pred_j[e]["pts3d_in_other_view"].shape[0:2])
        if imshapes[i]:
            assert imshapes[i] == shape_i, f"incorrect shape for image {i}"
        if imshapes[j]:
            assert imshapes[j] == shape_j, f"incorrect shape for image {j}"
        imshapes[i] = shape_i
        imshapes[j] = shape_j
    return imshapes


def get_conf_trf(mode):
    if mode == "log":

        def conf_trf(x):
            return x.log()

    elif mode == "sqrt":

        def conf_trf(x):
            return x.sqrt()

    elif mode == "m1":

        def conf_trf(x):
            return x - 1

    elif mode in ("id", "none"):

        def conf_trf(x):
            return x

    else:
        raise ValueError(f"bad mode for {mode=}")
    return conf_trf


@torch.no_grad()
def _compute_img_conf(imshapes, device, edges, edge2conf_i, edge2conf_j):
    im_conf = nn.ParameterList([torch.zeros(hw, device=device) for hw in imshapes])
    for e, (i, j) in enumerate(edges):
        im_conf[i] = torch.maximum(im_conf[i], edge2conf_i[edge_str(i, j)])
        im_conf[j] = torch.maximum(im_conf[j], edge2conf_j[edge_str(i, j)])
    return im_conf


def xy_grid(
    W,
    H,
    device=None,
    origin=(0, 0),
    unsqueeze=None,
    cat_dim=-1,
    homogeneous=False,
    **arange_kw,
):
    """Output a (H,W,2) array of int32
    with output[j,i,0] = i + origin[0]
         output[j,i,1] = j + origin[1]
    """
    if device is None:
        # numpy
        arange, meshgrid, stack, ones = np.arange, np.meshgrid, np.stack, np.ones
    else:
        # torch
        arange = lambda *a, **kw: torch.arange(*a, device=device, **kw)
        meshgrid, stack = torch.meshgrid, torch.stack
        ones = lambda *a: torch.ones(*a, device=device)

    tw, th = [arange(o, o + s, **arange_kw) for s, o in zip((W, H), origin)]
    grid = meshgrid(tw, th, indexing="xy")
    if homogeneous:
        grid = grid + (ones((H, W)),)
    if unsqueeze is not None:
        grid = (grid[0].unsqueeze(unsqueeze), grid[1].unsqueeze(unsqueeze))
    if cat_dim is not None:
        grid = stack(grid, cat_dim)
    return grid


def estimate_focal_knowing_depth(
    pts3d, pp, focal_mode="median", min_focal=0.0, max_focal=np.inf
):
    """Reprojection method, for when the absolute depth is known:
    1) estimate the camera focal using a robust estimator
    2) reproject points onto true rays, minimizing a certain error
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    # centered pixel grid
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(
        -1, 1, 2
    )  # B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)

    if focal_mode == "median":
        with torch.no_grad():
            # direct estimation of focal
            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # assume square pixels, hence same focal for X and Y
            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values

    elif focal_mode == "weiszfeld":
        # init focal with l2 closed form
        # we try to find focal = argmin Sum | pixel - focal * (x,y)/z|
        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(
            posinf=0, neginf=0
        )  # homogeneous (x,y,1)

        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)

        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

        # iterative re-weighted least-squares
        for iter in range(10):
            # re-weighting by inverse of distance
            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip(min=1e-8).reciprocal()
            # update the scaling with the new weights
            focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)
    else:
        raise ValueError(f"bad {focal_mode=}")

    focal_base = max(H, W) / (
        2 * np.tan(np.deg2rad(60) / 2)
    )  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal * focal_base, max=max_focal * focal_base)
    # print(focal)
    return focal


def estimate_focal(pts3d_i, pp=None):
    if pp is None:
        H, W, THREE = pts3d_i.shape
        assert THREE == 3
        pp = torch.tensor((W / 2, H / 2), device=pts3d_i.device)
    focal = estimate_focal_knowing_depth(
        pts3d_i.unsqueeze(0), pp.unsqueeze(0), focal_mode="weiszfeld"
    ).ravel()
    return float(focal)


def rigid_points_registration(pts1, pts2, conf):
    R, T, s = roma.rigid_points_registration(
        pts1.reshape(-1, 3),
        pts2.reshape(-1, 3),
        weights=conf.ravel(),
        compute_scaling=True,
    )
    return s, R, T  # return un-scaled (R, T)


def sRT_to_4x4(scale, R, T, device):
    trf = torch.eye(4, device=device)
    trf[:3, :3] = R * scale
    trf[:3, 3] = T.ravel()  # doesn't need scaling
    return trf


def geotrf(Trf, pts, ncol=None, norm=False):
    """Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)
    elif isinstance(Trf, torch.Tensor):
        pts = torch.as_tensor(pts, dtype=Trf.dtype)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if (
        isinstance(Trf, torch.Tensor)
        and isinstance(pts, torch.Tensor)
        and Trf.ndim == 3
        and pts.ndim == 4
    ):
        d = pts.shape[3]
        if Trf.shape[-1] == d:
            pts = torch.einsum("bij, bhwj -> bhwi", Trf, pts)
        elif Trf.shape[-1] == d + 1:
            pts = (
                torch.einsum("bij, bhwj -> bhwi", Trf[:, :d, :d], pts)
                + Trf[:, None, None, :d, d]
            )
        else:
            raise ValueError(f"bad shape, not ending with 3 or 4, for {pts.shape=}")
    else:
        if Trf.ndim >= 3:
            n = Trf.ndim - 2
            assert Trf.shape[:n] == pts.shape[:n], "batch size does not match"
            Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

            if pts.ndim > Trf.ndim:
                # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
                pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
            elif pts.ndim == 2:
                # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
                pts = pts[:, None, :]

        if pts.shape[-1] + 1 == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
        elif pts.shape[-1] == Trf.shape[-1]:
            Trf = Trf.swapaxes(-1, -2)  # transpose Trf
            pts = pts @ Trf
        else:
            pts = Trf @ pts.T
            if pts.ndim >= 2:
                pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res


def inv(mat):
    """Invert a torch or numpy matrix"""
    if isinstance(mat, torch.Tensor):
        return torch.linalg.inv(mat)
    if isinstance(mat, np.ndarray):
        return np.linalg.inv(mat)
    raise ValueError(f"bad matrix type = {type(mat)}")


@cache
def pixel_grid(H, W):
    return np.mgrid[:W, :H].T.astype(np.float32)


def fast_pnp(pts3d, focal, msk, device, pp=None, niter_PnP=10):
    # extract camera poses and focals with RANSAC-PnP
    if msk.sum() < 4:
        return None  # we need at least 4 points for PnP
    pts3d, msk = map(to_numpy, (pts3d, msk))

    H, W, THREE = pts3d.shape
    assert THREE == 3
    pixels = pixel_grid(H, W)

    if focal is None:
        S = max(W, H)
        tentative_focals = np.geomspace(S / 2, S * 3, 21)
    else:
        tentative_focals = [focal]

    if pp is None:
        pp = (W / 2, H / 2)
    else:
        pp = to_numpy(pp)

    best = (0,)
    for focal in tentative_focals:
        K = np.float32([(focal, 0, pp[0]), (0, focal, pp[1]), (0, 0, 1)])

        success, R, T, inliers = cv2.solvePnPRansac(
            pts3d[msk],
            pixels[msk],
            K,
            None,
            iterationsCount=niter_PnP,
            reprojectionError=5,
            flags=cv2.SOLVEPNP_SQPNP,
        )
        if not success:
            continue

        score = len(inliers)
        if success and score > best[0]:
            best = score, R, T, focal

    if not best[0]:
        return None

    _, R, T, best_focal = best
    R = cv2.Rodrigues(R)[0]  # world to cam
    R, T = map(torch.from_numpy, (R, T))
    return best_focal, inv(sRT_to_4x4(1, R, T, device))  # cam to world


def get_med_dist_between_poses(poses):
    from scipy.spatial.distance import pdist

    return np.median(pdist([to_numpy(p[:3, 3]) for p in poses]))


def align_multiple_poses(src_poses, target_poses):
    N = len(src_poses)
    assert src_poses.shape == target_poses.shape == (N, 4, 4)

    def center_and_z(poses):
        eps = get_med_dist_between_poses(poses) / 100
        return torch.cat((poses[:, :3, 3], poses[:, :3, 3] + eps * poses[:, :3, 2]))

    R, T, s = roma.rigid_points_registration(
        center_and_z(src_poses), center_and_z(target_poses), compute_scaling=True
    )
    return s, R, T


def cosine_schedule(t, lr_start, lr_end):
    assert 0 <= t <= 1
    return lr_end + (lr_start - lr_end) * (1 + np.cos(t * np.pi)) / 2


def linear_schedule(t, lr_start, lr_end):
    assert 0 <= t <= 1
    return lr_start + (lr_end - lr_start) * t


def cycled_linear_schedule(t, lr_start, lr_end, num_cycles=2):
    assert 0 <= t <= 1
    cycle_t = t * num_cycles
    cycle_t = cycle_t - int(cycle_t)
    if t == 1:
        cycle_t = 1
    return linear_schedule(cycle_t, lr_start, lr_end)


def adjust_learning_rate_by_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
