# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

import numpy as np
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import inv, geotrf


def reproject_view(pts3d, view2):
    shape = view2["pts3d"].shape[:2]
    return reproject(
        pts3d, view2["camera_intrinsics"], inv(view2["camera_pose"]), shape
    )


def reproject(pts3d, K, world2cam, shape):
    H, W, THREE = pts3d.shape
    assert THREE == 3

    # reproject in camera2 space
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = geotrf(K @ world2cam[:3], pts3d, norm=1, ncol=2)

    # quantize to pixel positions
    return (H, W), ravel_xy(pos, shape)


def ravel_xy(pos, shape):
    H, W = shape
    with np.errstate(invalid="ignore"):
        qx, qy = pos.reshape(-1, 2).round().astype(np.int32).T
    quantized_pos = qx.clip(min=0, max=W - 1, out=qx) + W * qy.clip(
        min=0, max=H - 1, out=qy
    )
    return quantized_pos


def unravel_xy(pos, shape):
    # convert (x+W*y) back to 2d (x,y) coordinates
    return np.unravel_index(pos, shape)[0].base[:, ::-1].copy()


def reciprocal_1d(corres_1_to_2, corres_2_to_1, ret_recip=False):
    is_reciprocal1 = corres_2_to_1[corres_1_to_2] == np.arange(len(corres_1_to_2))
    pos1 = is_reciprocal1.nonzero()[0]
    pos2 = corres_1_to_2[pos1]
    if ret_recip:
        return is_reciprocal1, pos1, pos2
    return pos1, pos2


def extract_correspondences_from_pts3d(
    view1, view2, target_n_corres, rng=np.random, ret_xy=True, nneg=0
):
    view1, view2 = to_numpy((view1, view2))
    # project pixels from image1 --> 3d points --> image2 pixels
    shape1, corres1_to_2 = reproject_view(view1["pts3d"], view2)
    shape2, corres2_to_1 = reproject_view(view2["pts3d"], view1)

    # compute reciprocal correspondences:
    # pos1 == valid pixels (correspondences) in image1
    is_reciprocal1, pos1, pos2 = reciprocal_1d(
        corres1_to_2, corres2_to_1, ret_recip=True
    )
    is_reciprocal2 = corres1_to_2[corres2_to_1] == np.arange(len(corres2_to_1))

    if target_n_corres is None:
        if ret_xy:
            pos1 = unravel_xy(pos1, shape1)
            pos2 = unravel_xy(pos2, shape2)
        return pos1, pos2

    available_negatives = min((~is_reciprocal1).sum(), (~is_reciprocal2).sum())
    target_n_positives = int(target_n_corres * (1 - nneg))
    n_positives = min(len(pos1), target_n_positives)
    n_negatives = min(target_n_corres - n_positives, available_negatives)

    if n_negatives + n_positives != target_n_corres:
        # should be really rare => when there are not enough negatives
        # in that case, break nneg and add a few more positives ?
        n_positives = target_n_corres - n_negatives
        assert n_positives <= len(pos1)

    assert n_positives <= len(pos1)
    assert n_positives <= len(pos2)
    assert n_negatives <= (~is_reciprocal1).sum()
    assert n_negatives <= (~is_reciprocal2).sum()
    assert n_positives + n_negatives == target_n_corres

    valid = np.ones(n_positives, dtype=bool)
    if n_positives < len(pos1):
        # random sub-sampling of valid correspondences
        perm = rng.permutation(len(pos1))[:n_positives]
        pos1 = pos1[perm]
        pos2 = pos2[perm]

    if n_negatives > 0:
        # add false correspondences if not enough
        def norm(p):
            return p / p.sum()

        pos1 = np.r_[
            pos1,
            rng.choice(
                shape1[0] * shape1[1],
                size=n_negatives,
                replace=False,
                p=norm(~is_reciprocal1),
            ),
        ]
        pos2 = np.r_[
            pos2,
            rng.choice(
                shape2[0] * shape2[1],
                size=n_negatives,
                replace=False,
                p=norm(~is_reciprocal2),
            ),
        ]
        valid = np.r_[valid, np.zeros(n_negatives, dtype=bool)]

    # convert (x+W*y) back to 2d (x,y) coordinates
    if ret_xy:
        pos1 = unravel_xy(pos1, shape1)
        pos2 = unravel_xy(pos2, shape2)
    return pos1, pos2, valid
