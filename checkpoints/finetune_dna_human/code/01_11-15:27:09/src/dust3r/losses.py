from copy import copy, deepcopy
import torch
import torch.nn as nn

from dust3r.inference import get_pred_pts3d, find_opt_scaling
from dust3r.utils.geometry import (
    inv,
    geotrf,
    normalize_pointcloud,
    normalize_pointcloud_group,
)
from dust3r.utils.geometry import (
    get_group_pointcloud_depth,
    get_group_pointcloud_center_scale,
    weighted_procrustes,
)
from gsplat import rasterization
import numpy as np
import lpips
from dust3r.utils.camera import (
    pose_encoding_to_camera,
    camera_to_pose_encoding,
    relative_pose_absT_quatR,
)


def Sum(*losses_and_masks):
    loss, mask = losses_and_masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        return losses_and_masks
    else:
        # we are returning the global loss
        for loss2, mask2 in losses_and_masks[1:]:
            loss = loss + loss2
        return loss


class BaseCriterion(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction


class LLoss(BaseCriterion):
    """L-norm loss"""

    def forward(self, a, b):
        assert (
            a.shape == b.shape and a.ndim >= 2 and 1 <= a.shape[-1] <= 3
        ), f"Bad shape = {a.shape}"
        dist = self.distance(a, b)
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(self, a, b):
        raise NotImplementedError()


class L21Loss(LLoss):
    """Euclidean distance between 3d points"""

    def distance(self, a, b):
        return torch.norm(a - b, dim=-1)  # normalized L2 distance


L21 = L21Loss()


class MSELoss(LLoss):
    def distance(self, a, b):
        return (a - b) ** 2


MSE = MSELoss()


class Criterion(nn.Module):
    def __init__(self, criterion=None):
        super().__init__()
        assert isinstance(
            criterion, BaseCriterion
        ), f"{criterion} is not a proper criterion!"
        self.criterion = copy(criterion)

    def get_name(self):
        return f"{type(self).__name__}({self.criterion})"

    def with_reduction(self, mode="none"):
        res = loss = deepcopy(self)
        while loss is not None:
            assert isinstance(loss, Criterion)
            loss.criterion.reduction = mode  # make it return the loss for each sample
            loss = loss._loss2  # we assume loss is a Multiloss
        return res


class MultiLoss(nn.Module):
    """Easily combinable losses (also keep track of individual loss values):
        loss = MyLoss1() + 0.1*MyLoss2()
    Usage:
        Inherit from this class and override get_name() and compute_loss()
    """

    def __init__(self):
        super().__init__()
        self._alpha = 1
        self._loss2 = None

    def compute_loss(self, *args, **kwargs):
        raise NotImplementedError()

    def get_name(self):
        raise NotImplementedError()

    def __mul__(self, alpha):
        assert isinstance(alpha, (int, float))
        res = copy(self)
        res._alpha = alpha
        return res

    __rmul__ = __mul__  # same

    def __add__(self, loss2):
        assert isinstance(loss2, MultiLoss)
        res = cur = copy(self)
        # find the end of the chain
        while cur._loss2 is not None:
            cur = cur._loss2
        cur._loss2 = loss2
        return res

    def __repr__(self):
        name = self.get_name()
        if self._alpha != 1:
            name = f"{self._alpha:g}*{name}"
        if self._loss2:
            name = f"{name} + {self._loss2}"
        return name

    def forward(self, *args, **kwargs):
        loss = self.compute_loss(*args, **kwargs)
        if isinstance(loss, tuple):
            loss, details = loss
        elif loss.ndim == 0:
            details = {self.get_name(): float(loss)}
        else:
            details = {}
        loss = loss * self._alpha

        if self._loss2:
            loss2, details2 = self._loss2(*args, **kwargs)
            loss = loss + loss2
            details |= details2

        return loss, details


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


class RGBLoss(Criterion, MultiLoss):
    def __init__(self, criterion):
        super().__init__(criterion)
        self.ssim = SSIM()

    def img_loss(self, a, b):
        return self.criterion(a, b)

    def compute_loss(self, gts, preds, **kw):
        gt_rgbs = [gt["img"].permute(0, 2, 3, 1) for gt in gts]
        pred_rgbs = [pred["rgb"] for pred in preds]
        ls = [
            self.img_loss(pred_rgb, gt_rgb)
            for pred_rgb, gt_rgb in zip(pred_rgbs, gt_rgbs)
        ]
        details = {}
        self_name = type(self).__name__
        for i, l in enumerate(ls):
            details[self_name + f"_rgb/{i+1}"] = float(l)
            details[f"pred_rgb_{i+1}"] = pred_rgbs[i]
        rgb_loss = sum(ls) / len(ls)
        return rgb_loss, details


class DepthScaleShiftInvLoss(BaseCriterion):
    """scale and shift invariant loss"""

    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, pred, gt, mask):
        assert pred.shape == gt.shape and pred.ndim == 3, f"Bad shape = {pred.shape}"
        dist = self.distance(pred, gt, mask)
        # assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def normalize(self, x, mask):
        x_valid = x[mask]
        splits = mask.sum(dim=(1, 2)).tolist()
        x_valid_list = torch.split(x_valid, splits)
        shift = [x.mean() for x in x_valid_list]
        x_valid_centered = [x - m for x, m in zip(x_valid_list, shift)]
        scale = [x.abs().mean() for x in x_valid_centered]
        scale = torch.stack(scale)
        shift = torch.stack(shift)
        x = (x - shift.view(-1, 1, 1)) / scale.view(-1, 1, 1).clamp(min=1e-6)
        return x

    def distance(self, pred, gt, mask):
        pred = self.normalize(pred, mask)
        gt = self.normalize(gt, mask)
        return torch.abs((pred - gt)[mask])


class ScaleInvLoss(BaseCriterion):
    """scale invariant loss"""

    def __init__(self, reduction="none"):
        super().__init__(reduction)

    def forward(self, pred, gt, mask):
        assert pred.shape == gt.shape and pred.ndim == 4, f"Bad shape = {pred.shape}"
        dist = self.distance(pred, gt, mask)
        # assert dist.ndim == a.ndim - 1  # one dimension less
        if self.reduction == "none":
            return dist
        if self.reduction == "sum":
            return dist.sum()
        if self.reduction == "mean":
            return dist.mean() if dist.numel() > 0 else dist.new_zeros(())
        raise ValueError(f"bad {self.reduction=} mode")

    def distance(self, pred, gt, mask):
        pred_norm_factor = (torch.norm(pred, dim=-1) * mask).sum(dim=(1, 2)) / mask.sum(
            dim=(1, 2)
        ).clamp(min=1e-6)
        gt_norm_factor = (torch.norm(gt, dim=-1) * mask).sum(dim=(1, 2)) / mask.sum(
            dim=(1, 2)
        ).clamp(min=1e-6)
        pred = pred / pred_norm_factor.view(-1, 1, 1, 1).clamp(min=1e-6)
        gt = gt / gt_norm_factor.view(-1, 1, 1, 1).clamp(min=1e-6)
        return torch.norm(pred - gt, dim=-1)[mask]


class Regr3DPose(Criterion, MultiLoss):
    """Ensure that all 3D points are correct.
    Asymmetric loss: view1 is supposed to be the anchor.

    P1 = RT1 @ D1
    P2 = RT2 @ D2
    loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
    loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
          = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(
        self,
        criterion,
        norm_mode="?avg_dis",
        gt_scale=False,
        sky_loss_value=2,
        max_metric_scale=False,
    ):
        super().__init__(criterion)
        if norm_mode.startswith("?"):
            # do no norm pts from metric scale datasets
            self.norm_all = False
            self.norm_mode = norm_mode[1:]
        else:
            self.norm_all = True
            self.norm_mode = norm_mode
        self.gt_scale = gt_scale

        self.sky_loss_value = sky_loss_value
        self.max_metric_scale = max_metric_scale

    def get_norm_factor_point_cloud(
        self, pts_self, pts_cross, valids, conf_self, conf_cross, norm_self_only=False
    ):
        if norm_self_only:
            norm_factor = normalize_pointcloud_group(
                pts_self, self.norm_mode, valids, conf_self, ret_factor_only=True
            )
        else:
            pts = [torch.cat([x, y], dim=2) for x, y in zip(pts_self, pts_cross)]
            valids = [torch.cat([x, x], dim=2) for x in valids]
            confs = [torch.cat([x, y], dim=2) for x, y in zip(conf_self, conf_cross)]
            norm_factor = normalize_pointcloud_group(
                pts, self.norm_mode, valids, confs, ret_factor_only=True
            )
        return norm_factor

    def get_norm_factor_poses(self, gt_trans, pr_trans, not_metric_mask):

        if self.norm_mode and not self.gt_scale:
            gt_trans = [x[:, None, None, :].clone() for x in gt_trans]
            valids = [torch.ones_like(x[..., 0], dtype=torch.bool) for x in gt_trans]
            norm_factor_gt = (
                normalize_pointcloud_group(
                    gt_trans,
                    self.norm_mode,
                    valids,
                    ret_factor_only=True,
                )
                .squeeze(-1)
                .squeeze(-1)
            )
        else:
            norm_factor_gt = torch.ones(
                len(gt_trans), dtype=gt_trans[0].dtype, device=gt_trans[0].device
            )

        norm_factor_pr = norm_factor_gt.clone()
        if self.norm_mode and not_metric_mask.sum() > 0 and not self.gt_scale:
            pr_trans_not_metric = [
                x[not_metric_mask][:, None, None, :].clone() for x in pr_trans
            ]
            valids = [
                torch.ones_like(x[..., 0], dtype=torch.bool)
                for x in pr_trans_not_metric
            ]
            norm_factor_pr_not_metric = (
                normalize_pointcloud_group(
                    pr_trans_not_metric,
                    self.norm_mode,
                    valids,
                    ret_factor_only=True,
                )
                .squeeze(-1)
                .squeeze(-1)
            )
            norm_factor_pr[not_metric_mask] = norm_factor_pr_not_metric
        return norm_factor_gt, norm_factor_pr

    def get_all_pts3d(
        self,
        gts,
        preds,
        dist_clip=None,
        norm_self_only=False,
        norm_pose_separately=False,
        eps=1e-3,
        camera1=None,
    ):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gts[0]["camera_pose"]) if camera1 is None else inv(camera1)
        gt_pts_self = [geotrf(inv(gt["camera_pose"]), gt["pts3d"]) for gt in gts]
        gt_pts_cross = [geotrf(in_camera1, gt["pts3d"]) for gt in gts]
        valids = [gt["valid_mask"].clone() for gt in gts]
        camera_only = gts[0]["camera_only"]

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis = [gt_pt.norm(dim=-1) for gt_pt in gt_pts_cross]
            valids = [valid & (dis <= dist_clip) for valid, dis in zip(valids, dis)]

        pr_pts_self = [pred["pts3d_in_self_view"] for pred in preds]
        pr_pts_cross = [pred["pts3d_in_other_view"] for pred in preds]
        conf_self = [torch.log(pred["conf_self"]).detach().clip(eps) for pred in preds]
        conf_cross = [torch.log(pred["conf"]).detach().clip(eps) for pred in preds]

        if not self.norm_all:
            if self.max_metric_scale:
                B = valids[0].shape[0]
                dist = [
                    torch.where(valid, torch.linalg.norm(gt_pt_cross, dim=-1), 0).view(
                        B, -1
                    )
                    for valid, gt_pt_cross in zip(valids, gt_pts_cross)
                ]
                for d in dist:
                    gts[0]["is_metric"] = gts[0]["is_metric_scale"] & (
                        d.max(dim=-1).values < self.max_metric_scale
                    )
            not_metric_mask = ~gts[0]["is_metric"]
        else:
            not_metric_mask = torch.ones_like(gts[0]["is_metric"])

        # normalize 3d points
        # compute the scale using only the self view point maps
        if self.norm_mode and not self.gt_scale:
            norm_factor_gt = self.get_norm_factor_point_cloud(
                gt_pts_self,
                gt_pts_cross,
                valids,
                conf_self,
                conf_cross,
                norm_self_only=norm_self_only,
            )
        else:
            norm_factor_gt = torch.ones_like(
                preds[0]["pts3d_in_other_view"][:, :1, :1, :1]
            )

        norm_factor_pr = norm_factor_gt.clone()
        if self.norm_mode and not_metric_mask.sum() > 0 and not self.gt_scale:
            norm_factor_pr_not_metric = self.get_norm_factor_point_cloud(
                [pr_pt_self[not_metric_mask] for pr_pt_self in pr_pts_self],
                [pr_pt_cross[not_metric_mask] for pr_pt_cross in pr_pts_cross],
                [valid[not_metric_mask] for valid in valids],
                [conf[not_metric_mask] for conf in conf_self],
                [conf[not_metric_mask] for conf in conf_cross],
                norm_self_only=norm_self_only,
            )
            norm_factor_pr[not_metric_mask] = norm_factor_pr_not_metric

        norm_factor_gt = norm_factor_gt.clip(eps)
        norm_factor_pr = norm_factor_pr.clip(eps)

        gt_pts_self = [pts / norm_factor_gt for pts in gt_pts_self]
        gt_pts_cross = [pts / norm_factor_gt for pts in gt_pts_cross]
        pr_pts_self = [pts / norm_factor_pr for pts in pr_pts_self]
        pr_pts_cross = [pts / norm_factor_pr for pts in pr_pts_cross]

        # [(Bx3, BX4), (BX3, BX4), ...], 3 for translation, 4 for quaternion
        gt_poses = [
            camera_to_pose_encoding(in_camera1 @ gt["camera_pose"]).clone()
            for gt in gts
        ]
        pr_poses = [pred["camera_pose"].clone() for pred in preds]
        pose_norm_factor_gt = norm_factor_gt.clone().squeeze(2, 3)
        pose_norm_factor_pr = norm_factor_pr.clone().squeeze(2, 3)

        if norm_pose_separately:
            gt_trans = [gt[:, :3] for gt in gt_poses]
            pr_trans = [pr[:, :3] for pr in pr_poses]
            pose_norm_factor_gt, pose_norm_factor_pr = self.get_norm_factor_poses(
                gt_trans, pr_trans, not_metric_mask
            )
        elif any(camera_only):
            gt_trans = [gt[:, :3] for gt in gt_poses]
            pr_trans = [pr[:, :3] for pr in pr_poses]
            pose_only_norm_factor_gt, pose_only_norm_factor_pr = (
                self.get_norm_factor_poses(gt_trans, pr_trans, not_metric_mask)
            )
            pose_norm_factor_gt = torch.where(
                camera_only[:, None], pose_only_norm_factor_gt, pose_norm_factor_gt
            )
            pose_norm_factor_pr = torch.where(
                camera_only[:, None], pose_only_norm_factor_pr, pose_norm_factor_pr
            )

        gt_poses = [
            (gt[:, :3] / pose_norm_factor_gt.clip(eps), gt[:, 3:]) for gt in gt_poses
        ]
        pr_poses = [
            (pr[:, :3] / pose_norm_factor_pr.clip(eps), pr[:, 3:]) for pr in pr_poses
        ]
        pose_masks = (pose_norm_factor_gt.squeeze() > eps) & (
            pose_norm_factor_pr.squeeze() > eps
        )

        if any(camera_only):
            # this is equal to a loss for camera intrinsics
            gt_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (gt / gt[..., -1:].clip(1e-6)).clip(-2, 2),
                    gt,
                )
                for gt in gt_pts_self
            ]
            pr_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (pr / pr[..., -1:].clip(1e-6)).clip(-2, 2),
                    pr,
                )
                for pr in pr_pts_self
            ]
            # # do not add cross view loss when there is only camera supervision

        skys = [gt["sky_mask"] & ~valid for gt, valid in zip(gts, valids)]
        return (
            gt_pts_self,
            gt_pts_cross,
            pr_pts_self,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            valids,
            skys,
            pose_masks,
            {},
        )

    def get_all_pts3d_with_scale_loss(
        self,
        gts,
        preds,
        dist_clip=None,
        norm_self_only=False,
        norm_pose_separately=False,
        eps=1e-3,
    ):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gts[0]["camera_pose"])
        gt_pts_self = [geotrf(inv(gt["camera_pose"]), gt["pts3d"]) for gt in gts]
        gt_pts_cross = [geotrf(in_camera1, gt["pts3d"]) for gt in gts]
        valids = [gt["valid_mask"].clone() for gt in gts]
        camera_only = gts[0]["camera_only"]

        if dist_clip is not None:
            # points that are too far-away == invalid
            dis = [gt_pt.norm(dim=-1) for gt_pt in gt_pts_cross]
            valids = [valid & (dis <= dist_clip) for valid, dis in zip(valids, dis)]

        pr_pts_self = [pred["pts3d_in_self_view"] for pred in preds]
        pr_pts_cross = [pred["pts3d_in_other_view"] for pred in preds]
        conf_self = [torch.log(pred["conf_self"]).detach().clip(eps) for pred in preds]
        conf_cross = [torch.log(pred["conf"]).detach().clip(eps) for pred in preds]

        if not self.norm_all:
            if self.max_metric_scale:
                B = valids[0].shape[0]
                dist = [
                    torch.where(valid, torch.linalg.norm(gt_pt_cross, dim=-1), 0).view(
                        B, -1
                    )
                    for valid, gt_pt_cross in zip(valids, gt_pts_cross)
                ]
                for d in dist:
                    gts[0]["is_metric"] = gts[0]["is_metric_scale"] & (
                        d.max(dim=-1).values < self.max_metric_scale
                    )
            not_metric_mask = ~gts[0]["is_metric"]
        else:
            not_metric_mask = torch.ones_like(gts[0]["is_metric"])

        # normalize 3d points
        # compute the scale using only the self view point maps
        if self.norm_mode and not self.gt_scale:
            norm_factor_gt = self.get_norm_factor_point_cloud(
                gt_pts_self[:1],
                gt_pts_cross[:1],
                valids[:1],
                conf_self[:1],
                conf_cross[:1],
                norm_self_only=norm_self_only,
            )
        else:
            norm_factor_gt = torch.ones_like(
                preds[0]["pts3d_in_other_view"][:, :1, :1, :1]
            )

        if self.norm_mode:
            norm_factor_pr = self.get_norm_factor_point_cloud(
                pr_pts_self[:1],
                pr_pts_cross[:1],
                valids[:1],
                conf_self[:1],
                conf_cross[:1],
                norm_self_only=norm_self_only,
            )
        else:
            raise NotImplementedError
        # only add loss to metric scale norm factor
        if (~not_metric_mask).sum() > 0:
            pts_scale_loss = torch.abs(
                norm_factor_pr[~not_metric_mask] - norm_factor_gt[~not_metric_mask]
            ).mean()
        else:
            pts_scale_loss = 0.0

        norm_factor_gt = norm_factor_gt.clip(eps)
        norm_factor_pr = norm_factor_pr.clip(eps)

        gt_pts_self = [pts / norm_factor_gt for pts in gt_pts_self]
        gt_pts_cross = [pts / norm_factor_gt for pts in gt_pts_cross]
        pr_pts_self = [pts / norm_factor_pr for pts in pr_pts_self]
        pr_pts_cross = [pts / norm_factor_pr for pts in pr_pts_cross]

        # [(Bx3, BX4), (BX3, BX4), ...], 3 for translation, 4 for quaternion
        gt_poses = [
            camera_to_pose_encoding(in_camera1 @ gt["camera_pose"]).clone()
            for gt in gts
        ]
        pr_poses = [pred["camera_pose"].clone() for pred in preds]
        pose_norm_factor_gt = norm_factor_gt.clone().squeeze(2, 3)
        pose_norm_factor_pr = norm_factor_pr.clone().squeeze(2, 3)

        if norm_pose_separately:
            gt_trans = [gt[:, :3] for gt in gt_poses][:1]
            pr_trans = [pr[:, :3] for pr in pr_poses][:1]
            pose_norm_factor_gt, pose_norm_factor_pr = self.get_norm_factor_poses(
                gt_trans, pr_trans, torch.ones_like(not_metric_mask)
            )
        elif any(camera_only):
            gt_trans = [gt[:, :3] for gt in gt_poses][:1]
            pr_trans = [pr[:, :3] for pr in pr_poses][:1]
            pose_only_norm_factor_gt, pose_only_norm_factor_pr = (
                self.get_norm_factor_poses(
                    gt_trans, pr_trans, torch.ones_like(not_metric_mask)
                )
            )
            pose_norm_factor_gt = torch.where(
                camera_only[:, None], pose_only_norm_factor_gt, pose_norm_factor_gt
            )
            pose_norm_factor_pr = torch.where(
                camera_only[:, None], pose_only_norm_factor_pr, pose_norm_factor_pr
            )
        # only add loss to metric scale norm factor
        if (~not_metric_mask).sum() > 0:
            pose_scale_loss = torch.abs(
                pose_norm_factor_pr[~not_metric_mask]
                - pose_norm_factor_gt[~not_metric_mask]
            ).mean()
        else:
            pose_scale_loss = 0.0
        gt_poses = [
            (gt[:, :3] / pose_norm_factor_gt.clip(eps), gt[:, 3:]) for gt in gt_poses
        ]
        pr_poses = [
            (pr[:, :3] / pose_norm_factor_pr.clip(eps), pr[:, 3:]) for pr in pr_poses
        ]

        pose_masks = (pose_norm_factor_gt.squeeze() > eps) & (
            pose_norm_factor_pr.squeeze() > eps
        )

        if any(camera_only):
            # this is equal to a loss for camera intrinsics
            gt_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (gt / gt[..., -1:].clip(1e-6)).clip(-2, 2),
                    gt,
                )
                for gt in gt_pts_self
            ]
            pr_pts_self = [
                torch.where(
                    camera_only[:, None, None, None],
                    (pr / pr[..., -1:].clip(1e-6)).clip(-2, 2),
                    pr,
                )
                for pr in pr_pts_self
            ]
            # # do not add cross view loss when there is only camera supervision

        skys = [gt["sky_mask"] & ~valid for gt, valid in zip(gts, valids)]
        return (
            gt_pts_self,
            gt_pts_cross,
            pr_pts_self,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            valids,
            skys,
            pose_masks,
            {"scale_loss": pose_scale_loss + pts_scale_loss},
        )

    def compute_relative_pose_loss(
        self, gt_trans, gt_quats, pr_trans, pr_quats, masks=None
    ):
        if masks is None:
            masks = torch.ones(len(gt_trans), dtype=torch.bool, device=gt_trans.device)
        gt_trans_matrix1 = gt_trans[:, :, None, :].repeat(1, 1, gt_trans.shape[1], 1)[
            masks
        ]
        gt_trans_matrix2 = gt_trans[:, None, :, :].repeat(1, gt_trans.shape[1], 1, 1)[
            masks
        ]
        gt_quats_matrix1 = gt_quats[:, :, None, :].repeat(1, 1, gt_quats.shape[1], 1)[
            masks
        ]
        gt_quats_matrix2 = gt_quats[:, None, :, :].repeat(1, gt_quats.shape[1], 1, 1)[
            masks
        ]
        pr_trans_matrix1 = pr_trans[:, :, None, :].repeat(1, 1, pr_trans.shape[1], 1)[
            masks
        ]
        pr_trans_matrix2 = pr_trans[:, None, :, :].repeat(1, pr_trans.shape[1], 1, 1)[
            masks
        ]
        pr_quats_matrix1 = pr_quats[:, :, None, :].repeat(1, 1, pr_quats.shape[1], 1)[
            masks
        ]
        pr_quats_matrix2 = pr_quats[:, None, :, :].repeat(1, pr_quats.shape[1], 1, 1)[
            masks
        ]

        gt_rel_trans, gt_rel_quats = relative_pose_absT_quatR(
            gt_trans_matrix1, gt_quats_matrix1, gt_trans_matrix2, gt_quats_matrix2
        )
        pr_rel_trans, pr_rel_quats = relative_pose_absT_quatR(
            pr_trans_matrix1, pr_quats_matrix1, pr_trans_matrix2, pr_quats_matrix2
        )
        rel_trans_err = torch.norm(gt_rel_trans - pr_rel_trans, dim=-1)
        rel_quats_err = torch.norm(gt_rel_quats - pr_rel_quats, dim=-1)
        return rel_trans_err.mean() + rel_quats_err.mean()

    def compute_pose_loss(self, gt_poses, pred_poses, masks=None):
        """
        gt_pose: list of (Bx3, Bx4)
        pred_pose: list of (Bx3, Bx4)
        masks: None, or B
        """
        gt_trans = torch.stack([gt[0] for gt in gt_poses], dim=1)  # BxNx3
        gt_quats = torch.stack([gt[1] for gt in gt_poses], dim=1)  # BXNX3
        pred_trans = torch.stack([pr[0] for pr in pred_poses], dim=1)  # BxNx4
        pred_quats = torch.stack([pr[1] for pr in pred_poses], dim=1)  # BxNx4
        if masks == None:
            pose_loss = (
                torch.norm(pred_trans - gt_trans, dim=-1).mean()
                + torch.norm(pred_quats - gt_quats, dim=-1).mean()
            )
        else:
            if not any(masks):
                return torch.tensor(0.0)
            pose_loss = (
                torch.norm(pred_trans - gt_trans, dim=-1)[masks].mean()
                + torch.norm(pred_quats - gt_quats, dim=-1)[masks].mean()
            )

        return pose_loss

    def compute_loss(self, gts, preds, **kw):
        (
            gt_pts_self,
            gt_pts_cross,
            pred_pts_self,
            pred_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        ) = self.get_all_pts3d(gts, preds, **kw)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            masks = [mask | sky for mask, sky in zip(masks, skys)]

        # self view loss and details
        if "Quantile" in self.criterion.__class__.__name__:
            # masks are overwritten taking into account self view losses
            ls_self, masks = self.criterion(
                pred_pts_self, gt_pts_self, masks, gts[0]["quantile"]
            )
        else:
            ls_self = [
                self.criterion(pred_pt[mask], gt_pt[mask])
                for pred_pt, gt_pt, mask in zip(pred_pts_self, gt_pts_self, masks)
            ]

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            for i, l in enumerate(ls_self):
                ls_self[i] = torch.where(skys[i][masks[i]], self.sky_loss_value, l)

        self_name = type(self).__name__

        details = {}
        for i in range(len(ls_self)):
            details[self_name + f"_self_pts3d/{i+1}"] = float(ls_self[i].mean())
            details[f"gt_img{i+1}"] = gts[i]["img"].permute(0, 2, 3, 1).detach()
            details[f"self_conf_{i+1}"] = preds[i]["conf_self"].detach()
            details[f"valid_mask_{i+1}"] = masks[i].detach()

            if "img_mask" in gts[i] and "ray_mask" in gts[i]:
                details[f"img_mask_{i+1}"] = gts[i]["img_mask"].detach()
                details[f"ray_mask_{i+1}"] = gts[i]["ray_mask"].detach()

            if "desc" in preds[i]:
                details[f"desc_{i+1}"] = preds[i]["desc"].detach()

        # cross view loss and details
        camera_only = gts[0]["camera_only"]
        pred_pts_cross = [pred_pts[~camera_only] for pred_pts in pred_pts_cross]
        gt_pts_cross = [gt_pts[~camera_only] for gt_pts in gt_pts_cross]
        masks_cross = [mask[~camera_only] for mask in masks]
        skys_cross = [sky[~camera_only] for sky in skys]

        if "Quantile" in self.criterion.__class__.__name__:
            # quantile masks have already been determined by self view losses, here pass in None as quantile
            ls_cross, _ = self.criterion(
                pred_pts_cross, gt_pts_cross, masks_cross, None
            )
        else:
            ls_cross = [
                self.criterion(pred_pt[mask], gt_pt[mask])
                for pred_pt, gt_pt, mask in zip(
                    pred_pts_cross, gt_pts_cross, masks_cross
                )
            ]

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            for i, l in enumerate(ls_cross):
                ls_cross[i] = torch.where(
                    skys_cross[i][masks_cross[i]], self.sky_loss_value, l
                )

        for i in range(len(ls_cross)):
            details[self_name + f"_pts3d/{i+1}"] = float(
                ls_cross[i].mean() if ls_cross[i].numel() > 0 else 0
            )
            details[f"conf_{i+1}"] = preds[i]["conf"].detach()

        ls = ls_self + ls_cross
        masks = masks + masks_cross
        details["is_self"] = [True] * len(ls_self) + [False] * len(ls_cross)
        details["img_ids"] = (
            np.arange(len(ls_self)).tolist() + np.arange(len(ls_cross)).tolist()
        )
        details["pose_loss"] = self.compute_pose_loss(gt_poses, pr_poses, pose_masks)

        return Sum(*list(zip(ls, masks))), (details | monitoring)


class Regr3DPoseBatchList(Regr3DPose):
    """Ensure that all 3D points are correct.
    Asymmetric loss: view1 is supposed to be the anchor.

    P1 = RT1 @ D1
    P2 = RT2 @ D2
    loss1 = (I @ pred_D1) - (RT1^-1 @ RT1 @ D1)
    loss2 = (RT21 @ pred_D2) - (RT1^-1 @ P2)
          = (RT21 @ pred_D2) - (RT1^-1 @ RT2 @ D2)
    """

    def __init__(
        self,
        criterion,
        norm_mode="?avg_dis",
        gt_scale=False,
        sky_loss_value=2,
        max_metric_scale=False,
    ):
        super().__init__(
            criterion, norm_mode, gt_scale, sky_loss_value, max_metric_scale
        )
        self.depth_only_criterion = DepthScaleShiftInvLoss()
        self.single_view_criterion = ScaleInvLoss()

    def reorg(self, ls_b, masks_b):
        ids_split = [mask.sum(dim=(1, 2)) for mask in masks_b]
        ls = [[] for _ in range(len(masks_b[0]))]
        for i in range(len(ls_b)):
            ls_splitted_i = torch.split(ls_b[i], ids_split[i].tolist())
            for j in range(len(masks_b[0])):
                ls[j].append(ls_splitted_i[j])
        ls = [torch.cat(l) for l in ls]
        return ls

    def compute_loss(self, gts, preds, **kw):
        (
            gt_pts_self,
            gt_pts_cross,
            pred_pts_self,
            pred_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        ) = self.get_all_pts3d(gts, preds, **kw)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            masks = [mask | sky for mask, sky in zip(masks, skys)]

        camera_only = gts[0]["camera_only"]
        depth_only = gts[0]["depth_only"]
        single_view = gts[0]["single_view"]
        is_metric = gts[0]["is_metric"]

        # self view loss and details
        if "Quantile" in self.criterion.__class__.__name__:
            raise NotImplementedError
        else:
            # list [(B, h, w, 3)] x num_views -> list [num_views, h, w, 3] x B
            gt_pts_self_b = torch.unbind(torch.stack(gt_pts_self, dim=1), dim=0)
            pred_pts_self_b = torch.unbind(torch.stack(pred_pts_self, dim=1), dim=0)
            masks_b = torch.unbind(torch.stack(masks, dim=1), dim=0)
            ls_self_b = []
            for i in range(len(gt_pts_self_b)):
                if depth_only[
                    i
                ]:  # if only have relative depth, no intrinsics or anything
                    ls_self_b.append(
                        self.depth_only_criterion(
                            pred_pts_self_b[i][..., -1],
                            gt_pts_self_b[i][..., -1],
                            masks_b[i],
                        )
                    )
                elif (
                    single_view[i] and not is_metric[i]
                ):  # if single view, with intrinsics and not metric
                    ls_self_b.append(
                        self.single_view_criterion(
                            pred_pts_self_b[i], gt_pts_self_b[i], masks_b[i]
                        )
                    )
                else:  # if multiple view, or metric single view
                    ls_self_b.append(
                        self.criterion(
                            pred_pts_self_b[i][masks_b[i]], gt_pts_self_b[i][masks_b[i]]
                        )
                    )
            ls_self = self.reorg(ls_self_b, masks_b)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            for i, l in enumerate(ls_self):
                ls_self[i] = torch.where(skys[i][masks[i]], self.sky_loss_value, l)

        self_name = type(self).__name__

        details = {}
        for i in range(len(ls_self)):
            details[self_name + f"_self_pts3d/{i+1}"] = float(ls_self[i].mean())
            details[f"self_conf_{i+1}"] = preds[i]["conf_self"].detach()
            details[f"gt_img{i+1}"] = gts[i]["img"].permute(0, 2, 3, 1).detach()
            details[f"valid_mask_{i+1}"] = masks[i].detach()

            if "img_mask" in gts[i] and "ray_mask" in gts[i]:
                details[f"img_mask_{i+1}"] = gts[i]["img_mask"].detach()
                details[f"ray_mask_{i+1}"] = gts[i]["ray_mask"].detach()

            if "desc" in preds[i]:
                details[f"desc_{i+1}"] = preds[i]["desc"].detach()

        if "Quantile" in self.criterion.__class__.__name__:
            # quantile masks have already been determined by self view losses, here pass in None as quantile
            raise NotImplementedError
        else:
            gt_pts_cross_b = torch.unbind(
                torch.stack(gt_pts_cross, dim=1)[~camera_only], dim=0
            )
            pred_pts_cross_b = torch.unbind(
                torch.stack(pred_pts_cross, dim=1)[~camera_only], dim=0
            )
            masks_cross_b = torch.unbind(torch.stack(masks, dim=1)[~camera_only], dim=0)
            ls_cross_b = []
            for i in range(len(gt_pts_cross_b)):
                if depth_only[~camera_only][i]:
                    ls_cross_b.append(
                        self.depth_only_criterion(
                            pred_pts_cross_b[i][..., -1],
                            gt_pts_cross_b[i][..., -1],
                            masks_cross_b[i],
                        )
                    )
                elif single_view[~camera_only][i] and not is_metric[~camera_only][i]:
                    ls_cross_b.append(
                        self.single_view_criterion(
                            pred_pts_cross_b[i], gt_pts_cross_b[i], masks_cross_b[i]
                        )
                    )
                else:
                    ls_cross_b.append(
                        self.criterion(
                            pred_pts_cross_b[i][masks_cross_b[i]],
                            gt_pts_cross_b[i][masks_cross_b[i]],
                        )
                    )
            ls_cross = self.reorg(ls_cross_b, masks_cross_b)

        if self.sky_loss_value > 0:
            assert (
                self.criterion.reduction == "none"
            ), "sky_loss_value should be 0 if no conf loss"
            masks_cross = [mask[~camera_only] for mask in masks]
            skys_cross = [sky[~camera_only] for sky in skys]
            for i, l in enumerate(ls_cross):
                ls_cross[i] = torch.where(
                    skys_cross[i][masks_cross[i]], self.sky_loss_value, l
                )

        for i in range(len(ls_cross)):
            details[self_name + f"_pts3d/{i+1}"] = float(
                ls_cross[i].mean() if ls_cross[i].numel() > 0 else 0
            )
            details[f"conf_{i+1}"] = preds[i]["conf"].detach()

        ls = ls_self + ls_cross
        masks = masks + masks_cross
        details["is_self"] = [True] * len(ls_self) + [False] * len(ls_cross)
        details["img_ids"] = (
            np.arange(len(ls_self)).tolist() + np.arange(len(ls_cross)).tolist()
        )
        pose_masks = pose_masks * gts[i]["img_mask"]
        details["pose_loss"] = self.compute_pose_loss(gt_poses, pr_poses, pose_masks)

        return Sum(*list(zip(ls, masks))), (details | monitoring)


class ConfLoss(MultiLoss):
    """Weighted regression by learned confidence.
        Assuming the input pixel_loss is a pixel-level regression loss.

    Principle:
        high-confidence means high conf = 0.1 ==> conf_loss = x / 10 + alpha*log(10)
        low  confidence means low  conf = 10  ==> conf_loss = x * 10 - alpha*log(10)

        alpha: hyperparameter
    """

    def __init__(self, pixel_loss, alpha=1):
        super().__init__()
        assert alpha > 0
        self.alpha = alpha
        self.pixel_loss = pixel_loss.with_reduction("none")

    def get_name(self):
        return f"ConfLoss({self.pixel_loss})"

    def get_conf_log(self, x):
        return x, torch.log(x)

    def compute_loss(self, gts, preds, **kw):
        # compute per-pixel loss
        losses_and_masks, details = self.pixel_loss(gts, preds, **kw)
        if "is_self" in details and "img_ids" in details:
            is_self = details["is_self"]
            img_ids = details["img_ids"]
        else:
            is_self = [False] * len(losses_and_masks)
            img_ids = list(range(len(losses_and_masks)))

        # weight by confidence
        conf_losses = []

        for i in range(len(losses_and_masks)):
            pred = preds[img_ids[i]]
            conf_key = "conf_self" if is_self[i] else "conf"
            if not is_self[i]:
                camera_only = gts[0]["camera_only"]
                conf, log_conf = self.get_conf_log(
                    pred[conf_key][~camera_only][losses_and_masks[i][1]]
                )
            else:
                conf, log_conf = self.get_conf_log(
                    pred[conf_key][losses_and_masks[i][1]]
                )

            conf_loss = losses_and_masks[i][0] * conf - self.alpha * log_conf
            conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0
            conf_losses.append(conf_loss)

            if is_self[i]:
                details[self.get_name() + f"_conf_loss_self/{img_ids[i]+1}"] = float(
                    conf_loss
                )
            else:
                details[self.get_name() + f"_conf_loss/{img_ids[i]+1}"] = float(
                    conf_loss
                )

        details.pop("is_self", None)
        details.pop("img_ids", None)

        final_loss = sum(conf_losses) / len(conf_losses) * 2.0
        if "pose_loss" in details:
            final_loss = (
                final_loss + details["pose_loss"] #.clip(max=0.3) * 5.0
            )  # , details
        if "scale_loss" in details:
            final_loss = final_loss + details["scale_loss"]
        return final_loss, details


class Regr3DPose_ScaleInv(Regr3DPose):
    """Same than Regr3D but invariant to depth shift.
    if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d(self, gts, preds):
        # compute depth-normalized points
        (
            gt_pts_self,
            gt_pts_cross,
            pr_pts_self,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        ) = super().get_all_pts3d(gts, preds)

        # measure scene scale
        _, gt_scale_self = get_group_pointcloud_center_scale(gt_pts_self, masks)
        _, pred_scale_self = get_group_pointcloud_center_scale(pr_pts_self, masks)

        _, gt_scale_cross = get_group_pointcloud_center_scale(gt_pts_cross, masks)
        _, pred_scale_cross = get_group_pointcloud_center_scale(pr_pts_cross, masks)

        # prevent predictions to be in a ridiculous range
        pred_scale_self = pred_scale_self.clip(min=1e-3, max=1e3)
        pred_scale_cross = pred_scale_cross.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            pr_pts_self = [
                pr_pt_self * gt_scale_self / pred_scale_self
                for pr_pt_self in pr_pts_self
            ]
            pr_pts_cross = [
                pr_pt_cross * gt_scale_cross / pred_scale_cross
                for pr_pt_cross in pr_pts_cross
            ]
        else:
            gt_pts_self = [gt_pt_self / gt_scale_self for gt_pt_self in gt_pts_self]
            gt_pts_cross = [
                gt_pt_cross / gt_scale_cross for gt_pt_cross in gt_pts_cross
            ]
            pr_pts_self = [pr_pt_self / pred_scale_self for pr_pt_self in pr_pts_self]
            pr_pts_cross = [
                pr_pt_cross / pred_scale_cross for pr_pt_cross in pr_pts_cross
            ]

        return (
            gt_pts_self,
            gt_pts_cross,
            pr_pts_self,
            pr_pts_cross,
            gt_poses,
            pr_poses,
            masks,
            skys,
            pose_masks,
            monitoring,
        )
