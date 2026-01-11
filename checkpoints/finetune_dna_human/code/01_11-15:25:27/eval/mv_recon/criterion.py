import torch
import torch.nn as nn
from copy import copy, deepcopy
from dust3r.utils.misc import invalid_to_zeros, invalid_to_nans
from dust3r.utils.geometry import inv, geotrf, depthmap_to_pts3d
from dust3r.utils.camera import pose_encoding_to_camera


class BaseCriterion(nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction


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


def get_pred_pts3d(gt, pred, use_pose=False):
    if "depth" in pred and "pseudo_focal" in pred:
        try:
            pp = gt["camera_intrinsics"][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif "pts3d" in pred:
        # pts3d from my camera
        pts3d = pred["pts3d"]

    elif "pts3d_in_other_view" in pred:
        # pts3d from the other camera, already transformed
        assert use_pose is True
        return pred["pts3d_in_other_view"]  # return!

    if use_pose:
        camera_pose = pred.get("camera_pose")
        pts3d = pred.get("pts3d_in_self_view")
        assert camera_pose is not None
        assert pts3d is not None
        pts3d = geotrf(pose_encoding_to_camera(camera_pose), pts3d)

    return pts3d


def Sum(losses, masks, conf=None):
    loss, mask = losses[0], masks[0]
    if loss.ndim > 0:
        # we are actually returning the loss for every pixels
        if conf is not None:
            return losses, masks, conf
        return losses, masks
    else:
        # we are returning the global loss
        for loss2 in losses[1:]:
            loss = loss + loss2
        return loss


def get_norm_factor(pts, norm_mode="avg_dis", valids=None, fix_first=True):
    assert pts[0].ndim >= 3 and pts[0].shape[-1] == 3
    assert pts[1] is None or (pts[1].ndim >= 3 and pts[1].shape[-1] == 3)
    norm_mode, dis_mode = norm_mode.split("_")

    nan_pts = []
    nnzs = []

    if norm_mode == "avg":
        # gather all points together (joint normalization)

        for i, pt in enumerate(pts):
            nan_pt, nnz = invalid_to_zeros(pt, valids[i], ndim=3)
            nan_pts.append(nan_pt)
            nnzs.append(nnz)

            if fix_first:
                break
        all_pts = torch.cat(nan_pts, dim=1)

        # compute distance to origin
        all_dis = all_pts.norm(dim=-1)
        if dis_mode == "dis":
            pass  # do nothing
        elif dis_mode == "log1p":
            all_dis = torch.log1p(all_dis)
        else:
            raise ValueError(f"bad {dis_mode=}")

        norm_factor = all_dis.sum(dim=1) / (torch.cat(nnzs).sum() + 1e-8)
    else:
        raise ValueError(f"Not implemented {norm_mode=}")

    norm_factor = norm_factor.clip(min=1e-8)
    while norm_factor.ndim < pts[0].ndim:
        norm_factor.unsqueeze_(-1)

    return norm_factor


def normalize_pointcloud_t(
    pts, norm_mode="avg_dis", valids=None, fix_first=True, gt=False
):
    if gt:
        norm_factor = get_norm_factor(pts, norm_mode, valids, fix_first)
        res = []

        for i, pt in enumerate(pts):
            res.append(pt / norm_factor)

    else:
        # pts_l, pts_r = pts
        # use pts_l and pts_r[-1] as pts to normalize
        norm_factor = get_norm_factor(pts, norm_mode, valids, fix_first)

        res = []

        for i in range(len(pts)):
            res.append(pts[i] / norm_factor)
            # res_r.append(pts_r[i] / norm_factor)

        # res = [res_l, res_r]

    return res, norm_factor


@torch.no_grad()
def get_joint_pointcloud_depth(zs, valid_masks=None, quantile=0.5):
    # set invalid points to NaN
    _zs = []
    for i in range(len(zs)):
        valid_mask = valid_masks[i] if valid_masks is not None else None
        _z = invalid_to_nans(zs[i], valid_mask).reshape(len(zs[i]), -1)
        _zs.append(_z)

    _zs = torch.cat(_zs, dim=-1)

    # compute median depth overall (ignoring nans)
    if quantile == 0.5:
        shift_z = torch.nanmedian(_zs, dim=-1).values
    else:
        shift_z = torch.nanquantile(_zs, quantile, dim=-1)
    return shift_z  # (B,)


@torch.no_grad()
def get_joint_pointcloud_center_scale(pts, valid_masks=None, z_only=False, center=True):
    # set invalid points to NaN

    _pts = []
    for i in range(len(pts)):
        valid_mask = valid_masks[i] if valid_masks is not None else None
        _pt = invalid_to_nans(pts[i], valid_mask).reshape(len(pts[i]), -1, 3)
        _pts.append(_pt)

    _pts = torch.cat(_pts, dim=1)

    # compute median center
    _center = torch.nanmedian(_pts, dim=1, keepdim=True).values  # (B,1,3)
    if z_only:
        _center[..., :2] = 0  # do not center X and Y

    # compute median norm
    _norm = ((_pts - _center) if center else _pts).norm(dim=-1)
    scale = torch.nanmedian(_norm, dim=1).values
    return _center[:, None, :, :], scale[:, None, None, None]


class Regr3D_t(Criterion, MultiLoss):
    def __init__(self, criterion, norm_mode="avg_dis", gt_scale=False, fix_first=True):
        super().__init__(criterion)
        self.norm_mode = norm_mode
        self.gt_scale = gt_scale
        self.fix_first = fix_first

    def get_all_pts3d_t(self, gts, preds, dist_clip=None):
        # everything is normalized w.r.t. camera of view1
        in_camera1 = inv(gts[0]["camera_pose"])

        gt_pts = []
        valids = []
        pr_pts = []

        for i, gt in enumerate(gts):
            # in_camera1: Bs, 4, 4 gt['pts3d']: Bs, H, W, 3
            gt_pts.append(geotrf(in_camera1, gt["pts3d"]))

            valid = gt["valid_mask"].clone()

            if dist_clip is not None:
                # points that are too far-away == invalid
                dis = gt["pts3d"].norm(dim=-1)
                valid = valid & (dis <= dist_clip)

            valids.append(valid)
            pr_pts.append(get_pred_pts3d(gt, preds[i], use_pose=True))
            # if i != len(gts)-1:
            #     pr_pts_l.append(get_pred_pts3d(gt, preds[i][0], use_pose=(i!=0)))

            # if i != 0:
            #     pr_pts_r.append(get_pred_pts3d(gt, preds[i-1][1], use_pose=(i!=0)))

        # pr_pts = (pr_pts_l, pr_pts_r)

        if self.norm_mode:
            pr_pts, pr_factor = normalize_pointcloud_t(
                pr_pts, self.norm_mode, valids, fix_first=self.fix_first, gt=False
            )
        else:
            pr_factor = None

        if self.norm_mode and not self.gt_scale:
            gt_pts, gt_factor = normalize_pointcloud_t(
                gt_pts, self.norm_mode, valids, fix_first=self.fix_first, gt=True
            )
        else:
            gt_factor = None

        return gt_pts, pr_pts, gt_factor, pr_factor, valids, {}

    def compute_frame_loss(self, gts, preds, **kw):
        gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
            self.get_all_pts3d_t(gts, preds, **kw)
        )

        pred_pts_l, pred_pts_r = pred_pts

        loss_all = []
        mask_all = []
        conf_all = []

        loss_left = 0
        loss_right = 0
        pred_conf_l = 0
        pred_conf_r = 0

        for i in range(len(gt_pts)):

            # Left (Reference)
            if i != len(gt_pts) - 1:
                frame_loss = self.criterion(
                    pred_pts_l[i][masks[i]], gt_pts[i][masks[i]]
                )

                loss_all.append(frame_loss)
                mask_all.append(masks[i])
                conf_all.append(preds[i][0]["conf"])

                # To compare target/reference loss
                if i != 0:
                    loss_left += frame_loss.cpu().detach().numpy().mean()
                    pred_conf_l += preds[i][0]["conf"].cpu().detach().numpy().mean()

            # Right (Target)
            if i != 0:
                frame_loss = self.criterion(
                    pred_pts_r[i - 1][masks[i]], gt_pts[i][masks[i]]
                )

                loss_all.append(frame_loss)
                mask_all.append(masks[i])
                conf_all.append(preds[i - 1][1]["conf"])

                # To compare target/reference loss
                if i != len(gt_pts) - 1:
                    loss_right += frame_loss.cpu().detach().numpy().mean()
                    pred_conf_r += preds[i - 1][1]["conf"].cpu().detach().numpy().mean()

        if pr_factor is not None and gt_factor is not None:
            filter_factor = pr_factor[pr_factor > gt_factor]
        else:
            filter_factor = []

        if len(filter_factor) > 0:
            factor_loss = (filter_factor - gt_factor).abs().mean()
        else:
            factor_loss = 0.0

        self_name = type(self).__name__
        details = {
            self_name + "_pts3d_1": float(loss_all[0].mean()),
            self_name + "_pts3d_2": float(loss_all[1].mean()),
            self_name + "loss_left": float(loss_left),
            self_name + "loss_right": float(loss_right),
            self_name + "conf_left": float(pred_conf_l),
            self_name + "conf_right": float(pred_conf_r),
        }

        return Sum(loss_all, mask_all, conf_all), (details | monitoring), factor_loss


class ConfLoss_t(MultiLoss):
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

    def compute_frame_loss(self, gts, preds, **kw):
        # compute per-pixel loss
        (losses, masks, confs), details, loss_factor = (
            self.pixel_loss.compute_frame_loss(gts, preds, **kw)
        )

        # weight by confidence
        conf_losses = []
        conf_sum = 0
        for i in range(len(losses)):
            conf, log_conf = self.get_conf_log(confs[i][masks[i]])
            conf_sum += conf.mean()
            conf_loss = losses[i] * conf - self.alpha * log_conf
            conf_loss = conf_loss.mean() if conf_loss.numel() > 0 else 0
            conf_losses.append(conf_loss)

        conf_losses = torch.stack(conf_losses) * 2.0
        conf_loss_mean = conf_losses.mean()

        return (
            conf_loss_mean,
            dict(
                conf_loss_1=float(conf_losses[0]),
                conf_loss2=float(conf_losses[1]),
                conf_mean=conf_sum / len(losses),
                **details,
            ),
            loss_factor,
        )


class Regr3D_t_ShiftInv(Regr3D_t):
    """Same than Regr3D but invariant to depth shift."""

    def get_all_pts3d_t(self, gts, preds):
        # compute unnormalized points
        gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
            super().get_all_pts3d_t(gts, preds)
        )

        # pred_pts_l, pred_pts_r = pred_pts
        gt_zs = [gt_pt[..., 2] for gt_pt in gt_pts]

        pred_zs = [pred_pt[..., 2] for pred_pt in pred_pts]
        # pred_zs.append(pred_pts_r[-1][..., 2])

        # compute median depth
        gt_shift_z = get_joint_pointcloud_depth(gt_zs, masks)[:, None, None]
        pred_shift_z = get_joint_pointcloud_depth(pred_zs, masks)[:, None, None]

        # subtract the median depth
        for i in range(len(gt_pts)):
            gt_pts[i][..., 2] -= gt_shift_z

        for i in range(len(pred_pts)):
            # for j in range(len(pred_pts[i])):
            pred_pts[i][..., 2] -= pred_shift_z

        monitoring = dict(
            monitoring,
            gt_shift_z=gt_shift_z.mean().detach(),
            pred_shift_z=pred_shift_z.mean().detach(),
        )
        return gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring


class Regr3D_t_ScaleInv(Regr3D_t):
    """Same than Regr3D but invariant to depth shift.
    if gt_scale == True: enforce the prediction to take the same scale than GT
    """

    def get_all_pts3d_t(self, gts, preds):
        # compute depth-normalized points
        gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = (
            super().get_all_pts3d_t(gts, preds)
        )

        # measure scene scale

        # pred_pts_l, pred_pts_r = pred_pts

        pred_pts_all = [
            x.clone() for x in pred_pts
        ]  # [pred_pt for pred_pt in pred_pts_l]
        # pred_pts_all.append(pred_pts_r[-1])

        _, gt_scale = get_joint_pointcloud_center_scale(gt_pts, masks)
        _, pred_scale = get_joint_pointcloud_center_scale(pred_pts_all, masks)

        # prevent predictions to be in a ridiculous range
        pred_scale = pred_scale.clip(min=1e-3, max=1e3)

        # subtract the median depth
        if self.gt_scale:
            for i in range(len(pred_pts)):
                # for j in range(len(pred_pts[i])):
                pred_pts[i] *= gt_scale / pred_scale

        else:
            for i in range(len(pred_pts)):
                # for j in range(len(pred_pts[i])):
                pred_pts[i] *= pred_scale / gt_scale

            for i in range(len(gt_pts)):
                gt_pts[i] *= gt_scale / pred_scale

        monitoring = dict(
            monitoring, gt_scale=gt_scale.mean(), pred_scale=pred_scale.mean().detach()
        )

        return gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring


class Regr3D_t_ScaleShiftInv(Regr3D_t_ScaleInv, Regr3D_t_ShiftInv):
    # calls Regr3D_ShiftInv first, then Regr3D_ScaleInv
    pass
