import tqdm
import torch
from dust3r.utils.device import to_cpu, collate_with_cat
from dust3r.utils.misc import invalid_to_nans
from dust3r.utils.geometry import depthmap_to_pts3d, geotrf
from dust3r.model import ARCroco3DStereo
from accelerate import Accelerator
import re


def custom_sort_key(key):
    text = key.split("/")
    if len(text) > 1:
        text, num = text[0], text[-1]
        return (text, int(num))
    else:
        return (key, -1)


def merge_chunk_dict(old_dict, curr_dict, add_number):
    new_dict = {}
    for key, value in curr_dict.items():

        match = re.search(r"(\d+)$", key)
        if match:

            num_part = int(match.group()) + add_number

            new_key = re.sub(r"(\d+)$", str(num_part), key, 1)
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    new_dict = old_dict | new_dict
    return {k: new_dict[k] for k in sorted(new_dict.keys(), key=custom_sort_key)}


def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res


def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2


def loss_of_one_batch(
    batch,
    model,
    criterion,
    accelerator: Accelerator,
    symmetrize_batch=False,
    use_amp=False,
    ret=None,
    img_mask=None,
    inference=False,
    teacher_model=None,
    distill_bg_weight=0.0,
):
    if len(batch) > 2:
        assert (
            symmetrize_batch is False
        ), "cannot symmetrize batch with more than 2 views"
    if symmetrize_batch:
        batch = make_batch_symmetric(batch)

    # Generic fix: ensure input images match model dtype
    target_dtype = torch.bfloat16
    device = accelerator.device
    for view in batch:
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                dtype = target_dtype if k in ["img", "ray_map"] else None
                view[k] = v.to(device=device, dtype=dtype)


    with torch.autocast("cuda", dtype=torch.bfloat16):
        if inference:
            output, state_args = model(batch, ret_state=True)
            preds, batch = output.ress, output.views
            result = dict(views=batch, pred=preds)
            return result[ret] if ret else result, state_args
        else:
            output = model(batch)
            preds, batch = output.ress, output.views

        with torch.cuda.amp.autocast(enabled=False):
            loss = criterion(batch, preds) if criterion is not None else None

            if (
                teacher_model is not None
                and distill_bg_weight > 0.0
                and loss is not None
            ):
                with torch.no_grad():
                    teacher_out = teacher_model(batch)
                    teacher_preds = teacher_out.ress

                bg_losses = []
                for i, view in enumerate(batch):
                    fg_mask = view.get("fg_mask", None)
                    if fg_mask is None:
                        continue
                    fg_mask = fg_mask > 0.5
                    bg_mask = ~fg_mask
                    valid_mask = view.get("valid_mask", None)
                    if valid_mask is not None:
                        bg_mask = bg_mask & valid_mask
                    if not bg_mask.any():
                        continue
                    student_pts = preds[i]["pts3d_in_self_view"]
                    teacher_pts = teacher_preds[i]["pts3d_in_self_view"]
                    bg_losses.append(
                        (student_pts[bg_mask] - teacher_pts[bg_mask]).abs().mean()
                    )

                if bg_losses:
                    bg_loss = torch.stack(bg_losses).mean()
                    loss_value, loss_details = loss
                    loss_details = dict(loss_details)
                    loss_details["bg_distill"] = float(bg_loss)
                    loss = (loss_value + distill_bg_weight * bg_loss, loss_details)

    result = dict(views=batch, pred=preds, loss=loss)
    return result[ret] if ret else result


def loss_of_one_batch_tbptt(
    batch,
    model,
    criterion,
    chunk_size,
    loss_scaler,
    optimizer,
    accelerator: Accelerator,
    log_writer=None,
    symmetrize_batch=False,
    use_amp=False,
    ret=None,
    img_mask=None,
    inference=False,
    teacher_model=None,
    distill_bg_weight=0.0,
):
    if len(batch) > 2:
        assert (
            symmetrize_batch is False
        ), "cannot symmetrize batch with more than 2 views"
    if symmetrize_batch:
        batch = make_batch_symmetric(batch)
    # Generic fix: ensure input images match model dtype
    target_dtype = torch.bfloat16
    device = accelerator.device
    for view in batch:
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                dtype = target_dtype if k in ["img", "ray_map"] else None
                view[k] = v.to(device=device, dtype=dtype)

    all_preds = []
    all_loss = 0.0
    all_loss_details = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        with torch.no_grad():
            (feat, pos, shape), (
                init_state_feat,
                init_mem,
                state_feat,
                state_pos,
                mem,
            ) = accelerator.unwrap_model(model)._forward_encoder(batch)
        feat = [f.detach() for f in feat]
        pos = [p.detach() for p in pos]
        shape = [s.detach() for s in shape]
        init_state_feat = init_state_feat.detach()
        init_mem = init_mem.detach()

        for chunk_id in range((len(batch) - 1) // chunk_size + 1):
            preds = []
            chunk = []
            state_feat = state_feat.detach()
            state_pos = state_pos.detach()
            mem = mem.detach()
            if chunk_id < ((len(batch) - 1) // chunk_size + 1) - 4:
                with torch.no_grad():
                    for in_chunk_idx in range(chunk_size):
                        i = chunk_id * chunk_size + in_chunk_idx
                        if i >= len(batch):
                            break
                        res, (state_feat, mem) = accelerator.unwrap_model(
                            model
                        )._forward_decoder_step(
                            batch,
                            i,
                            feat_i=feat[i],
                            pos_i=pos[i],
                            shape_i=shape[i],
                            init_state_feat=init_state_feat,
                            init_mem=init_mem,
                            state_feat=state_feat,
                            state_pos=state_pos,
                            mem=mem,
                        )
                        preds.append(res)
                        all_preds.append({k: v.detach() for k, v in res.items()})
                        chunk.append(batch[i])
                with torch.cuda.amp.autocast(enabled=False):
                    loss, loss_details = (
                        criterion(chunk, preds, camera1=batch[0]["camera_pose"])
                        if criterion is not None
                        else None
                    )
                    if (
                        teacher_model is not None
                        and distill_bg_weight > 0.0
                        and loss is not None
                    ):
                        with torch.no_grad():
                            teacher_out = teacher_model(chunk)
                            teacher_preds = teacher_out.ress
                        bg_losses = []
                        for i, view in enumerate(chunk):
                            fg_mask = view.get("fg_mask", None)
                            if fg_mask is None:
                                continue
                            fg_mask = fg_mask > 0.5
                            bg_mask = ~fg_mask
                            valid_mask = view.get("valid_mask", None)
                            if valid_mask is not None:
                                bg_mask = bg_mask & valid_mask
                            if not bg_mask.any():
                                continue
                            student_pts = preds[i]["pts3d_in_self_view"]
                            teacher_pts = teacher_preds[i]["pts3d_in_self_view"]
                            bg_losses.append(
                                (student_pts[bg_mask] - teacher_pts[bg_mask])
                                .abs()
                                .mean()
                            )
                        if bg_losses:
                            bg_loss = torch.stack(bg_losses).mean()
                            loss_value, loss_details = loss
                            loss_details = dict(loss_details)
                            loss_details["bg_distill"] = float(bg_loss)
                            loss = (
                                loss_value + distill_bg_weight * bg_loss,
                                loss_details,
                            )
                    all_loss += float(loss)
                    all_loss_details = merge_chunk_dict(
                        all_loss_details, loss_details, chunk_id * chunk_size
                    )
                    del loss
            else:
                for in_chunk_idx in range(chunk_size):
                    i = chunk_id * chunk_size + in_chunk_idx
                    if i >= len(batch):
                        break
                    res, (state_feat, mem) = accelerator.unwrap_model(
                        model
                    )._forward_decoder_step(
                        batch,
                        i,
                        feat_i=feat[i],
                        pos_i=pos[i],
                        shape_i=shape[i],
                        init_state_feat=init_state_feat,
                        init_mem=init_mem,
                        state_feat=state_feat,
                        state_pos=state_pos,
                        mem=mem,
                    )
                    preds.append(res)
                    all_preds.append({k: v.detach() for k, v in res.items()})
                    chunk.append(batch[i])
                with torch.cuda.amp.autocast(enabled=False):
                    loss, loss_details = (
                        criterion(chunk, preds, camera1=batch[0]["camera_pose"])
                        if criterion is not None
                        else None
                    )
                    if (
                        teacher_model is not None
                        and distill_bg_weight > 0.0
                        and loss is not None
                    ):
                        with torch.no_grad():
                            teacher_out = teacher_model(chunk)
                            teacher_preds = teacher_out.ress
                        bg_losses = []
                        for i, view in enumerate(chunk):
                            fg_mask = view.get("fg_mask", None)
                            if fg_mask is None:
                                continue
                            fg_mask = fg_mask > 0.5
                            bg_mask = ~fg_mask
                            valid_mask = view.get("valid_mask", None)
                            if valid_mask is not None:
                                bg_mask = bg_mask & valid_mask
                            if not bg_mask.any():
                                continue
                            student_pts = preds[i]["pts3d_in_self_view"]
                            teacher_pts = teacher_preds[i]["pts3d_in_self_view"]
                            bg_losses.append(
                                (student_pts[bg_mask] - teacher_pts[bg_mask])
                                .abs()
                                .mean()
                            )
                        if bg_losses:
                            bg_loss = torch.stack(bg_losses).mean()
                            loss_value, loss_details = loss
                            loss_details = dict(loss_details)
                            loss_details["bg_distill"] = float(bg_loss)
                            loss = (
                                loss_value + distill_bg_weight * bg_loss,
                                loss_details,
                            )
                    all_loss += float(loss)
                    all_loss_details = merge_chunk_dict(
                        all_loss_details, loss_details, chunk_id * chunk_size
                    )
                    loss_scaler(
                        loss,
                        optimizer,
                        parameters=model.parameters(),
                        update_grad=True,
                        clip_grad=1.0,
                    )
                    optimizer.zero_grad()
                    del loss
    result = dict(
        views=batch,
        pred=all_preds,
        loss=(all_loss / ((len(batch) - 1) // chunk_size + 1), all_loss_details),
        already_backprop=True,
    )
    return result[ret] if ret else result


@torch.no_grad()
def inference(groups, model, device, verbose=True, profile=False, accelerator=None):
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    for view in groups:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            if isinstance(view[name], tuple) or isinstance(view[name], list):
                view[name] = [x.to(device, non_blocking=True) for x in view[name]]
            else:
                view[name] = view[name].to(device, non_blocking=True)

    if verbose:
        print(f">> Inference with model on {len(groups)} image/raymaps")

    # Instrumentation for profiling
    if profile:
        events = []
        
        def start_hook(module, input):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            events.append({"start": start, "type": "start", "module": "decoder"})

        def end_hook(module, input, output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            events.append({"end": end, "type": "end", "module": "decoder"})
            
        def enc_start_hook(module, input):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            events.append({"start": start, "type": "start", "module": "encoder"})

        def enc_end_hook(module, input, output):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            events.append({"end": end, "type": "end", "module": "encoder"})

        # Register hooks
        # Encoder measures batch processing time (total for all views)
        handle1_enc = model.patch_embed.register_forward_pre_hook(enc_start_hook)
        handle2_enc = model.enc_norm.register_forward_hook(enc_end_hook)
        
        # Decoder measures per-frame recurrent loop
        handle1_dec = model.decoder_embed.register_forward_pre_hook(start_hook)
        handle2_dec = model.downstream_head.register_forward_hook(end_hook)

        handle2_dec = model.downstream_head.register_forward_hook(end_hook)

    try:
        # Mock accelerator if None
        accelerator = accelerator
        if accelerator is None:
            class MockAccelerator:
                def __init__(self, dev):
                    self.device = dev
            accelerator = MockAccelerator(device)
            
        res, state_args = loss_of_one_batch(groups, model, None, accelerator, inference=True)
    finally:
        if profile:
            handle1_enc.remove()
            handle2_enc.remove()
            handle1_dec.remove()
            handle2_dec.remove()

    if profile:
        torch.cuda.synchronize()
        
        # Process events
        enc_start = [e["start"] for e in events if e["type"] == "start" and e["module"] == "encoder"][0]
        enc_end = [e["end"] for e in events if e["type"] == "end" and e["module"] == "encoder"][0]
        total_enc_time = enc_start.elapsed_time(enc_end)
        avg_enc_time = total_enc_time / len(groups)
        
        dec_starts = [e["start"] for e in events if e["type"] == "start" and e["module"] == "decoder"]
        dec_ends = [e["end"] for e in events if e["type"] == "end" and e["module"] == "decoder"]
        
        print("\n--- Profiling Report (Full Model) ---")
        print(f"Total Encoder Time (Batch): {total_enc_time:.2f} ms")
        print(f"Average Encoder Time per Frame: {avg_enc_time:.2f} ms")
        
        avg_total = 0
        for i, (s, e) in enumerate(zip(dec_starts, dec_ends)):
            dec_time = s.elapsed_time(e)
            total_frame_time = avg_enc_time + dec_time
            avg_total += total_frame_time
            print(f"Frame {i:03d} Total Time: {total_frame_time:.2f} ms (Enc: {avg_enc_time:.2f} + Dec: {dec_time:.2f})")
        
        if len(dec_starts) > 0:
            print(f"Average Total Time Per Frame: {avg_total/len(dec_starts):.2f} ms")
        print("-----------------------------------------\n")

    result = to_cpu(res)
    return result, state_args


@torch.no_grad()
def inference_step(view, state_args, model, device, verbose=True):
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    for name in view.keys():  # pseudo_focal
        if name in ignore_keys:
            continue
        if isinstance(view[name], tuple) or isinstance(view[name], list):
            view[name] = [x.to(device, non_blocking=True) for x in view[name]]
        else:
            view[name] = view[name].to(device, non_blocking=True)

    with torch.cuda.amp.autocast(enabled=False):
        state_feat, state_pos, init_state_feat, mem, init_mem = state_args
        pred, _ = model.inference_step(
            view, state_feat, state_pos, init_state_feat, mem, init_mem
        )

    res = dict(pred=pred)
    result = to_cpu(res)
    return result


@torch.no_grad()
def inference_recurrent(groups, model, device, verbose=True):
    ignore_keys = set(
        ["depthmap", "dataset", "label", "instance", "idx", "true_shape", "rng"]
    )
    for view in groups:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            if isinstance(view[name], tuple) or isinstance(view[name], list):
                view[name] = [x.to(device, non_blocking=True) for x in view[name]]
            else:
                view[name] = view[name].to(device, non_blocking=True)

    if verbose:
        print(f">> Inference with model on {len(groups)} image/raymaps")

    with torch.cuda.amp.autocast(enabled=False):
        preds, batch, state_args = model.forward_recurrent(
            groups, device, ret_state=True
        )
        res = dict(views=batch, pred=preds)
    result = to_cpu(res)
    return result, state_args


def check_if_same_size(pairs):
    shapes1 = [img1["img"].shape[-2:] for img1, img2 in pairs]
    shapes2 = [img2["img"].shape[-2:] for img1, img2 in pairs]
    return all(shapes1[0] == s for s in shapes1) and all(
        shapes2[0] == s for s in shapes2
    )


def get_pred_pts3d(gt, pred, use_pose=False, inplace=False):
    if "depth" in pred and "pseudo_focal" in pred:
        try:
            pp = gt["camera_intrinsics"][..., :2, 2]
        except KeyError:
            pp = None
        pts3d = depthmap_to_pts3d(**pred, pp=pp)

    elif "pts3d" in pred:

        pts3d = pred["pts3d"]

    elif "pts3d_in_other_view" in pred:

        assert use_pose is True
        return (
            pred["pts3d_in_other_view"]
            if inplace
            else pred["pts3d_in_other_view"].clone()
        )

    if use_pose:
        camera_pose = pred.get("camera_pose")
        assert camera_pose is not None
        pts3d = geotrf(camera_pose, pts3d)

    return pts3d


def find_opt_scaling(
    gt_pts1,
    gt_pts2,
    pr_pts1,
    pr_pts2=None,
    fit_mode="weiszfeld_stop_grad",
    valid1=None,
    valid2=None,
):
    assert gt_pts1.ndim == pr_pts1.ndim == 4
    assert gt_pts1.shape == pr_pts1.shape
    if gt_pts2 is not None:
        assert gt_pts2.ndim == pr_pts2.ndim == 4
        assert gt_pts2.shape == pr_pts2.shape

    nan_gt_pts1 = invalid_to_nans(gt_pts1, valid1).flatten(1, 2)
    nan_gt_pts2 = (
        invalid_to_nans(gt_pts2, valid2).flatten(1, 2) if gt_pts2 is not None else None
    )

    pr_pts1 = invalid_to_nans(pr_pts1, valid1).flatten(1, 2)
    pr_pts2 = (
        invalid_to_nans(pr_pts2, valid2).flatten(1, 2) if pr_pts2 is not None else None
    )

    all_gt = (
        torch.cat((nan_gt_pts1, nan_gt_pts2), dim=1)
        if gt_pts2 is not None
        else nan_gt_pts1
    )
    all_pr = torch.cat((pr_pts1, pr_pts2), dim=1) if pr_pts2 is not None else pr_pts1

    dot_gt_pr = (all_pr * all_gt).sum(dim=-1)
    dot_gt_gt = all_gt.square().sum(dim=-1)

    if fit_mode.startswith("avg"):

        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)
    elif fit_mode.startswith("median"):
        scaling = (dot_gt_pr / dot_gt_gt).nanmedian(dim=1).values
    elif fit_mode.startswith("weiszfeld"):

        scaling = dot_gt_pr.nanmean(dim=1) / dot_gt_gt.nanmean(dim=1)

        for iter in range(10):

            dis = (all_pr - scaling.view(-1, 1, 1) * all_gt).norm(dim=-1)

            w = dis.clip_(min=1e-8).reciprocal()

            scaling = (w * dot_gt_pr).nanmean(dim=1) / (w * dot_gt_gt).nanmean(dim=1)
    else:
        raise ValueError(f"bad {fit_mode=}")

    if fit_mode.endswith("stop_grad"):
        scaling = scaling.detach()

    scaling = scaling.clip(min=1e-3)

    return scaling
