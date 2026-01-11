# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# modified from DUSt3R

import PIL.Image
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from dust3r.utils.geometry import (
    geotrf,
    get_med_dist_between_poses,
    depthmap_to_absolute_camera_coordinates,
)
from dust3r.utils.device import to_numpy
from dust3r.utils.image import rgb, img_to_arr
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

try:
    import trimesh
except ImportError:
    print("/!\\ module trimesh is not installed, cannot visualize results /!\\")


def float2uint8(x):
    return (255.0 * x).astype(np.uint8)


def uint82float(img):
    return np.ascontiguousarray(img) / 255.0


def cat_3d(vecs):
    if isinstance(vecs, (np.ndarray, torch.Tensor)):
        vecs = [vecs]
    return np.concatenate([p.reshape(-1, 3) for p in to_numpy(vecs)])


def show_raw_pointcloud(pts3d, colors, point_size=2):
    scene = trimesh.Scene()

    pct = trimesh.PointCloud(cat_3d(pts3d), colors=cat_3d(colors))
    scene.add_geometry(pct)

    scene.show(line_settings={"point_size": point_size})


def pts3d_to_trimesh(img, pts3d, valid=None):
    H, W, THREE = img.shape
    assert THREE == 3
    assert img.shape == pts3d.shape

    vertices = pts3d.reshape(-1, 3)

    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()  # top-left corner
    idx2 = idx[:-1, +1:].ravel()  # right-left corner
    idx3 = idx[+1:, :-1].ravel()  # bottom-left corner
    idx4 = idx[+1:, +1:].ravel()  # bottom-right corner
    faces = np.concatenate(
        (
            np.c_[idx1, idx2, idx3],
            np.c_[
                idx3, idx2, idx1
            ],  # same triangle, but backward (cheap solution to cancel face culling)
            np.c_[idx2, idx3, idx4],
            np.c_[
                idx4, idx3, idx2
            ],  # same triangle, but backward (cheap solution to cancel face culling)
        ),
        axis=0,
    )

    face_colors = np.concatenate(
        (
            img[:-1, :-1].reshape(-1, 3),
            img[:-1, :-1].reshape(-1, 3),
            img[+1:, +1:].reshape(-1, 3),
            img[+1:, +1:].reshape(-1, 3),
        ),
        axis=0,
    )

    if valid is not None:
        assert valid.shape == (H, W)
        valid_idxs = valid.ravel()
        valid_faces = valid_idxs[faces].all(axis=-1)
        faces = faces[valid_faces]
        face_colors = face_colors[valid_faces]

    assert len(faces) == len(face_colors)
    return dict(vertices=vertices, face_colors=face_colors, faces=faces)


def cat_meshes(meshes):
    vertices, faces, colors = zip(
        *[(m["vertices"], m["faces"], m["face_colors"]) for m in meshes]
    )
    n_vertices = np.cumsum([0] + [len(v) for v in vertices])
    for i in range(len(faces)):
        faces[i][:] += n_vertices[i]

    vertices = np.concatenate(vertices)
    colors = np.concatenate(colors)
    faces = np.concatenate(faces)
    return dict(vertices=vertices, face_colors=colors, faces=faces)


def show_duster_pairs(view1, view2, pred1, pred2):
    import matplotlib.pyplot as pl

    pl.ion()

    for e in range(len(view1["instance"])):
        i = view1["idx"][e]
        j = view2["idx"][e]
        img1 = rgb(view1["img"][e])
        img2 = rgb(view2["img"][e])
        conf1 = pred1["conf"][e].squeeze()
        conf2 = pred2["conf"][e].squeeze()
        score = conf1.mean() * conf2.mean()
        print(f">> Showing pair #{e} {i}-{j} {score=:g}")
        pl.clf()
        pl.subplot(221).imshow(img1)
        pl.subplot(223).imshow(img2)
        pl.subplot(222).imshow(conf1, vmin=1, vmax=30)
        pl.subplot(224).imshow(conf2, vmin=1, vmax=30)
        pts1 = pred1["pts3d"][e]
        pts2 = pred2["pts3d_in_other_view"][e]
        pl.subplots_adjust(0, 0, 1, 1, 0, 0)
        if input("show pointcloud? (y/n) ") == "y":
            show_raw_pointcloud(cat(pts1, pts2), cat(img1, img2), point_size=5)


def auto_cam_size(im_poses):
    return 0.1 * get_med_dist_between_poses(im_poses)


class SceneViz:
    def __init__(self):
        self.scene = trimesh.Scene()

    def add_rgbd(
        self, image, depth, intrinsics=None, cam2world=None, zfar=np.inf, mask=None
    ):
        image = img_to_arr(image)

        if intrinsics is None:
            H, W, THREE = image.shape
            focal = max(H, W)
            intrinsics = np.float32([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])

        pts3d = depthmap_to_pts3d(depth, intrinsics, cam2world=cam2world)

        return self.add_pointcloud(
            pts3d, image, mask=(depth < zfar) if mask is None else mask
        )

    def add_pointcloud(self, pts3d, color=(0, 0, 0), mask=None, denoise=False):
        pts3d = to_numpy(pts3d)
        mask = to_numpy(mask)
        if not isinstance(pts3d, list):
            pts3d = [pts3d.reshape(-1, 3)]
            if mask is not None:
                mask = [mask.ravel()]
        if not isinstance(color, (tuple, list)):
            color = [color.reshape(-1, 3)]
        if mask is None:
            mask = [slice(None)] * len(pts3d)

        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        pct = trimesh.PointCloud(pts)

        if isinstance(color, (list, np.ndarray, torch.Tensor)):
            color = to_numpy(color)
            col = np.concatenate([p[m] for p, m in zip(color, mask)])
            assert col.shape == pts.shape, bb()
            pct.visual.vertex_colors = uint8(col.reshape(-1, 3))
        else:
            assert len(color) == 3
            pct.visual.vertex_colors = np.broadcast_to(uint8(color), pts.shape)

        if denoise:

            centroid = np.median(pct.vertices, axis=0)
            dist_to_centroid = np.linalg.norm(pct.vertices - centroid, axis=-1)
            dist_thr = np.quantile(dist_to_centroid, 0.99)
            valid = dist_to_centroid < dist_thr

            pct = trimesh.PointCloud(
                pct.vertices[valid], color=pct.visual.vertex_colors[valid]
            )

        self.scene.add_geometry(pct)
        return self

    def add_rgbd(
        self, image, depth, intrinsics=None, cam2world=None, zfar=np.inf, mask=None
    ):

        if intrinsics is None:
            H, W, THREE = image.shape
            focal = max(H, W)
            intrinsics = np.float32([[focal, 0, W / 2], [0, focal, H / 2], [0, 0, 1]])

        pts3d, mask2 = depthmap_to_absolute_camera_coordinates(
            depth, intrinsics, cam2world
        )
        mask2 &= depth < zfar

        if mask is not None:
            mask2 &= mask

        return self.add_pointcloud(pts3d, image, mask=mask2)

    def add_camera(
        self,
        pose_c2w,
        focal=None,
        color=(0, 0, 0),
        image=None,
        imsize=None,
        cam_size=0.03,
    ):
        pose_c2w, focal, color, image = to_numpy((pose_c2w, focal, color, image))
        image = img_to_arr(image)
        if isinstance(focal, np.ndarray) and focal.shape == (3, 3):
            intrinsics = focal
            focal = (intrinsics[0, 0] * intrinsics[1, 1]) ** 0.5
            if imsize is None:
                imsize = (2 * intrinsics[0, 2], 2 * intrinsics[1, 2])

        add_scene_cam(
            self.scene,
            pose_c2w,
            color,
            image,
            focal,
            imsize=imsize,
            screen_width=cam_size,
            marker=None,
        )
        return self

    def add_cameras(
        self, poses, focals=None, images=None, imsizes=None, colors=None, **kw
    ):
        get = lambda arr, idx: None if arr is None else arr[idx]
        for i, pose_c2w in enumerate(poses):
            self.add_camera(
                pose_c2w,
                get(focals, i),
                image=get(images, i),
                color=get(colors, i),
                imsize=get(imsizes, i),
                **kw,
            )
        return self

    def show(self, point_size=2):
        self.scene.show(line_settings={"point_size": point_size})


def show_raw_pointcloud_with_cams(
    imgs, pts3d, mask, focals, cams2world, point_size=2, cam_size=0.05, cam_color=None
):
    """Visualization of a pointcloud with cameras
    imgs = (N, H, W, 3) or N-size list of [(H,W,3), ...]
    pts3d = (N, H, W, 3) or N-size list of [(H,W,3), ...]
    focals = (N,) or N-size list of [focal, ...]
    cams2world = (N,4,4) or N-size list of [(4,4), ...]
    """
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
    col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
    pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
    scene.add_geometry(pct)

    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        add_scene_cam(
            scene,
            pose_c2w,
            camera_edge_color,
            imgs[i] if i < len(imgs) else None,
            focals[i],
            screen_width=cam_size,
        )

    scene.show(line_settings={"point_size": point_size})


def add_scene_cam(
    scene,
    pose_c2w,
    edge_color,
    image=None,
    focal=None,
    imsize=None,
    screen_width=0.03,
    marker=None,
):
    if image is not None:
        image = np.asarray(image)
        H, W, THREE = image.shape
        assert THREE == 3
        if image.dtype != np.uint8:
            image = np.uint8(255 * image)
    elif imsize is not None:
        W, H = imsize
    elif focal is not None:
        H = W = focal / 1.1
    else:
        H = W = 1

    if isinstance(focal, np.ndarray):
        focal = focal[0]
    if not focal:
        focal = min(H, W) * 1.1  # default value

    height = max(screen_width / 10, focal * screen_width / H)
    width = screen_width * 0.5**0.5
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler("z", np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = W / H
    transform = pose_c2w @ OPENGL @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)  # , transform=transform)

    if image is not None:
        vertices = geotrf(transform, cam.vertices[[4, 5, 1, 3]])
        faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
        img = trimesh.Trimesh(vertices=vertices, faces=faces)
        uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
        img.visual = trimesh.visual.TextureVisuals(
            uv_coords, image=PIL.Image.fromarray(image)
        )
        scene.add_geometry(img)

    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler("z", np.deg2rad(2)).as_matrix()
    vertices = np.r_[cam.vertices, 0.95 * cam.vertices, geotrf(rot2, cam.vertices)]
    vertices = geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2 * len(cam.vertices)

        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    faces += [(c, b, a) for a, b, c in faces]

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = edge_color
    scene.add_geometry(cam)

    if marker == "o":
        marker = trimesh.creation.icosphere(3, radius=screen_width / 4)
        marker.vertices += pose_c2w[:3, 3]
        marker.visual.face_colors[:, :3] = edge_color
        scene.add_geometry(marker)


def cat(a, b):
    return np.concatenate((a.reshape(-1, 3), b.reshape(-1, 3)))


OPENGL = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


CAM_COLORS = [
    (255, 0, 0),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 255),
    (255, 204, 0),
    (0, 204, 204),
    (128, 255, 255),
    (255, 128, 255),
    (255, 255, 128),
    (0, 0, 0),
    (128, 128, 128),
]


def uint8(colors):
    if not isinstance(colors, np.ndarray):
        colors = np.array(colors)
    if np.issubdtype(colors.dtype, np.floating):
        colors *= 255
    assert 0 <= colors.min() and colors.max() < 256
    return np.uint8(colors)


def segment_sky(image):
    import cv2
    from scipy import ndimage

    image = to_numpy(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.uint8(255 * image.clip(min=0, max=1))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([0, 0, 100])
    upper_blue = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue).view(bool)

    mask |= (hsv[:, :, 1] < 10) & (hsv[:, :, 2] > 150)
    mask |= (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 180)
    mask |= (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 220)

    kernel = np.ones((5, 5), np.uint8)
    mask2 = ndimage.binary_opening(mask, structure=kernel)

    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask2.view(np.uint8), connectivity=8
    )
    cc_sizes = stats[1:, cv2.CC_STAT_AREA]
    order = cc_sizes.argsort()[::-1]  # bigger first
    i = 0
    selection = []
    while i < len(order) and cc_sizes[order[i]] > cc_sizes[order[0]] / 2:
        selection.append(1 + order[i])
        i += 1
    mask3 = np.in1d(labels, selection).reshape(labels.shape)

    return torch.from_numpy(mask3)


def get_vertical_colorbar(h, vmin, vmax, cmap_name="jet", label=None, cbar_precision=2):
    """
    :param w: pixels
    :param h: pixels
    :param vmin: min value
    :param vmax: max value
    :param cmap_name:
    :param label
    :return:
    """
    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, ticks=tick_loc, orientation="vertical"
    )

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)
    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.0
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(
    x,
    cmap_name="jet",
    mask=None,
    range=None,
    append_cbar=False,
    cbar_in_image=False,
    cbar_precision=2,
):
    """
    turn a grayscale image into a color image
    :param x: input grayscale, [H, W]
    :param cmap_name: the colorization method
    :param mask: the mask image, [H, W]
    :param range: the range for scaling, automatic if None, [min, max]
    :param append_cbar: if append the color bar
    :param cbar_in_image: put the color bar inside the image to keep the output image the same size as the input image
    :return: colorized image, [H, W]
    """
    if range is not None:
        vmin, vmax = range
    elif mask is not None:

        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])

        x[np.logical_not(mask)] = vmin

    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += 1e-6

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1.0 - mask)

    cbar = get_vertical_colorbar(
        h=x.shape[0],
        vmin=vmin,
        vmax=vmax,
        cmap_name=cmap_name,
        cbar_precision=cbar_precision,
    )

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1] :, :] = cbar
        else:
            x_new = np.concatenate(
                (x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1
            )
        return x_new
    else:
        return x_new


def colorize(
    x, cmap_name="jet", mask=None, range=None, append_cbar=False, cbar_in_image=False
):
    """
    turn a grayscale image into a color image
    :param x: torch.Tensor, grayscale image, [H, W] or [B, H, W]
    :param mask: torch.Tensor or None, mask image, [H, W] or [B, H, W] or None
    """

    device = x.device
    x = x.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy() > 0.99
        kernel = np.ones((3, 3), np.uint8)

    if x.ndim == 2:
        x = x[None]
        if mask is not None:
            mask = mask[None]

    out = []
    for x_ in x:
        if mask is not None:
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        x_ = colorize_np(x_, cmap_name, mask, range, append_cbar, cbar_in_image)
        out.append(torch.from_numpy(x_).to(device).float())
    out = torch.stack(out).squeeze(0)
    return out


def draw_correspondences(
    imgs1, imgs2, coords1, coords2, interval=10, color_by=0, radius=2
):
    """
    draw correspondences between two images
    :param img1: tensor [B, H, W, 3]
    :param img2: tensor [B, H, W, 3]
    :param coord1: tensor [B, N, 2]
    :param coord2: tensor [B, N, 2]
    :param interval: int the interval between two points
    :param color_by: specify the color based on image 1 or image 2, 0 or 1
    :return: [B, 2*H, W, 3]
    """
    batch_size = len(imgs1)
    out = []
    for i in range(batch_size):
        img1 = imgs1[i].detach().cpu().numpy()
        img2 = imgs2[i].detach().cpu().numpy()
        coord1 = (
            coords1[i].detach().cpu().numpy()[::interval, ::interval].reshape(-1, 2)
        )
        coord2 = (
            coords2[i].detach().cpu().numpy()[::interval, ::interval].reshape(-1, 2)
        )
        img = drawMatches(
            img1, img2, coord1, coord2, radius=radius, color_by=color_by, row_cat=True
        )
        out.append(img)
    out = np.stack(out)
    return out


def draw_correspondences_lines(
    imgs1, imgs2, coords1, coords2, interval=10, color_by=0, radius=2
):
    """
    draw correspondences between two images
    :param img1: tensor [B, H, W, 3]
    :param img2: tensor [B, H, W, 3]
    :param coord1: tensor [B, N, 2]
    :param coord2: tensor [B, N, 2]
    :param interval: int the interval between two points
    :param color_by: specify the color based on image 1 or image 2, 0 or 1
    :return: [B, 2*H, W, 3]
    """
    batch_size = len(imgs1)
    out = []
    for i in range(batch_size):
        img1 = imgs1[i].detach().cpu().numpy()
        img2 = imgs2[i].detach().cpu().numpy()
        coord1 = (
            coords1[i].detach().cpu().numpy()[::interval, ::interval].reshape(-1, 2)
        )
        coord2 = (
            coords2[i].detach().cpu().numpy()[::interval, ::interval].reshape(-1, 2)
        )
        img = drawMatches_lines(
            img1, img2, coord1, coord2, radius=radius, color_by=color_by, row_cat=True
        )
        out.append(img)
    out = np.stack(out)
    return out


def drawMatches(img1, img2, kp1, kp2, radius=2, mask=None, color_by=0, row_cat=False):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img1 = np.ascontiguousarray(float2uint8(img1))
    img2 = np.ascontiguousarray(float2uint8(img2))

    center1 = np.median(kp1, axis=0)
    center2 = np.median(kp2, axis=0)

    set_max = range(128)
    colors = {m: i for i, m in enumerate(set_max)}
    colors = {
        m: (255 * np.array(plt.cm.hsv(i / float(len(colors))))[:3][::-1]).astype(
            np.int32
        )
        for m, i in colors.items()
    }

    if mask is not None:
        ind = np.argsort(mask)[::-1]
        kp1 = kp1[ind]
        kp2 = kp2[ind]
        mask = mask[ind]

    for i, (pt1, pt2) in enumerate(zip(kp1, kp2)):

        if color_by == 0:
            coord_angle = np.arctan2(pt1[1] - center1[1], pt1[0] - center1[0])
        elif color_by == 1:
            coord_angle = np.arctan2(pt2[1] - center2[1], pt2[0] - center2[0])

        corr_color = np.int32(64 * coord_angle / np.pi) % 128
        color = tuple(colors[corr_color].tolist())

        if (
            (pt1[0] <= w1 - 1)
            and (pt1[0] >= 0)
            and (pt1[1] <= h1 - 1)
            and (pt1[1] >= 0)
        ):
            img1 = cv2.circle(
                img1, (int(pt1[0]), int(pt1[1])), radius, color, -1, cv2.LINE_AA
            )

        if (
            (pt2[0] <= w2 - 1)
            and (pt2[0] >= 0)
            and (pt2[1] <= h2 - 1)
            and (pt2[1] >= 0)
        ):
            if mask is not None and mask[i]:
                img2 = cv2.drawMarker(
                    img2,
                    (int(pt2[0]), int(pt2[1])),
                    color,
                    markerType=cv2.MARKER_CROSS,
                    markerSize=int(5 * radius),
                    thickness=int(radius / 2),
                    line_type=cv2.LINE_AA,
                )
            else:
                img2 = cv2.circle(
                    img2, (int(pt2[0]), int(pt2[1])), radius, color, -1, cv2.LINE_AA
                )
    if row_cat:
        whole_img = np.concatenate([img1, img2], axis=0)
    else:
        whole_img = np.concatenate([img1, img2], axis=1)
    return whole_img
    if row_cat:
        return np.concatenate([img1, img2], axis=0)
    return np.concatenate([img1, img2], axis=1)


def drawMatches_lines(
    img1, img2, kp1, kp2, radius=2, mask=None, color_by=0, row_cat=False
):

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    img1 = np.ascontiguousarray(float2uint8(img1))
    img2 = np.ascontiguousarray(float2uint8(img2))

    center1 = np.median(kp1, axis=0)
    center2 = np.median(kp2, axis=0)

    set_max = range(128)
    colors = {m: i for i, m in enumerate(set_max)}
    colors = {
        m: (255 * np.array(plt.cm.hsv(i / float(len(colors))))[:3][::-1]).astype(
            np.int32
        )
        for m, i in colors.items()
    }

    if mask is not None:
        ind = np.argsort(mask)[::-1]
        kp1 = kp1[ind]
        kp2 = kp2[ind]
        mask = mask[ind]

    if row_cat:
        whole_img = np.concatenate([img1, img2], axis=0)
    else:
        whole_img = np.concatenate([img1, img2], axis=1)
    for i, (pt1, pt2) in enumerate(zip(kp1, kp2)):
        if color_by == 0:
            coord_angle = np.arctan2(pt1[1] - center1[1], pt1[0] - center1[0])
        elif color_by == 1:
            coord_angle = np.arctan2(pt2[1] - center2[1], pt2[0] - center2[0])

        corr_color = np.int32(64 * coord_angle / np.pi) % 128
        color = tuple(colors[corr_color].tolist())
        rand_val = np.random.rand()
        if rand_val < 0.1:
            if (
                (pt1[0] <= w1 - 1)
                and (pt1[0] >= 0)
                and (pt1[1] <= h1 - 1)
                and (pt1[1] >= 0)
            ) and (
                (pt2[0] <= w2 - 1)
                and (pt2[0] >= 0)
                and (pt2[1] <= h2 - 1)
                and (pt2[1] >= 0)
            ):

                whole_img = cv2.circle(
                    whole_img,
                    (int(pt1[0]), int(pt1[1])),
                    radius,
                    color,
                    -1,
                    cv2.LINE_AA,
                )

                if row_cat:
                    whole_img = cv2.circle(
                        whole_img,
                        (int(pt2[0]), int(pt2[1] + h1)),
                        radius,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.line(
                        whole_img,
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0]), int(pt2[1] + h1)),
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                else:
                    whole_img = cv2.circle(
                        whole_img,
                        (int(pt2[0] + w1), int(pt2[1])),
                        radius,
                        color,
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.line(
                        whole_img,
                        (int(pt1[0]), int(pt1[1])),
                        (int(pt2[0] + w1), int(pt2[1])),
                        color,
                        1,
                        cv2.LINE_AA,
                    )
    return whole_img
    if row_cat:
        return np.concatenate([img1, img2], axis=0)
    return np.concatenate([img1, img2], axis=1)


import torch
import os
import time
import viser


def rotation_matrix_to_quaternion(R):
    """
    :param R: [3, 3]
    :return: [4]
    """
    tr = np.trace(R)
    Rxx = R[0, 0]
    Ryy = R[1, 1]
    Rzz = R[2, 2]
    q = np.zeros(4)
    q[0] = 0.5 * np.sqrt(1 + tr)
    q[1] = (R[2, 1] - R[1, 2]) / (4 * q[0])
    q[2] = (R[0, 2] - R[2, 0]) / (4 * q[0])
    q[3] = (R[1, 0] - R[0, 1]) / (4 * q[0])
    return q


class PointCloudViewer:
    def __init__(self, pc_dir, device="cpu"):
        self.server = viser.ViserServer()
        self.server.set_up_direction("-y")
        self.device = device
        self.tt = lambda x: torch.from_numpy(x).float().to(device)
        self.pc_dir = pc_dir
        self.pcs, self.all_steps = self.read_data()
        self.num_frames = len(self.all_steps)

        self.fix_camera = False
        self.camera_scale = self.server.add_gui_slider(
            "camera_scale",
            min=0.01,
            max=1.0,
            step=0.01,
            initial_value=0.1,
        )

        self.camera_handles = []

    def read_data(self):
        pc_list = os.listdir(self.pc_dir)
        pc_list.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
        pcs = {}
        step_list = []
        for pc_name in pc_list:
            pc = np.load(os.path.join(self.pc_dir, pc_name))
            step = int(pc_name.split(".")[0].split("_")[-1])
            pcs.update({step: {"pc": pc}})
            step_list.append(step)
        return pcs, step_list

    def parse_pc_data(self, pc, batch_idx=-1):
        idx = batch_idx
        ret_dict = {}
        for i in range(len(pc.keys()) // 2):
            pred_pts = pc[f"pts3d_{i+1}"][idx].reshape(-1, 3)  # [N, 3]
            color = pc[f"colors_{i+1}"][idx].reshape(-1, 3)  # [N, 3]
            ret_dict.update({f"pred_pts_{i+1}": pred_pts, f"color_{i+1}": color})
        return ret_dict

    def add_pc(self, step):
        pc = self.pcs[step]["pc"]
        pc_dict = self.parse_pc_data(pc)

        for i in range(len(pc_dict.keys()) // 2):
            self.server.add_point_cloud(
                name=f"/frames/{step}/pred_pts_{i+1}_{step}",
                points=pc_dict[f"pred_pts_{i+1}"],
                colors=pc_dict[f"color_{i+1}"],
                point_size=0.002,
            )

        if not self.fix_camera:
            raise NotImplementedError

            R21, T21 = find_rigid_alignment_batched(
                torch.from_numpy(pc_dict["pred_pts1_2"][None]),
                torch.from_numpy(pc_dict["pred_pts1_1"][None]),
            )
            R12, T12 = find_rigid_alignment_batched(
                torch.from_numpy(pc_dict["pred_pts2_1"][None]),
                torch.from_numpy(pc_dict["pred_pts2_2"][None]),
            )
            R21 = R21[0].numpy()
            T21 = T21.numpy()
            R12 = R12[0].numpy()
            T12 = T12.numpy()
            pred_pts1_2 = pc_dict["pred_pts1_2"] @ R21.T + T21
            pred_pts2_1 = pc_dict["pred_pts2_1"] @ R12.T + T12
            self.server.add_point_cloud(
                name=f"/frames/{step}/pred_pts1_2_{step}",
                points=pred_pts1_2,
                colors=pc_dict["color1_2"],
                point_size=0.002,
            )

            self.server.add_point_cloud(
                name=f"/frames/{step}/pred_pts2_1_{step}",
                points=pred_pts2_1,
                colors=pc_dict["color2_1"],
                point_size=0.002,
            )
            img1 = pc_dict["color1_1"].reshape(224, 224, 3)
            img2 = pc_dict["color2_2"].reshape(224, 224, 3)
            self.camera_handles.append(
                self.server.add_camera_frustum(
                    name=f"/frames/{step}/camera1_{step}",
                    fov=2.0 * np.arctan(224.0 / 490.0),
                    aspect=1.0,
                    scale=self.camera_scale.value,
                    color=(1.0, 0, 0),
                    image=img1,
                )
            )
            self.camera_handles.append(
                self.server.add_camera_frustum(
                    name=f"/frames/{step}/camera2_{step}",
                    fov=2.0 * np.arctan(224.0 / 490.0),
                    aspect=1.0,
                    scale=self.camera_scale.value,
                    color=(0, 0, 1.0),
                    wxyz=rotation_matrix_to_quaternion(R21),
                    position=T21,
                    image=img2,
                )
            )

    def animate(self):
        with self.server.add_gui_folder("Playback"):
            gui_timestep = self.server.add_gui_slider(
                "Train Step",
                min=0,
                max=self.num_frames - 1,
                step=1,
                initial_value=0,
                disabled=True,
            )
            gui_next_frame = self.server.add_gui_button("Next Step", disabled=True)
            gui_prev_frame = self.server.add_gui_button("Prev Step", disabled=True)
            gui_playing = self.server.add_gui_checkbox("Playing", False)
            gui_framerate = self.server.add_gui_slider(
                "FPS", min=1, max=60, step=0.1, initial_value=1
            )
            gui_framerate_options = self.server.add_gui_button_group(
                "FPS options", ("10", "20", "30", "60")
            )

        @gui_next_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value + 1) % self.num_frames

        @gui_prev_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value - 1) % self.num_frames

        @gui_playing.on_update
        def _(_) -> None:
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

        @gui_framerate_options.on_click
        def _(_) -> None:
            gui_framerate.value = int(gui_framerate_options.value)

        prev_timestep = gui_timestep.value

        @gui_timestep.on_update
        def _(_) -> None:
            nonlocal prev_timestep
            current_timestep = gui_timestep.value
            with self.server.atomic():
                frame_nodes[current_timestep].visible = True
                frame_nodes[prev_timestep].visible = False
            prev_timestep = current_timestep
            self.server.flush()  # Optional!

        self.server.add_frame(
            "/frames",
            show_axes=False,
        )
        frame_nodes = []
        for i in range(self.num_frames):
            step = self.all_steps[i]
            frame_nodes.append(
                self.server.add_frame(
                    f"/frames/{step}",
                    show_axes=False,
                )
            )
            self.add_pc(step)

        for i, frame_node in enumerate(frame_nodes):

            frame_node.visible = i == gui_timestep.value

        prev_timestep = gui_timestep.value
        while True:
            if gui_playing.value:
                gui_timestep.value = (gui_timestep.value + 1) % self.num_frames
            for handle in self.camera_handles:
                handle.scale = self.camera_scale.value
            time.sleep(1.0 / gui_framerate.value)

    def run(self):
        self.animate()
        while True:
            time.sleep(10.0)


from sklearn.decomposition import PCA


def colorize_feature_map(x):
    """
    Args:
        x: torch.Tensor, [B, H, W, D]
    Returns:
        torch.Tensor, [B, H, W, 3]
    """
    device = x.device
    x = x.cpu().numpy()

    out = []
    for x_ in x:
        x_ = colorize_feature_map_np(x_)
        out.append(torch.from_numpy(x_).to(device))
    out = torch.stack(out).squeeze(0)
    return out


def colorize_feature_map_np(x):
    """
    Args:
        x: np.ndarray, [H, W, D]
    """
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(x.reshape(-1, x.shape[-1]))

    pca_features = (pca_features - pca_features.min()) / (
        pca_features.max() - pca_features.min()
    )
    pca_features = pca_features.reshape(x.shape[0], x.shape[1], 3)
    return pca_features
