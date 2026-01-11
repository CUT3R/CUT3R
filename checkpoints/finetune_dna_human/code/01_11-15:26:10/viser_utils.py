import torch
import os
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib as mpl
import cv2
import numpy as np
import matplotlib.cm as cm
import viser
import viser.transforms as tf
import time
import trimesh
import dataclasses
from scipy.spatial.transform import Rotation
from src.dust3r.viz import (
    add_scene_cam,
    CAM_COLORS,
    OPENGL,
    pts3d_to_trimesh,
    cat_meshes,
)


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


def segment_sky(image):
    import cv2
    from scipy import ndimage

    # Convert to HSV
    image = to_numpy(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.uint8(255 * image.clip(min=0, max=1))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for blue color and create mask
    lower_blue = np.array([0, 0, 100])
    upper_blue = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue).view(bool)

    # add luminous gray
    mask |= (hsv[:, :, 1] < 10) & (hsv[:, :, 2] > 150)
    mask |= (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 180)
    mask |= (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 220)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask2 = ndimage.binary_opening(mask, structure=kernel)

    # keep only largest CC
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

    # Apply mask
    return torch.from_numpy(mask3)


def convert_scene_output_to_glb(
    outdir,
    imgs,
    pts3d,
    mask,
    focals,
    cams2world,
    cam_size=0.05,
    show_cam=True,
    cam_color=None,
    as_pointcloud=False,
    transparent_cams=False,
    silent=False,
    save_name=None,
):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    if show_cam:
        for i, pose_c2w in enumerate(cams2world):
            if isinstance(cam_color, list):
                camera_edge_color = cam_color[i]
            else:
                camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
            add_scene_cam(
                scene,
                pose_c2w,
                camera_edge_color,
                None if transparent_cams else imgs[i],
                focals[i],
                imsize=imgs[i].shape[1::-1],
                screen_width=cam_size,
            )

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if save_name is None:
        save_name = "scene"
    outfile = os.path.join(outdir, save_name + ".glb")
    if not silent:
        print("(exporting 3D scene to", outfile, ")")
    scene.export(file_obj=outfile)
    return outfile


@dataclasses.dataclass
class CameraState(object):
    fov: float
    aspect: float
    c2w: np.ndarray

    def get_K(self, img_wh):
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return K


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


class PointCloudViewer:
    def __init__(
        self,
        model,
        state_args,
        pc_list,
        color_list,
        conf_list,
        cam_dict,
        image_mask=None,
        edge_color_list=None,
        device="cpu",
        port=8080,
        show_camera=True,
        vis_threshold=1,
        size=512
    ):
        self.model = model
        self.size=size
        self.state_args = state_args
        self.server = viser.ViserServer(port=port)
        self.server.set_up_direction("-y")
        self.device = device
        self.conf_list = conf_list
        self.vis_threshold = vis_threshold
        self.tt = lambda x: torch.from_numpy(x).float().to(device)
        self.pcs, self.all_steps = self.read_data(
            pc_list, color_list, conf_list, edge_color_list
        )
        self.cam_dict = cam_dict
        self.num_frames = len(self.all_steps)
        self.image_mask = image_mask
        self.show_camera = show_camera
        self.on_replay = False
        self.vis_pts_list = []
        self.traj_list = []
        self.orig_img_list = [x[0] for x in color_list]
        self.via_points = []

        gui_reset_up = self.server.gui.add_button(
            "Reset up direction",
            hint="Set the camera control 'up' direction to the current camera's 'up'.",
        )

        @gui_reset_up.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None
            client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                [0.0, -1.0, 0.0]
            )

        button3 = self.server.gui.add_button("4D (Only Show Current Frame)")
        button4 = self.server.gui.add_button("3D (Show All Frames)")
        self.is_render = False
        self.fourd = False

        @button3.on_click
        def _(event: viser.GuiEvent) -> None:
            self.fourd = True

        @button4.on_click
        def _(event: viser.GuiEvent) -> None:
            self.fourd = False

        self.focal_slider = self.server.add_gui_slider(
            "Focal Length",
            min=0.1,
            max=99999,
            step=1,
            initial_value=533,
        )

        self.psize_slider = self.server.add_gui_slider(
            "Point Size",
            min=0.0001,
            max=0.1,
            step=0.0001,
            initial_value=0.005,
        )
        self.camsize_slider = self.server.add_gui_slider(
            "Camera Size",
            min=0.01,
            max=0.5,
            step=0.01,
            initial_value=0.1,
        )

        self.pc_handles = []
        self.cam_handles = []

        @self.psize_slider.on_update
        def _(_) -> None:
            for handle in self.pc_handles:
                handle.point_size = self.psize_slider.value

        @self.camsize_slider.on_update
        def _(_) -> None:
            for handle in self.cam_handles:
                handle.scale = self.camsize_slider.value
                handle.line_thickness = 0.03 * handle.scale

        self.server.on_client_connect(self._connect_client)

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate(
                    [tf.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1
                ),
                [[0, 0, 0, 1]],
            ],
            0,
        )
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=c2w,
        )

    @staticmethod
    def generate_pseudo_intrinsics(h, w):
        focal = (h**2 + w**2) ** 0.5
        return np.array([[focal, 0, w // 2], [0, focal, h // 2], [0, 0, 1]]).astype(
            np.float32
        )

    def get_ray_map(self, c2w, h, w, intrinsics=None):
        if intrinsics is None:
            intrinsics = self.generate_pseudo_intrinsics(h, w)
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
        grid = np.stack([i, j, np.ones_like(i)], axis=-1)
        ro = c2w[:3, 3]
        rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
        rd = (c2w @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
        rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
        ro = np.broadcast_to(ro, (h, w, 3))
        ray_map = np.concatenate([ro, rd], axis=-1)
        return ray_map

    def set_camera_loc(camera, pose, K):
        """
        pose: 4x4 matrix
        K: 3x3 matrix
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        aspect = float(cx) / float(cy)
        fov = 2 * np.arctan(2 * cx / fx)
        wxyz_xyz = tf.SE3.from_matrix(pose).wxyz_xyz
        wxyz = wxyz_xyz[:4]
        xyz = wxyz_xyz[4:]
        camera.wxyz = wxyz
        camera.position = xyz
        camera.fov = fov
        camera.aspect = aspect

    def _connect_client(self, client: viser.ClientHandle):
        from src.dust3r.inference import inference_step
        from src.dust3r.utils.geometry import geotrf

        wxyz_panel = client.gui.add_text("wxyz:", f"{client.camera.wxyz}")
        position_panel = client.gui.add_text("position:", f"{client.camera.position}")
        fov_panel = client.gui.add_text(
            "fov:", f"{2 * np.arctan(self.size/self.focal_slider.value) * 180 / np.pi}"
        )
        aspect_panel = client.gui.add_text("aspect:", "1.0")

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            with self.server.atomic():
                wxyz_panel.value = f"{client.camera.wxyz}"
                position_panel.value = f"{client.camera.position}"
                fov_panel.value = (
                    f"{2 * np.arctan(self.size/self.focal_slider.value) * 180 / np.pi}"
                )
                aspect_panel.value = "1.0"

        gui_set_current_camera = client.gui.add_button(
            "Set Current Camera to Infer Raymap"
        )

        @gui_set_current_camera.on_click
        def _(_) -> None:
            try:
                cam = self.get_camera_state(client)
                cam.fov = 2 * np.arctan(self.size / self.focal_slider.value)
                cam.aspect = (512 / 384) if self.size==512 else 1.0
                pose = cam.c2w
                if self.size == 512:
                    intrins = self.generate_pseudo_intrinsics(384, 512)
                    raymap = torch.from_numpy(self.get_ray_map(pose, 384, 512, intrins))[
                        None
                    ].float()
                else:
                    intrins = self.generate_pseudo_intrinsics(224, 224)
                    raymap = torch.from_numpy(self.get_ray_map(pose, 224, 224, intrins))[
                        None
                    ].float()
                
                
                view = {
                    "img": torch.full((1, 3, 384, 512), torch.nan) if self.size==512 else torch.full((1, 3, 224, 224), torch.nan),
                    "ray_map": raymap,
                    "true_shape": torch.from_numpy(np.int32([raymap.shape[1:-1]])),
                    "idx": self.num_frames + 1,
                    "instance": str(self.num_frames + 1),
                    "camera_pose": torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(
                        0
                    ),
                    "img_mask": torch.tensor(False).unsqueeze(0),
                    "ray_mask": torch.tensor(True).unsqueeze(0),
                    "update": torch.tensor(False).unsqueeze(0),
                    "reset": torch.tensor(False).unsqueeze(0),
                }
                print("Start Inference Raymap")
                output = inference_step(
                    view, self.state_args[-1], self.model, device=self.device
                )
                print("Finish Inference Raymap")
                pts3ds = output["pred"]["pts3d_in_self_view"].cpu().numpy()
                pts3ds = geotrf(pose[None], pts3ds)
                colors = 0.5 * (output["pred"]["rgb"].cpu().numpy() + 1.0)
                depthmap = output["pred"]["pts3d_in_self_view"].cpu().numpy()[0][..., -1]
                conf = output["pred"]["conf"].cpu().numpy()
                disp = 1.0 / depthmap
                pts3ds, colors = self.parse_pc_data(pts3ds, colors, set_border_color=True)
                mask = (conf > 1.0).reshape(-1)
                self.num_frames += 1
                self.pc_handles.append(
                    self.server.add_point_cloud(
                        name=f"/frames/{self.num_frames-1}/pred_pts",
                        points=pts3ds[mask],
                        colors=colors[mask],
                        point_size=0.005,
                    )
                )

                self.server.add_camera_frustum(
                    name=f"/frames/{self.num_frames-1}/camera",
                    fov=cam.fov,
                    aspect=cam.aspect,
                    wxyz=client.camera.wxyz,
                    position=client.camera.position,
                    scale=0.1,
                    color=[64, 179, 230],
                )
                print("Adding new pointcloud: ", pts3ds.shape)
            except Exception as e:
                print(e)

    @staticmethod
    def set_color_border(image, border_width=5, color=[1, 0, 0]):

        image[:border_width, :, 0] = color[0]  # Red channel
        image[:border_width, :, 1] = color[1]  # Green channel
        image[:border_width, :, 2] = color[2]  # Blue channel
        image[-border_width:, :, 0] = color[0]
        image[-border_width:, :, 1] = color[1]
        image[-border_width:, :, 2] = color[2]

        image[:, :border_width, 0] = color[0]
        image[:, :border_width, 1] = color[1]
        image[:, :border_width, 2] = color[2]
        image[:, -border_width:, 0] = color[0]
        image[:, -border_width:, 1] = color[1]
        image[:, -border_width:, 2] = color[2]

        return image

    def read_data(self, pc_list, color_list, conf_list, edge_color_list=None):
        pcs = {}
        step_list = []
        for i, pc in enumerate(pc_list):
            step = i
            pcs.update(
                {
                    step: {
                        "pc": pc,
                        "color": color_list[i],
                        "conf": conf_list[i],
                        "edge_color": (
                            None if edge_color_list[i] is None else edge_color_list[i]
                        ),
                    }
                }
            )
            step_list.append(step)
        normalized_indices = (
            np.array(list(range(len(pc_list))))
            / np.array(list(range(len(pc_list)))).max()
        )
        cmap = cm.viridis
        self.camera_colors = cmap(normalized_indices)
        return pcs, step_list

    def parse_pc_data(
        self,
        pc,
        color,
        conf=None,
        edge_color=[0.251, 0.702, 0.902],
        set_border_color=False,
    ):

        pred_pts = pc.reshape(-1, 3)  # [N, 3]

        if set_border_color and edge_color is not None:
            color = self.set_color_border(color[0], color=edge_color)
        if np.isnan(color).any():

            color = np.zeros((pred_pts.shape[0], 3))
            color[:, 2] = 1
        else:
            color = color.reshape(-1, 3)
        if conf is not None:
            conf = conf[0].reshape(-1)
            pred_pts = pred_pts[conf > self.vis_threshold]
            color = color[conf > self.vis_threshold]
        return pred_pts, color

    def add_pc(self, step):
        pc = self.pcs[step]["pc"]
        color = self.pcs[step]["color"]
        conf = self.pcs[step]["conf"]
        edge_color = self.pcs[step].get("edge_color", None)

        pred_pts, color = self.parse_pc_data(
            pc, color, conf, edge_color, set_border_color=True
        )

        self.vis_pts_list.append(pred_pts)
        self.pc_handles.append(
            self.server.add_point_cloud(
                name=f"/frames/{step}/pred_pts",
                points=pred_pts,
                colors=color,
                point_size=0.005,
            )
        )

    def add_camera(self, step):
        cam = self.cam_dict
        focal = cam["focal"][step]
        pp = cam["pp"][step]
        R = cam["R"][step]
        t = cam["t"][step]

        q = tf.SO3.from_matrix(R).wxyz
        fov = 2 * np.arctan(pp[0] / focal)
        aspect = pp[0] / pp[1]
        self.traj_list.append((q, t))
        self.cam_handles.append(
            self.server.add_camera_frustum(
                name=f"/frames/{step}/camera",
                fov=fov,
                aspect=aspect,
                wxyz=q,
                position=t,
                scale=0.1,
                color=(50, 205, 50),
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
                disabled=False,
            )
            gui_next_frame = self.server.add_gui_button("Next Step", disabled=False)
            gui_prev_frame = self.server.add_gui_button("Prev Step", disabled=False)
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
                self.frame_nodes[current_timestep].visible = True
                self.frame_nodes[prev_timestep].visible = False
            prev_timestep = current_timestep
            self.server.flush()  # Optional!

        self.server.add_frame(
            "/frames",
            show_axes=False,
        )
        self.frame_nodes = []
        for i in range(self.num_frames):
            step = self.all_steps[i]
            self.frame_nodes.append(
                self.server.add_frame(
                    f"/frames/{step}",
                    show_axes=False,
                )
            )
            self.add_pc(step)
            if self.show_camera:
                self.add_camera(step)

        prev_timestep = gui_timestep.value
        while True:
            if self.on_replay:
                pass
            else:
                if gui_playing.value:
                    gui_timestep.value = (gui_timestep.value + 1) % self.num_frames

                for i, frame_node in enumerate(self.frame_nodes):
                    frame_node.visible = (
                        i <= gui_timestep.value
                        if not self.fourd
                        else i == gui_timestep.value
                    )

            time.sleep(1.0 / gui_framerate.value)

    def run(self):
        self.animate()
        while True:
            time.sleep(10.0)
