import argparse
import numpy as np
import torch
import gtracer
import dearpygui.dearpygui as dpg
from scipy.spatial.transform import Rotation as R
from gaussian import GaussianModel
from PIL import Image
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text

class OrbitCamera:
    def __init__(self, W, H, r=2, fovx=60,fovy=60):
        self.W = W
        self.H = H
        self.radius = r # camera distance from center
        self.fovy = fovy # in degree
        self.fovx = fovx
        self.center = np.array([0, 0, 0], dtype=np.float32) # look at this point
        self.rot = R.from_quat([1, 0, 0, 0]) # init camera matrix: [[1, 0, 0], [0, -1, 0], [0, 0, 1]] (to suit ngp convention)
        self.up = np.array([0, 1, 0], dtype=np.float32) # need to be normalized!

    # pose
    @property
    def pose(self):
        # first move camera to radius
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        # rotate
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res
        # translate
        res[:3, 3] -= self.center
        return res
    
    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])
    
    def orbit(self, dx, dy):
        # rotate along camera up/side axis!
        side = self.rot.as_matrix()[:3, 0] # why this is side --> ? # already normalized.
        rotvec_x = self.up * np.radians(-0.05 * dx)
        rotvec_y = side * np.radians(-0.05 * dy)
        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        self.radius *= 1.1 ** (-delta)

    def pan(self, dx, dy, dz=0):
        # pan in camera coordinate system (careful on the sensitivity!)
        self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, dy, dz])

    def from_colmap(path,self, cam_extrinsics, cam_intrinsics, images_folder):
        # 假设 Colmap 的内外参数已经通过其它方式读取并传入
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        reading_dir = "images"
        cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))

        for idx, key in enumerate(cam_extrinsics):
            extr = cam_extrinsics[key]
            intr = cam_intrinsics[extr.camera_id]
            height = intr.height
            width = intr.width

            uid = intr.id
            R_mat = np.transpose(qvec2rotmat(extr.qvec))  # 外参中的旋转矩阵
            T = np.array(extr.tvec)

            # 更新相机的位置和方向
            self.rot = R.from_matrix(R_mat)
            self.center = T  # 相机中心点

            if intr.model=="SIMPLE_PINHOLE":
                focal_length_x = intr.params[0]
                FovY = focal2fov(focal_length_x, height)
                FovX = focal2fov(focal_length_x, width)
            elif intr.model=="PINHOLE":
                focal_length_x = intr.params[0]
                focal_length_y = intr.params[1]
                FovY = focal2fov(focal_length_y, height)
                FovX = focal2fov(focal_length_x, width)
            else:
                assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

            image_path = os.path.join(images_folder, os.path.basename(extr.name))
            image_name = os.path.basename(image_path).split(".")[0]
            image = Image.open(image_path)
            
            self.fovx = Fovx
            self.fovy = FovY
            self.radius = np.linalg.norm(self.center)  # 计算相机到中心的距离


def get_rays(poses, intrinsics, H, W, N=-1, error_map=None):
    ''' get rays
    Args:
        poses: [B, 4, 4], cam2world
        intrinsics: [4]
        H, W, N: int
        error_map: [B, 128 * 128], sample probability based on training error
    Returns:
        rays_o, rays_d: [B, N, 3]
        inds: [B, N]
    '''

    device = poses.device
    B = poses.shape[0]
    fx, fy, cx, cy = intrinsics

    i, j = torch.meshgrid(torch.linspace(0, W-1, W, device=device), torch.linspace(0, H-1, H, device=device), indexing='ij')
    i = i.t().reshape([1, H*W]).expand([B, H*W]) + 0.5
    j = j.t().reshape([1, H*W]).expand([B, H*W]) + 0.5

    results = {}

    if N > 0:
        N = min(N, H*W)

        if error_map is None:
            inds = torch.randint(0, H*W, size=[N], device=device) # may duplicate
            inds = inds.expand([B, N])
        else:

            # weighted sample on a low-reso grid
            inds_coarse = torch.multinomial(error_map.to(device), N, replacement=False) # [B, N], but in [0, 128*128)

            # map to the original resolution with random perturb.
            inds_x, inds_y = inds_coarse // 128, inds_coarse % 128 # `//` will throw a warning in torch 1.10... anyway.
            sx, sy = H / 128, W / 128
            inds_x = (inds_x * sx + torch.rand(B, N, device=device) * sx).long().clamp(max=H - 1)
            inds_y = (inds_y * sy + torch.rand(B, N, device=device) * sy).long().clamp(max=W - 1)
            inds = inds_x * W + inds_y

            results['inds_coarse'] = inds_coarse # need this when updating error_map

        i = torch.gather(i, -1, inds)
        j = torch.gather(j, -1, inds)

        results['inds'] = inds

    else:
        inds = torch.arange(H*W, device=device).expand([B, H*W])

    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    directions = torch.stack((xs, ys, zs), dim=-1)
    directions = directions / torch.norm(directions, dim=-1, keepdim=True)
    rays_d = directions @ poses[:, :3, :3].transpose(-1, -2) # (B, N, 3)

    rays_o = poses[..., :3, 3] # [B, 3]
    rays_o = rays_o[..., None, :].expand_as(rays_d) # [B, N, 3]
    
    results['rays_o'] = rays_o
    results['rays_d'] = rays_d

    return results
    
        
class GUI:
    def __init__(self, opt, debug=True):
        self.opt = opt # shared with the trainer's opt to support in-place modification of rendering parameters.
        self.W = opt.W
        self.H = opt.H
        self.debug = debug
        self.cam = OrbitCamera(opt.W, opt.H, r=opt.radius, fovy=opt.fovy)
        self.bg_color = torch.ones(3, dtype=torch.float32) # default white bg

        self.render_buffer = np.zeros((self.W, self.H, 3), dtype=np.float32)
        self.need_update = True # camera moved, should reset accumulation
        
        self.mode = 'render'

        self.gaussians = GaussianModel(0)
        self.gaussians.load_ply(opt.point_cloud)
        self.gaussians.build_bvh()
        
        #dpg.create_context()
        #self.register_dpg()
        self.pic_number=0
        self.step()
        
        

    def __del__(self):
        dpg.destroy_context()


    def prepare_buffer(self, outputs):
        if self.mode == 'render':
            colors = outputs['render'].detach().cpu().numpy().reshape(self.H, self.W, 3)
            return colors
        elif self.mode == 'depth':
            depth = outputs['depth'].detach().cpu().numpy().reshape(self.H, self.W, 1)
            mask = depth >= 10
            mn = depth[~mask].min()
            mx = depth[~mask].max()
            depth = (depth - mn) / (mx - mn + 1e-5)
            depth[mask] = 0
            depth = depth.repeat(3, -1)
            return depth
        elif self.mode == 'alpha':
            alpha = outputs['alpha'].detach().cpu().numpy().reshape(self.H, self.W, 1)
            alpha = alpha.repeat(3, -1)
            return alpha
        else:
            raise NotImplementedError()

    
    def step(self):

        if self.need_update:
            pose = torch.from_numpy(self.cam.pose).unsqueeze(0).cuda()
            print("intrinsics")
            print(self.cam.intrinsics)
            rays = get_rays(pose, self.cam.intrinsics, self.H, self.W, -1)
            rays_o = rays['rays_o'].contiguous().view(-1, 3)
            rays_d = rays['rays_d'].contiguous().view(-1, 3)
            
            SinvR = self.gaussians.get_SinvR()
            means3D = self.gaussians.get_xyz
            shs = self.gaussians.get_features
            opacity = self.gaussians.get_opacity
            
            starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            starter.record()
            outputs = self.gaussians.trace(rays_o, rays_d)
            ender.record()
            torch.cuda.synchronize()
            t = starter.elapsed_time(ender)
            '''
            if self.need_update:
                self.render_buffer = self.prepare_buffer(outputs)
                self.need_update = False
            else:
                self.render_buffer = (self.render_buffer * self.spp + self.prepare_buffer(outputs)) / (self.spp + 1)
            '''
            self.render_buffer = self.prepare_buffer(outputs)
            print(self.render_buffer.shape)
            #dpg.set_value("_log_infer_time", f'{t:.4f}ms ({int(1000/t)} FPS)')
            #dpg.set_value("_texture", self.render_buffer)
            
            out_path=f"/home/qinhaoran/cuda_output/output_image_{self.pic_number}.png"
            image_np = (self.render_buffer * 255).astype(np.uint8)
            pic = Image.fromarray(image_np)
            pic = pic.rotate(180)
            pic.save(out_path)

        
    def register_dpg(self):

        ### register texture 

        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.W, self.H, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window

        # the rendered image, as the primary window
        with dpg.window(tag="_primary_window", width=self.W, height=self.H):

            # add the texture
            dpg.add_image("_texture")

        dpg.set_primary_window("_primary_window", True)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=300, height=200):

            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)              

            with dpg.group(horizontal=True):
                dpg.add_text("Infer time: ")
                dpg.add_text("no data", tag="_log_infer_time")
            
            # rendering options
            with dpg.collapsing_header(label="Options", default_open=True):

                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True
                
                dpg.add_combo(('render', 'depth', 'alpha'), label='mode', default_value=self.mode, callback=callback_change_mode)

                # # bg_color picker
                # def callback_change_bg(sender, app_data):
                #     self.bg_color = torch.tensor(app_data[:3], dtype=torch.float32) # only need RGB in [0, 1]
                #     self.need_update = True

                # dpg.add_color_edit((255, 255, 255), label="Background Color", width=200, tag="_color_editor", no_alpha=True, callback=callback_change_bg)

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = app_data
                    self.need_update = True

                dpg.add_slider_int(label="FoV (vertical)", min_value=1, max_value=120, format="%d deg", default_value=self.cam.fovy, callback=callback_set_fovy)
                
            # debug info
            if self.debug:
                with dpg.collapsing_header(label="Debug"):
                    # pose
                    dpg.add_separator()
                    dpg.add_text("Camera Pose:")
                    dpg.add_text(str(self.cam.pose), tag="_log_pose")


        ### register camera handler

        def callback_camera_drag_rotate(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_wheel_scale(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        def callback_camera_drag_pan(sender, app_data):

            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

            if self.debug:
                dpg.set_value("_log_pose", str(self.cam.pose))


        with dpg.handler_registry():
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, callback=callback_camera_drag_rotate)
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan)

        
        dpg.create_viewport(title='mesh viewer', width=self.W, height=self.H, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        #dpg.show_metrics()

        dpg.show_viewport()


    def render(self):
        
        for i in range(32):
            print(self.pic_number)
            self.step()
            self.pic_number+=1
            dx=-10
            dy=-10
            self.cam.orbit(dx, dy)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--point_cloud', default='point_cloud.ply', type=str)
    parser.add_argument('--W', type=int, default=980, help="GUI width")
    parser.add_argument('--H', type=int, default=544, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")

    opt = parser.parse_args()

    gui = GUI(opt)
    gui.render()
