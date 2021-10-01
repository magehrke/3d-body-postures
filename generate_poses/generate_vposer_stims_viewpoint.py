import os
import cv2
import torch
import trimesh
import numpy as np
from tqdm import tqdm
import imageio as iio
import scipy.linalg as sl
from human_body_prior.body_model.body_model import BodyModelWithPoser
from human_body_prior.mesh.mesh_viewer import MeshViewer
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import id_generator, makepath
from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_
import generate_utils as g_utils


def rand_ndim_onb(ndim):
    '''
    X=[];
    for i = 31:-1:1
    R = (rand(32,i)-0.5)*2;
    N = null([X R]');
    X = [X N];
    end
    X = [X null(X')];
    A=X'*X;
    mean(A(triu(A,1)>0))
    '''
    X = np.zeros([ndim, ndim])
    for i in range(ndim):
        R = (np.random.rand(ndim,ndim-(i+1))-0.5)*2
        if i>0:
            N = sl.null_space(np.concatenate((X[:,0:i],R),axis=1).transpose())
        else:
            N = sl.null_space(R.transpose())
        X[:,i]=N.squeeze()

    return X


def perspective_projection(points, rotation, translation,
                           focal_length, camera_center):
    """
    This function computes the perspective projection of a set of points.
    Input:
        points (bs, N, 3): 3D points
        rotation (bs, 3, 3): Camera rotation
        translation (bs, 3): Camera translation
        focal_length (bs,) or scalar: Focal length
        camera_center (bs, 2): Camera center
    """
    batch_size = points.shape[0]
    K = torch.zeros([batch_size, 3, 3], device=points.device)
    K[:,0,0] = focal_length
    K[:,1,1] = focal_length
    K[:,2,2] = 1.
    K[:,:-1, -1] = camera_center

    # Transform points
    points = torch.einsum('bij,bkj->bki', rotation, points)
    points = points + translation.unsqueeze(1)

    # Apply perspective distortion
    projected_points = points / points[:,:,-1].unsqueeze(-1)

    # Apply camera intrinsics
    projected_points = torch.einsum('bij,bkj->bki', K, projected_points)

    return projected_points[:, :, :-1]


class GenerateVposerStimsViewpoint:
    def __init__(self, smpl_exp_dir: str, bm_path: str, out_dir: str, device: torch.device):
        self.smpl_exp_dir = smpl_exp_dir
        self.bm_path = bm_path
        self.out_dir = out_dir
        self.device = device

        self.bm = BodyModelWithPoser(bm_path=bm_path, batch_size=1, model_type='smplx', poser_type='vposer',
                                     smpl_exp_dir=smpl_exp_dir).to('cuda')
        self.bm2 = BodyModelWithPoser(bm_path=bm_path, batch_size=1, model_type='smplx', poser_type='vposer',
                                      smpl_exp_dir=smpl_exp_dir).to('cuda')
        self.bm3 = BodyModelWithPoser(bm_path=bm_path, batch_size=1, model_type='smplx', poser_type='vposer',
                                      smpl_exp_dir=smpl_exp_dir).to('cuda')

        self.num_interpol = 18
        self.view_angles = np.linspace(0, 2, num=self.num_interpol)
        self.imw, self.imh = 400, 400
        self.fraction = 15

        # TODO
        self.poz_sel = np.arange(32)
        # TODO
        self.nrand = 6

        # Viewpoint
        self.num_vp = 3
        self.vp = [-45, 0, 45]
        self.scale = [8, 8, 32, 32, 96, 96]

        # TODO
        self.nexp = self.poz_sel.shape[0] * self.nrand

        self.t3mat = np.zeros((self.nexp, self.num_vp, self.num_interpol, 32))
        self.kp2dmat = np.zeros((self.nexp, self.num_vp, self.num_interpol, 55, 2))
        self.kp3dmat = np.zeros((self.nexp, self.num_vp, self.num_interpol, 55, 3))
        self.poseaamat = np.zeros((self.nexp, self.num_vp, self.num_interpol, 63))

        self.posemat = np.zeros((self.nexp, 32))
        self.mat2d = np.zeros((self.nexp, self.num_vp, 55))
        self.viewpointmat = np.zeros((self.nexp, self.num_vp))

        for i in range(self.nrand):
            X = rand_ndim_onb(self.poz_sel.shape[0])
            self.posemat[i * self.poz_sel.shape[0]:(i + 1) * self.poz_sel.shape[0], self.poz_sel] \
                = X.transpose() * self.scale[i]

        # Save activations of decoding layers
        # Will be overwritten everytime forward() or decode() is again
        self.activation = {}
        self.bm3.poser_body_pt.bodyprior_dec_fc1.register_forward_hook(self.activation_hook('bodyprior_dec_fc1'))
        self.bm3.poser_body_pt.bodyprior_dec_fc2.register_forward_hook(self.activation_hook('bodyprior_dec_fc2'))
        self.bm3.poser_body_pt.bodyprior_dec_out.register_forward_hook(self.activation_hook('bodyprior_dec_out'))

    def activation_hook(self, name):
        def hook(model, input, output):
            self.activation[name] = output.detach()
        return hook

    def create_poses(self):
        for i in tqdm(range(0, self.nexp)):
            self.bm2.randomize_pose()

            t1 = self.bm.poZ_body.new(self.posemat[i, :]).detach()
            t2 = self.bm2.poZ_body.data

            mv = MeshViewer(width=self.imw, height=self.imh, use_offscreen=True)

            for vp_id in range(3):
                images = np.zeros([self.num_interpol, self.imh, self.imw, 3])
                self.viewpointmat[i, vp_id] = self.vp[vp_id]
                for ipol_id in range(0, self.num_interpol):
                    alpha = 1 - (ipol_id / (self.num_interpol * self.fraction))
                    t3 = (alpha * t1 + (1 - alpha) * t2)
                    self.t3mat[i, vp_id, ipol_id, :] = t3.detach().cpu().numpy()

                    self.bm3.poZ_body.data[:] = t3

                    self.bm3.pose_body.data[:] = self.bm3.poser_body_pt.decode(self.bm3.poZ_body, output_type='aa').view(
                        self.bm3.batch_size, -1)

                    self.poseaamat[i, vp_id, ipol_id, :] = torch.squeeze(self.bm3.pose_body).detach().cpu().numpy()

                    points = self.bm3.forward().Jtr  # joints

                    body_mesh = trimesh.Trimesh(vertices=c2c(self.bm3.forward().v)[0], faces=c2c(self.bm.f),
                                                vertex_colors=np.tile([135, 250, 206],
                                                                      (c2c(self.bm3.forward().v).shape[1], 1)))

                    # Calculate and save kinematic 2D and 3D points
                    # This also changes 'body_mesh' in place - todo
                    self.kp3dmat[i, vp_id, ipol_id, :, :] = torch.squeeze(points).detach().cpu().numpy()
                    proj_2d_points = self._calculate_2d_points(vp_id, ipol_id, points, body_mesh, mv)
                    self.kp2dmat[i, vp_id, ipol_id, :, :] = torch.squeeze(proj_2d_points).detach().cpu().numpy()

                    mv.set_meshes([body_mesh], group_name='static')
                    images[ipol_id] = mv.render()

                self._save_images(images, i, vp_id)

    def _calculate_2d_points(self, vp_id, ipol_id, points, body_mesh, mv):
        # Calculate projected points
        points_rot = self._rotate_points(vp_id, ipol_id, points, body_mesh)
        rotation, translation, focal_length = g_utils.get_camera_params(mv, self.device)
        camera_center = 0.5 * self.imh * torch.ones(1, 2, device=self.device, dtype=torch.float32)
        projected_points = perspective_projection(points_rot.float(), rotation, translation,
                                                  focal_length, camera_center)
        return projected_points

    def _rotate_points(self, vp_id: int, ipol_id: int, points, body_mesh: trimesh.Trimesh):
        T = trimesh.transformations.translation_matrix([0, 0.2, 0])
        R = trimesh.transformations.rotation_matrix(np.radians(self.view_angles[ipol_id] + self.vp[vp_id]), (0, 1, 0))
        apply_mesh_tranfsormations_([body_mesh], T)
        apply_mesh_tranfsormations_([body_mesh], R)

        # rotate point as we do the mesh
        p2 = torch.cat((points, torch.ones((1, 55, 1)).to('cuda')), 2)
        p3 = torch.matmul(torch.from_numpy(T).to('cuda'), torch.t(p2.squeeze()).double())
        p3 = torch.matmul(torch.from_numpy(R).to('cuda'), p3)
        return torch.unsqueeze(p3[:3, :].t(), 0)

    def _save_images(self, images: np.array, iteration: int, vp_id: int):
        """
        Save pose as gif and each interpolation step as png.
        In total there a 2 x num_interpolation images, because
        the images are duplicated and added in reverse order.
        """
        images = images.astype(np.uint8)
        a1 = list(images)
        a12 = a1.copy()
        a12.reverse()
        a1.extend(a12)
        iio.mimsave(os.path.join(self.out_dir, 'gif', 'VAE_%02d_%02d.gif' % (iteration, vp_id)), a1, duration=1 / 24)

        for ipol_id, imgi in enumerate(a1):
            cv2.imwrite(os.path.join(self.out_dir, 'png', 'VAE_%02d_%02d_%02d.png' % (iteration, vp_id, ipol_id)), imgi)

    def save_numpy_arrays(self):
        np.save(os.path.join(self.out_dir, 'params_time_VAE2.npy'), self.t3mat)
        np.save(os.path.join(self.out_dir, 'params_viewpoint_VAE2.npy'), self.viewpointmat)
        np.save(os.path.join(self.out_dir, 'kp2dmat.npy'), self.kp2dmat)
        np.save(os.path.join(self.out_dir, 'kp3dmat.npy'), self.kp3dmat)
        np.save(os.path.join(self.out_dir, 'poseaamat.npy'), self.poseaamat)
        np.save(os.path.join(self.out_dir, 'exp_params.npy'), [self.num_interpol, self.nrand])
        np.save(os.path.join(self.out_dir, 'scale.npy'), self.scale)
        np.save(os.path.join(self.out_dir, 'poz_sel.npy'), self.poz_sel)


if __name__ == "__main__":
    # Experiment directory & body model
    # Obtain from https://smpl-x.is.tue.mpg.de/downloads
    _smpl_exp_dir = '../data/vposer_v1_0/'
    _bm_path = '../data/models/smplx/SMPLX_MALE.npz'

    _out_dir = makepath(os.path.join('../data/evaluations', 'blapose_stims_vp_p17'))
    makepath(os.path.join(_out_dir, 'gif'))
    makepath(os.path.join(_out_dir, 'png'))
    print(f'Output directory: {_out_dir}')

    _device = torch.device('cuda')

    # Execute
    generator = GenerateVposerStimsViewpoint(_smpl_exp_dir, _bm_path, _out_dir, _device)
    generator.create_poses()
    generator.save_numpy_arrays()









