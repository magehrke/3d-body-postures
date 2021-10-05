import os
import cv2
import torch
import trimesh
import numpy as np
from tqdm import tqdm
import imageio as iio
import scipy.linalg
import scipy.io
from human_body_prior.body_model.body_model import BodyModelWithPoser
from human_body_prior.mesh.mesh_viewer import MeshViewer
from human_body_prior.tools.omni_tools import copy2cpu as c2c
from human_body_prior.tools.omni_tools import id_generator, makepath
from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_
import generate_utils as g_utils


def rand_ndim_onb(ndim):
    """
    Create a random n-dimensional orthonormal basis.
    """
    x = np.zeros([ndim, ndim])
    for i in range(ndim):
        # Random vector in [-1, 1]
        r = (np.random.rand(ndim, ndim-(i+1))-0.5)*2
        if i > 0:
            n = scipy.linalg.null_space(np.concatenate((x[:, 0:i], r), axis=1).transpose())
        else:
            n = scipy.linalg.null_space(r.transpose())
        x[:, i] = n.squeeze()
    return x


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


def load_poz_stimuli_from_mat_file(mat_file_path: str) -> (np.array, np.array):
    """
    Load and stack the latent stimuli (poz) arrays of a matlab file.
    """
    mat_file = scipy.io.loadmat(mat_file_path)['body']
    raw_stimuli = mat_file['param']
    raw_ids = mat_file['uparam']

    poz_ids = []
    poz_stimuli = []
    for i in range(0, len(raw_stimuli)):
        poz_stimuli.append(raw_stimuli[i][0])
        poz_ids.append(raw_ids[i][0])
    poz_ids = np.vstack(poz_ids)
    poz_stimuli = np.vstack(poz_stimuli)
    assert poz_ids.shape[0] == poz_stimuli.shape[0]

    poz_ids_unique, indices = np.unique(poz_ids, return_index=True)
    return poz_ids_unique, poz_stimuli[indices]


def generate_random_poz_stimuli(out_dir: str) -> (np.array, np.array):
    # Description of experiment parameter
    stim_params = {
        # Some arbitrary array that creates poses
        'poz_sel_array': np.arange(32),
        # The number of orthonormal basis we will create
        # We will use each column of each basis as latent stimuli (poZ)
        # Advantage of using ONB: we create stimuli in each latent direction
        'num_onb': 6,
        # Scaling of each orthonormal basis
        # The higher the scaling, the more extreme the samples
        # The latent space (poZ) is constructed by using centered normal distributions
        'scale': [8, 8, 32, 32, 96, 96]
    }

    # Create orthonormal basis, scale it and use the columns to fill poz stimuli vector
    poz_mat = np.zeros((stim_params['poz_sel_array'].shape[0] * stim_params['num_onb'], 32))
    for i in range(stim_params['num_onb']):
        x = rand_ndim_onb(stim_params['poz_sel_array'].shape[-1])
        poz_mat[i * stim_params['poz_sel_array'].shape[-1]: (i + 1) * stim_params['poz_sel_array'].shape[0],
                stim_params['poz_sel_array']] = x.transpose() * stim_params['scale'][i]

    uparam = np.arange(1, poz_mat.shape[0]+1)
    np.save(os.path.join(out_dir, 'stim_creation_params.npy'), stim_params)
    return uparam, poz_mat


class GenerateVposerStimsViewpoint:
    def __init__(self, smpl_exp_dir: str, bm_path: str, out_dir: str,
                 device: torch.device, poz_mat: np.array = None, uparam: np.array = None):
        """
        :param poz_mat: Numpy array with the latent stimuli to create poses.
            The array must have dimensions [X, 32]. If poz_mat is not given, then
            random latent vectors and poses are created.
        :param uparam: Unique ID's for each stimuli, which we add when saving data.
        """
        self.smpl_exp_dir = smpl_exp_dir
        self.bm_path = bm_path
        self.out_dir = out_dir
        self.device = device

        self.bm = BodyModelWithPoser(bm_path=bm_path, batch_size=1, model_type='smplx', poser_type='vposer',
                                     smpl_exp_dir=smpl_exp_dir).to('cuda')

        self.num_interpol = 18
        self.view_angles = np.linspace(0, 2, num=self.num_interpol)
        self.imw, self.imh = 400, 400
        self.fraction = 15

        if poz_mat is not None:
            assert poz_mat.shape[1] == 32
            self.poz_mat = poz_mat
            assert uparam.shape[0] == poz_mat.shape[0]
            self.uparam  = uparam
        else:
            self.uparam, self.poz_mat = generate_random_poz_stimuli(out_dir=self.out_dir)

        # Number of latent (poZ) stimuli
        self.n_stim = self.poz_mat.shape[0]

        # Viewpoint
        self.num_vp = 3
        self.vp = [-45, 0, 45]
        self.viewpointmat = np.zeros((self.n_stim, self.num_vp))

        self.t3mat = np.zeros((self.n_stim, self.num_vp, self.num_interpol, 32))
        self.kp2dmat = np.zeros((self.n_stim, self.num_vp, self.num_interpol, 55, 2))
        self.kp3dmat = np.zeros((self.n_stim, self.num_vp, self.num_interpol, 55, 3))
        self.poseaamat = np.zeros((self.n_stim, self.num_vp, self.num_interpol, 63))

        self.mat2d = np.zeros((self.n_stim, self.num_vp, 55))

        # Save activations of decoding layers
        self.act = {}  # will be overwritten everytime forward() or decode() is called
        self.act_fc1 = np.zeros((self.n_stim, 512))
        self.act_fc2 = np.zeros((self.n_stim, 512))
        self.act_out = np.zeros((self.n_stim, 126))

        # Register activation forward hooks
        self.bm.poser_body_pt.bodyprior_dec_fc1.register_forward_hook(self._activation_hook('bodyprior_dec_fc1'))
        self.bm.poser_body_pt.bodyprior_dec_fc2.register_forward_hook(self._activation_hook('bodyprior_dec_fc2'))
        self.bm.poser_body_pt.bodyprior_dec_out.register_forward_hook(self._activation_hook('bodyprior_dec_out'))

    def _activation_hook(self, name):
        def hook(model, input, output):
            self.act[name] = output.detach()
        return hook

    def create_poses(self):
        for i in tqdm(range(self.n_stim)):

            t1 = self.bm.poZ_body.new(self.poz_mat[i, :]).detach()
            mv = MeshViewer(width=self.imw, height=self.imh, use_offscreen=True)

            for vp_id in range(3):
                images = np.zeros([self.num_interpol, self.imh, self.imw, 3])
                self.viewpointmat[i, vp_id] = self.vp[vp_id]
                for ipol_id in range(0, self.num_interpol):
                    alpha = 1 - (ipol_id / (self.num_interpol * self.fraction))
                    # Interpolation in the direction of the zero vector
                    t2 = (alpha * t1 + (1 - alpha) * torch.zeros([1, 32]).to('cuda'))
                    self.t3mat[i, vp_id, ipol_id, :] = t2.detach().cpu().numpy()

                    self.bm.poZ_body.data[:] = t2

                    self.bm.pose_body.data[:] = \
                        self.bm.poser_body_pt.decode(self.bm.poZ_body, output_type='aa').view(self.bm.batch_size, -1)

                    # Save activations of neural network
                    self.act_fc1[i, :] = torch.squeeze(self.act['bodyprior_dec_fc1']).detach().cpu().numpy()
                    self.act_fc2[i, :] = torch.squeeze(self.act['bodyprior_dec_fc2']).detach().cpu().numpy()
                    self.act_out[i, :] = torch.squeeze(self.act['bodyprior_dec_out']).detach().cpu().numpy()

                    self.poseaamat[i, vp_id, ipol_id, :] = torch.squeeze(self.bm.pose_body).detach().cpu().numpy()

                    points = self.bm.forward().Jtr  # 55 joints ?

                    body_mesh = trimesh.Trimesh(vertices=c2c(self.bm.forward().v)[0], faces=c2c(self.bm.f),
                                                vertex_colors=np.tile([135, 250, 206],
                                                                      (c2c(self.bm.forward().v).shape[1], 1)))

                    # Apply transformations to the mesh
                    T = trimesh.transformations.translation_matrix([0, 0.2, 0])
                    radians = np.radians(self.view_angles[ipol_id] + self.vp[vp_id])
                    R = trimesh.transformations.rotation_matrix(radians, (0, 1, 0))
                    apply_mesh_tranfsormations_([body_mesh], T)
                    apply_mesh_tranfsormations_([body_mesh], R)

                    mv.set_meshes([body_mesh], group_name='static')
                    images[ipol_id] = mv.render()

                    # Calculate and save kinematic 2D and 3D points
                    self.kp3dmat[i, vp_id, ipol_id, :, :] = torch.squeeze(points).detach().cpu().numpy()
                    proj_2d_points = self._calculate_2d_points(points, mv, T, R)
                    self.kp2dmat[i, vp_id, ipol_id, :, :] = torch.squeeze(proj_2d_points).detach().cpu().numpy()

                self._save_images(images, i, vp_id)

    def _calculate_2d_points(self, points, mv, T, R):
        # Calculate projected points
        points_rot = self._rotate_points(points, T, R)
        rotation, translation, focal_length = g_utils.get_camera_params(mv, self.device)
        camera_center = 0.5 * self.imh * torch.ones(1, 2, device=self.device, dtype=torch.float32)
        projected_points = perspective_projection(points_rot.float(), rotation, translation,
                                                  focal_length, camera_center)
        return projected_points

    @staticmethod
    def _rotate_points(points, T, R):
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
        iio.mimsave(os.path.join(self.out_dir, 'gif', 'Stim_uparam_%d_Viewpoint_%d.gif'
                                 % (self.uparam[i], vp_id+1)), a1, duration=1 / 24)

        for ipol_id, imgi in enumerate(a1):
            cv2.imwrite(os.path.join(self.out_dir, 'png', 'Stim_uparam_%d_Viewpoint_%d_Ipol_%d.png'
                                     % (self.uparam[i], vp_id+1, ipol_id)), imgi)

    def save_numpy_arrays(self):
        np.save(os.path.join(self.out_dir, 'params_time_VAE2.npy'), self.t3mat)
        np.save(os.path.join(self.out_dir, 'params_viewpoint_VAE2.npy'), self.viewpointmat)
        np.save(os.path.join(self.out_dir, 'kp2dmat.npy'), self.kp2dmat)
        np.save(os.path.join(self.out_dir, 'kp3dmat.npy'), self.kp3dmat)
        np.save(os.path.join(self.out_dir, 'poseaamat.npy'), self.poseaamat)
        np.save(os.path.join(self.out_dir, 'exp_params.npy'), [self.num_interpol])
        act_dict = {'bodyprior_dec_fc1': self.act_fc1, 'bodyprior_dec_fc2': self.act_fc2,
                    'bodyprior_dec_out': self.act_out}
        if self.uparam is not None:
            act_dict['uparam'] = self.uparam
        scipy.io.savemat(os.path.join(self.out_dir, 'activations.npy'), act_dict)


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

    # Get stimuli
    poz_ids, poz_mat = load_poz_stimuli_from_mat_file('../data/VAEparams.mat')

    # Execute
    generator = GenerateVposerStimsViewpoint(_smpl_exp_dir, _bm_path, _out_dir, _device, poz_mat=poz_mat, uparam=poz_ids)
    generator.create_poses()
    generator.save_numpy_arrays()
