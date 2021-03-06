from human_body_prior.body_model.body_model import BodyModelWithPoser
import trimesh
from human_body_prior.tools.omni_tools import copy2cpu as c2c
import os
from human_body_prior.tools.omni_tools import colors, makepath
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.mesh.mesh_viewer import MeshViewer
from human_body_prior.tools.visualization_tools import imagearray2file, smpl_params2ply
from human_body_prior.tools.omni_tools import id_generator, makepath
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch.nn as nn
import scipy.spatial.distance as sd
import scipy.linalg as sl
from smplx.lbs import lbs

from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_

#from SPIN.utils.geometry import perspective_projection

import imageio as iio

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



# smpl_exp_dir = "D:\\smpl\\vposer_v1_0"
smpl_exp_dir = os.path.join('D:\\smpl\\vposer_v1_0')

# directory for the trained model along with the model code. obtain from https://smpl-x.is.tue.mpg.de/downloads
# D:\smpl\human_body_prior-master

bm_path = 'D:\smpl\models\smplx\SMPLX_MALE.npz'  # obtain from https://smpl-x.is.tue.mpg.de/downloads
# bm_path = 'D:\smpl\models\smpl\SMPL.pkl'

device = torch.device('cuda')

bm = BodyModelWithPoser(bm_path=bm_path, batch_size=1, model_type='smplx', poser_type='vposer',
                        smpl_exp_dir=smpl_exp_dir, mano_exp_dir=None).to('cuda')

# vertices = c2c(bm.forward().v)[0]
# faces = c2c(bm.f)
# mesh = trimesh.base.Trimesh(vertices, faces).show()

bm2 = BodyModelWithPoser(bm_path=bm_path, batch_size=1, model_type='smplx', poser_type='vposer',
                         smpl_exp_dir=smpl_exp_dir).to('cuda')

# vertices = c2c(bm2.forward().v)[0]
# faces = c2c(bm2.f)
# mesh = trimesh.base.Trimesh(vertices, faces).show()

bm3 = BodyModelWithPoser(bm_path=bm_path, batch_size=1, model_type='smplx', poser_type='vposer',
                         smpl_exp_dir=smpl_exp_dir).to('cuda')

# bm.poZ_body.data[:]=bm.poZ_body.new(np.zeros(bm.poZ_body.shape)).detach()
# bm2.poZ_body.data[:]=bm2.poZ_body.new(np.zeros(bm.poZ_body.shape)).detach()

# bm3.pose_body.data[:]=bm.poser_body_pt.decode(bm.poZ_body, output_type='aa').view(bm.batch_size, -1)
# bm3.pose_body.data[:]=bm2.poser_body_pt.decode(bm2.poZ_body, output_type='aa').view(bm2.batch_size, -1)

num_interp = 18
# view_angles = [0, 90]
view_angles = np.linspace(0, 2, num=num_interp)
imw, imh = 400, 400
fraction = 15
#magification = 2

#poz_sel = np.asarray([6, 8, 9, 10, 12, 13, 14, 17, 18, 20, 22, 23, 26, 27, 29, 30, 31])
poz_sel = np.arange(32)
poz_mul = np.zeros(32)
#poz_mul[poz_sel] = poz_mul[poz_sel] * magification
poz_mul[poz_sel] = poz_mul[poz_sel] + 1
#poz_mul = torch.tensor(poz_mul, dtype=torch.float, requires_grad=False).to('cuda')

#nexp = 100
nrand = 11
nrand = 6
#scale = [8, 8, 32, 32, 96, 96]
# since d is lower for d=17 selection, the average max values will be higher. thus scale down to have approx the same max values for d=32 and d=17
scale = np.asarray([8, 8, 8, 32, 32, 32, 32, 96, 96, 96, 96])*0.74

nexp = poz_sel.shape[0]*nrand

t3mat = np.zeros((nexp, num_interp, 32))
kp2dmat = np.zeros((nexp, num_interp, 55, 2))
kp3dmat = np.zeros((nexp, num_interp, 55, 3))
poseaamat = np.zeros((nexp, num_interp, 63))

posemat = np.zeros((nexp, 32))
mat2d = np.zeros((nexp, 55))
viewpointmat = np.zeros(nexp)
viewpointmat[0]=0
#w = 6
#posemat[0,:] = np.random.randn(32) * magification * poz_mul



#fn = "pose_stims_p17_"
#for i in range(nrand):
#    fn = fn + "%d_" % scale[i]

#outdir = makepath(os.path.join(smpl_exp_dir, 'evaluations', fn))
out_dir = makepath(os.path.join(smpl_exp_dir, 'evaluations', 'pose_stims_p17_v4'))


for i in range(nrand):
    X = rand_ndim_onb(poz_sel.shape[0])
    #posemat[i*poz_sel.shape[0]:(i+1)*poz_sel.shape[0], poz_sel] = X.transpose() * (i+1) * 5
    posemat[i * poz_sel.shape[0]:(i + 1) * poz_sel.shape[0], poz_sel] = X.transpose() * scale[i]

'''        
cnt = 0
for i in range(nrand):
    rv = (np.random.rand(poz_sel.shape[0], 1)-1)*2
    rv = rv/np.linalg.norm(rv)
    N = sl.null_space(rv.transpose())
    B = np.concatenate((rv,N),axis = 1)
    for j in range(poz_sel.shape[0]):
        #posemat[cnt,poz_sel] = B[:,j]*np.random.choice(np.array([1, 3.5, 7]), 1)
        posemat[cnt, poz_sel] = B[:, j] * (i*4 + 1) * np.random.choice(np.array([-1, 1]), 1)
        cnt = cnt + 1
'''

'''
d_thresh = 15 #threshold on euc distance between samples
c_thresh = 5 #theshold on max parameter value

cnt=1
for i in range(1,nexp):
    posemat[i, :] = np.random.randn(32) * magification * poz_mul
    backind = np.arange(np.max((0, i - w)),i+1)
    d = sd.pdist(posemat[backind[:,np.newaxis],poz_sel[np.newaxis,:]])
    while not ((np.all(d>d_thresh)) and (np.all(np.abs(posemat[i, :]<c_thresh)))):
        posemat[i, :] = np.random.randn(32) * magification * poz_mul
        d = sd.pdist(posemat[backind[:,np.newaxis],poz_sel[np.newaxis,:]])
        cnt=cnt+1

print(cnt)
'''

cnt = 0
#np.random.seed(0)
for i in tqdm(range(0, nexp)):
    # bm.poZ_body.data[:]=bm.poZ_body.new(np.zeros(bm.poZ_body.shape)).detach()
    # bm2.poZ_body.data[:]=bm2.poZ_body.new(np.zeros(bm.poZ_body.shape)).detach()
    #bm.randomize_pose()
    bm2.randomize_pose()
    #t1 = bm.poZ_body.data
    t1 = bm.poZ_body.new(posemat[i,:]).detach()
    t2 = bm2.poZ_body.data

    # t1[0][i]=30
    # t2[0][i]=-30

    # print(t1)
    # print(t2)
    # vertices = c2c(bm3.forward().v)[0]
    # faces = c2c(bm3.f)
    # mesh = trimesh.base.Trimesh(vertices, faces).show()

    out_imgpath = os.path.join(out_dir, 'VAE_%d.png' % i)

    mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

    # images = np.zeros([len(view_angles), num_interp+1,1, imw, imh, 3])
    images = np.zeros([num_interp, imh, imw, 3])

    backind = np.arange(np.max((0, i - 2)), i )

    '''
    make viepoint not random, but render all poses at predifined viewpoints!!
    '''
    #randrot = np.random.choice(np.array([180, -90, -45 , 0, 45, 90]), 1)
    randrot = np.random.choice(np.array([-90, -45, 0, 45, 90]), 1)
    while np.any(viewpointmat[backind]==randrot):
        #randrot = np.random.choice(np.array([180, -90, -45, 0, 45, 90]), 1)
        randrot = np.random.choice(np.array([-90, -45, 0, 45, 90]), 1)
    viewpointmat[i] = randrot

    for cId in range(0, num_interp):
        # only go 10% in t2 direction
        alpha = 1 - (cId / (num_interp * fraction))
        # print(alpha)

        t3 = (alpha * t1 + (1 - alpha) * t2)
        t3mat[i, cId, :] = t3.detach().cpu().numpy()

        bm3.poZ_body.data[:] = t3

        bm3.pose_body.data[:] = bm3.poser_body_pt.decode(bm3.poZ_body, output_type='aa').view(bm3.batch_size, -1)

        poseaamat[i, cId, : ] = torch.squeeze(bm3.pose_body).detach().cpu().numpy();

        points = bm3.forward().Jtr

        kp3dmat[i, cId, : , :] = torch.squeeze(points).detach().cpu().numpy()

        rotation = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).expand(1, -1, -1)
        a = list(mv.scene.get_nodes(name='pc-camera'))[0]
        translation = torch.from_numpy(np.expand_dims(a.translation,axis=0).astype(np.float32)).to(device)
        translation[0,2] = 5
        focal_length = 1000
        camera_center = 0.5 * imh * torch.ones(1, 2, device=device, dtype=torch.float32)

        #projected_points = perspective_projection(points, rotation, translation,
        #                   focal_length, camera_center)

        #body_mesh = trimesh.Trimesh(vertices=c2c(bm3.forward().v)[0], faces=c2c(bm.f),
        #                            vertex_colors=np.tile(colors['blue'], (6890, 1)))
        body_mesh = trimesh.Trimesh(vertices=c2c(bm3.forward().v)[0], faces=c2c(bm.f),
                                    vertex_colors=np.tile([135, 250, 206], (c2c(bm3.forward().v).shape[1], 1)))

        # body_mesh = trimesh.Trimesh(vertices=c2c(bm3.forward(betas = bm3.betas.new(np.random.randn(bm3.batch_size, 10))).v)[0], faces=c2c(bm.f), vertex_colors=np.tile(colors['grey'], (6890, 1)))

        T = trimesh.transformations.translation_matrix([0, 0.2, 0])
        R = trimesh.transformations.rotation_matrix(np.radians(view_angles[cId] + randrot), (0, 1, 0))
        apply_mesh_tranfsormations_([body_mesh], T)
        apply_mesh_tranfsormations_([body_mesh], R)

        #rotate point as we do the mesh

        p2 = torch.cat((points, torch.ones((1, 55, 1)).to('cuda')), 2)
        p3 = torch.matmul(torch.from_numpy(T).to('cuda'), torch.t(p2.squeeze()).double())
        p3 = torch.matmul(torch.from_numpy(R).to('cuda'), p3)
        p4 = torch.unsqueeze(p3[:3, :].t(), 0)

        projected_points = perspective_projection(p4.float(), rotation, translation,
                                                  focal_length, camera_center)

        kp2dmat[i, cId, : , :] = torch.squeeze(projected_points).detach().cpu().numpy()

        mv.set_meshes([body_mesh], group_name='static')
        images[cId] = mv.render()

        #f = plt.figure()
        #plt.scatter(torch.squeeze(projected_points[0, :, 0]).detach().cpu().numpy(),
        #            torch.squeeze(projected_points[0, :, 1]).detach().cpu().numpy())
        #a = f.gca()
        #a.axis('equal')
        #f2 = plt.figure()
        #plt.imshow(np.mean(images[0], 2))


        # cv2.imwrite(os.path.join(out_dir, 'VAE0_%02d_%d.png' % (i, cId)), images[cId])
        '''
            for rId, angle in enumerate(view_angles):
                apply_mesh_tranfsormations_([body_mesh], trimesh.transformations.rotation_matrix(np.radians(angle), (0, 1, 0)))
                mv.set_meshes([body_mesh], group_name='static')
                images[rId, cId,0] = mv.render()
                apply_mesh_tranfsormations_([body_mesh], trimesh.transformations.rotation_matrix(np.radians(-angle), (0, 1, 0)))
            '''
    # imagearray2file(images, out_imgpath)

    # i1 = images[0, :]
    # i12 = np.squeeze(i1)
    images = images.astype(np.uint8)
    a1 = list(images)
    a12 = a1.copy()
    a12.reverse()
    a1.extend(a12)
    iio.mimsave(os.path.join(out_dir, 'VAE0_%02d.gif' % i), a1, duration=1 / 24)

    for cId, imgi in enumerate(a1):
        cv2.imwrite(os.path.join(out_dir, 'VAE_%04d.png' % cnt), imgi)
        cnt = cnt + 1

t = np.transpose(t3mat, (2, 1, 0))
t2 = np.reshape(t,(32, num_interp*nexp), order='F')
plt.imsave(os.path.join(out_dir, 'params_time_VAE2.png'),t2)

np.save(os.path.join(out_dir, 'params_time_VAE2.npy'),t2)
np.save(os.path.join(out_dir, 'params_viewpoint_VAE2.npy'),viewpointmat)

np.save(os.path.join(out_dir, 'kp2dmat.npy'),kp2dmat)
np.save(os.path.join(out_dir, 'kp3dmat.npy'),kp3dmat)
np.save(os.path.join(out_dir, 'poseaamat.npy'),poseaamat)
np.save(os.path.join(out_dir, 'exp_params.npy'),[num_interp, nrand])
np.save(os.path.join(out_dir, 'scale.npy'),scale)
np.save(os.path.join(out_dir, 'poz_sel.npy'),poz_sel)

'''
   i1 = images[1, :]
   i12 = np.squeeze(i1)
   a1 = list(i12[1:10, :])
   a12 = a1.copy()
   a12.reverse()
   a1.extend(a12)
   iio.mimsave(os.path.join(out_dir, 'VAE1_%d.gif' % i), a1, duration=0.05)
'''
# np.savez(out_imgpath.replace('.png', '.npz'), pose=pose_body)





