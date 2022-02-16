import trimesh
import os
from human_body_prior.mesh.mesh_viewer import MeshViewer
from human_body_prior.tools.omni_tools import id_generator, makepath
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_
import imageio as iio



device = torch.device('cuda')

#from vposer stims script:
#body_mesh_std = [ 0.12915216,  0.43018637,  0.13732606]
body_mesh_std = 0.2
#body_mesh_mean = [-0.13075693,  0.08731171,  0.08791853]
body_mesh_mean = [-0.13075693,  0,  0]
body_mesh_mean = [0,  0,  0]
num_interp = 18
# view_angles = [0, 90]
view_angles = np.linspace(0, 2, num=num_interp)
imw, imh = 400, 400

num_vp = 3 #-45 0 45
vp = [-45, 0, 45]

smpl_exp_dir = os.path.join('D:\\smpl\\vposer_v1_0')

#files = ['D:\\stimsets\\ModelNet10\\chair\\test\\chair_0919.off','D:\\stimsets\\ModelNet10\\chair\\test\\chair_0931.off', 'D:\\stimsets\\ModelNet10\\chair\\test\\chair_0964.off', 'D:\\stimsets\\ModelNet10\\chair\\test\\chair_0985.off', 'D:\\stimsets\\ModelNet10\\chair\\train\\chair_0090.off', 'D:\\stimsets\\ModelNet10\\chair\\train\\chair_0060.off', 'D:\\stimsets\\ModelNet10\\chair\\train\\chair_0075.off']
#out_dir = makepath(os.path.join(smpl_exp_dir, 'evaluations', 'localizer_stims_chairs'))
#files=['D:\\stimsets\\houses\\house1.stl','D:\\stimsets\\houses\\house2.stl','D:\\stimsets\\houses\\house3.stl','D:\\stimsets\\houses\\247_House15_obj.stl','D:\\stimsets\\houses\\farmhouse.stl','D:\\stimsets\\houses\\cyprus_house.stl','D:\\stimsets\\houses\\cottage.stl','D:\\stimsets\\houses\\cartoonhouse.stl','D:\\stimsets\\houses\\building.stl']
#out_dir = makepath(os.path.join(smpl_exp_dir, 'evaluations', 'localizer_stims_houses'))
files=['D:\\stimsets\\faces\\00001_20061015_00418_neutral_face05.ply','D:\\stimsets\\faces\\00002_20061015_00448_neutral_face05.ply','D:\\stimsets\\faces\\00006_20080430_04384_neutral_face05.ply','D:\\stimsets\\faces\\00014_20080430_04338_neutral_face05.ply','D:\\stimsets\\faces\\00017_20061201_00812_neutral_face05.ply']
out_dir = makepath(os.path.join(smpl_exp_dir, 'evaluations', 'localizer_stims_faces'))
#files=['D:\\stimsets\\tools\\clock.stl','D:\\stimsets\\tools\\drill2.stl','D:\\stimsets\\tools\\extinguisher.stl','D:\\stimsets\\tools\\knife2.stl','D:\\stimsets\\tools\\wrench2.stl']
#out_dir = makepath(os.path.join(smpl_exp_dir, 'evaluations', 'localizer_stims_tools'))
#files=['D:\\stimsets\\ModelNet40\\vase\\train\\vase_0027.off','D:\\stimsets\\ModelNet40\\vase\\train\\vase_0042.off','D:\\stimsets\\ModelNet40\\vase\\train\\vase_0053.off','D:\\stimsets\\ModelNet40\\vase\\train\\vase_0090.off','D:\\stimsets\\ModelNet40\\vase\\train\\vase_0108.off','D:\\stimsets\\ModelNet40\\vase\\train\\vase_0139.off']
#out_dir = makepath(os.path.join(smpl_exp_dir, 'evaluations', 'localizer_stims_vases'))
#files=['D:\\stimsets\\ModelNet40\\lamp\\train\\lamp_0027.off','D:\\stimsets\\ModelNet40\\lamp\\train\\lamp_0031.off','D:\\stimsets\\ModelNet40\\lamp\\train\\lamp_0061.off','D:\\stimsets\\ModelNet40\\lamp\\train\\lamp_0076.off','D:\\stimsets\\ModelNet40\\lamp\\train\\lamp_0120.off']
#out_dir = makepath(os.path.join(smpl_exp_dir, 'evaluations', 'localizer_stims_lamps'))

nexp = len(files)

viewpointmat = np.zeros((nexp, num_vp))

#outdir = makepath(os.path.join(smpl_exp_dir, 'evaluations', fn))

cnt = 0
mv = MeshViewer(width=imw, height=imh, use_offscreen=True)
for i in tqdm(range(0, nexp)):
    model = trimesh.load(files[i])

    #model.vertices = model.vertices - np.mean(model.vertices, 0)
    #model.vertices = model.vertices * (np.asarray(body_mesh_std) / np.max(np.std(model.vertices, 0)))
    #model.vertices = model.vertices - (np.mean(model.vertices, 0) - np.asarray(body_mesh_mean))

    print(np.mean(model.vertices, 0))
    print(np.std(model.vertices, 0))

    for vId in range(3):

        images = np.zeros([num_interp, imh, imw, 3])
        viewpointmat[i, vId] = vp[vId]
        for cId in range(0, num_interp):
            #mesh = trimesh.Trimesh(vertices=model.vertices, faces=model.faces,
            #                       vertex_colors=np.tile([135, 250, 206], (model.vertices.shape[1], 1)))
            mesh = trimesh.Trimesh(vertices=model.vertices, faces=model.faces)
            mesh.visual.vertex_colors = [135, 250, 206, 255]

            #print(np.mean(mesh.vertices, 0))

            #R = trimesh.transformations.rotation_matrix(np.radians(-90), (1, 0, 0))
            #apply_mesh_tranfsormations_([mesh], R)
            #T = trimesh.transformations.translation_matrix([0, -0.5, 0])

            mesh.vertices = mesh.vertices - np.mean(mesh.vertices, 0)
            mesh.vertices = mesh.vertices * (np.asarray(body_mesh_std) / np.max(np.std(mesh.vertices, 0)))
            mesh.vertices = mesh.vertices - (np.mean(mesh.vertices, 0) - np.asarray(body_mesh_mean))

            #print(np.mean(mesh.vertices,0))

            R = trimesh.transformations.rotation_matrix(np.radians(view_angles[cId] + vp[vId]), (0, 1, 0))
            #apply_mesh_tranfsormations_([mesh], T)
            apply_mesh_tranfsormations_([mesh], R)

            mv.set_meshes([mesh], group_name='static')
            images[cId] = mv.render()

        images = images.astype(np.uint8)
        a1 = list(images)
        a12 = a1.copy()
        a12.reverse()
        a1.extend(a12)
        iio.mimsave(os.path.join(out_dir, 'VAE_%02d_%02d.gif' % (i , vId)), a1, duration=1 / 24)
        for cId, imgi in enumerate(a1):
            cv2.imwrite(os.path.join(out_dir, 'VAE_%04d.png' % cnt), imgi)
            cnt = cnt + 1
