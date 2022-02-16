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

import torch
import torch.nn as nn

from smplx.lbs import lbs

from human_body_prior.tools.omni_tools import apply_mesh_tranfsormations_



#smpl_exp_dir = "D:\\smpl\\vposer_v1_0"
smpl_exp_dir = os.path.join('.\\vposer_v1_0')

print (smpl_exp_dir)
print (os.path.exists(smpl_exp_dir))

# directory for the trained model along with the model code. obtain from https://smpl-x.is.tue.mpg.de/downloads
#D:\smpl\human_body_prior-master

bm_path = 'D:\smpl\models\smplx\SMPLX_MALE.npz' # obtain from https://smpl-x.is.tue.mpg.de/downloads
#bm_path = 'D:\smpl\models\smpl\SMPL.pkl'


bm = BodyModelWithPoser(bm_path=bm_path, batch_size=1, model_type='smplx', poser_type='vposer', smpl_exp_dir=smpl_exp_dir).to('cuda')
bm.randomize_pose()
#vertices = c2c(bm.forward().v)[0]
#faces = c2c(bm.f)
#mesh = trimesh.base.Trimesh(vertices, faces).show()
	
bm2 = BodyModelWithPoser(bm_path=bm_path, batch_size=1, model_type='smplx', poser_type='vposer', smpl_exp_dir=smpl_exp_dir).to('cuda')
bm2.randomize_pose()
#vertices = c2c(bm2.forward().v)[0]
#faces = c2c(bm2.f)
#mesh = trimesh.base.Trimesh(vertices, faces).show()

bm3 = BodyModelWithPoser(bm_path=bm_path, batch_size=1, model_type='smplx', poser_type='vposer', smpl_exp_dir=smpl_exp_dir).to('cuda')

bm.poZ_body.data[:]=bm.poZ_body.new(np.zeros(bm.poZ_body.shape)).detach()
bm2.poZ_body.data[:]=bm2.poZ_body.new(np.zeros(bm.poZ_body.shape)).detach()

#bm3.pose_body.data[:]=bm.poser_body_pt.decode(bm.poZ_body, output_type='aa').view(bm.batch_size, -1)
#bm3.pose_body.data[:]=bm2.poser_body_pt.decode(bm2.poZ_body, output_type='aa').view(bm2.batch_size, -1)

for i in range(10):
	#bm.poZ_body.data[:]=bm.poZ_body.new(np.zeros(bm.poZ_body.shape)).detach()
	#bm2.poZ_body.data[:]=bm2.poZ_body.new(np.zeros(bm.poZ_body.shape)).detach()
	#t1 = bm.poZ_body.data
	#t2 = bm2.poZ_body.data

	#t1[0][i]=30
	#t2[0][i]=-30

	#print(t1)
	#print(t2)
	#vertices = c2c(bm3.forward().v)[0]
	#faces = c2c(bm3.f)
	#mesh = trimesh.base.Trimesh(vertices, faces).show()
	out_dir = makepath(os.path.join(smpl_exp_dir, 'evaluations', 'pose_interp_shape'))
	out_imgpath = os.path.join(out_dir, 'VAE_%d.png' % i)
		
	view_angles = [0, 90]
	imw, imh = 400,400
	mv = MeshViewer(width=imw, height=imh, use_offscreen=True)

	num_interp = 10

	images = np.zeros([len(view_angles), num_interp+1,1, imw, imh, 3])

	for cId in range(0, num_interp + 1):
			
			#alpha = 1 - (cId / num_interp)
			alpha = (cId - num_interp/2)*2
			print(alpha)
			
			#t3 = (alpha*t1 + (1-alpha)*t2) / 2
	 
			bm3.poZ_body.data[:]=bm3.poZ_body.new(np.zeros(bm3.poZ_body.shape)).detach()
	 
			bm3.pose_body.data[:]=bm3.poser_body_pt.decode(bm3.poZ_body, output_type='aa').view(bm3.batch_size, -1)
			betas = bm3.betas.new(np.zeros([bm3.batch_size, 10]))
			betas[0][i]=alpha
			print(betas)
			#body_mesh = trimesh.Trimesh(vertices=c2c(bm3.forward().v)[0], faces=c2c(bm.f), vertex_colors=np.tile(colors['grey'], (6890, 1)))
			body_mesh = trimesh.Trimesh(vertices=c2c(bm3.forward(betas = betas).v)[0], faces=c2c(bm.f), vertex_colors=np.tile(colors['grey'], (6890, 1)))
			
			for rId, angle in enumerate(view_angles):
				apply_mesh_tranfsormations_([body_mesh], trimesh.transformations.rotation_matrix(np.radians(angle), (0, 1, 0)))
				mv.set_meshes([body_mesh], group_name='static')
				images[rId, cId,0] = mv.render()
				apply_mesh_tranfsormations_([body_mesh], trimesh.transformations.rotation_matrix(np.radians(-angle), (0, 1, 0)))

	imagearray2file(images, out_imgpath)

#np.savez(out_imgpath.replace('.png', '.npz'), pose=pose_body)


