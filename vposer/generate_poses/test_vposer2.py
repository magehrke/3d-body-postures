from human_body_prior.tools.model_loader import load_vposer
import os

expr_dir = os.path.join('.\\vposer_v1_0') # in this directory the trained model along with the model code exist
num_sample_batch = 10 # number of body poses in each batch
 
vposer_pt, ps = load_vposer(expr_dir, vp_model='snapshot')
sampled_pose_body = vposer_pt.sample_poses(num_poses=num_sample_batch) # will a generate Nx1x21x3 tensor of body poses


from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tutorials.sample_body_pose import sample_vposer

bm_path = 'D:\smpl\models\smplx\SMPLX_MALE.npz' # obtain from https://smpl-x.is.tue.mpg.de/downloads
bm = BodyModel(bm_path, 'smplx')

# expr_dir: directory for the trained model along with the model code. obtain from https://smpl-x.is.tue.mpg.de/downloads
#expr_dir = 'TRAINED_MODEL_DIRECTORY'
sample_vposer(expr_dir, bm, 5, vp_model='snapshot')
