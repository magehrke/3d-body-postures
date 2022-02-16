import torch
import numpy as np
from human_body_prior.mesh.mesh_viewer import MeshViewer

def get_camera_params(mv: MeshViewer, device: torch.device):
    rotation = torch.eye(3, device=device, dtype=torch.float32).unsqueeze(0).expand(1, -1, -1)
    a = list(mv.scene.get_nodes(name='pc-camera'))[0]
    translation = torch.from_numpy(np.expand_dims(a.translation, axis=0).astype(np.float32)).to(device)
    translation[0, 2] = 5
    focal_length = 1000
    return rotation, translation, focal_length
