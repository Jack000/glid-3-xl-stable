import torch
import sys

model_path = sys.argv[1]
diffusion_path = sys.argv[2]

state = torch.load(model_path)
diffusion = torch.load(diffusion_path)

diffusion_prefix = 'model.diffusion_model.'

for key in diffusion.keys():
    state['state_dict'][diffusion_prefix + key] = diffusion[key]

torch.save(state, 'model-merged.pt')
