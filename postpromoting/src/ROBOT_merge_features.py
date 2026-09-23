#%%
import torch
import numpy as np
import os

#%%
base_path = '../data/ROBOT/features'

all_frames_data = []
for i in range(2570):
    file_name = f'{base_path}/feature_{i}.pt'
    print(file_name)
    data = torch.load(file_name, weights_only=False)
    np_data = data['downsampled_features'].to(torch.float).numpy().reshape(1, -1)
    all_frames_data.append(np_data[:, :3000])

print(len(all_frames_data))
all_frames_data = np.concatenate(all_frames_data)
print(all_frames_data.shape)

#%% baocuntezheng
save_path = os.path.join(base_path, 'robot_task')
if not os.path.exists(save_path):
    os.makedirs(save_path)
np.save(os.path.join(save_path, 'robot_task_001.npy'), all_frames_data)
