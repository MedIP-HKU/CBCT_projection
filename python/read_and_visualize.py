import os, glob
import matplotlib.pyplot as plt
import numpy as np


def read_data(path):
    data = np.load(path)
    primary_rescaler = data['primary_rescaler']
    # primary_mc_rescaler = data['primary_mc_rescaler']
    scatter_rescaler = data['scatter_rescaler']
    primary_only = data['primary'].astype(np.float32) * primary_rescaler[0] + primary_rescaler[1]
    scatter_only = data['scatter'].astype(np.float32) * scatter_rescaler[0] + scatter_rescaler[1]
    return primary_only, scatter_only


data_path = '../../data/truebeam/hn/HN-CHUM-006/'
plot_path = './plots'
nu = 512
nv = 384

if not os.path.isdir(plot_path):
    os.makedirs(plot_path)
proj_list = sorted(glob.glob(os.path.join(data_path, '*.npz')))
proj_size = np.zeros([len(proj_list), nv, nu], dtype=np.float32)
for i in range(1):
    primary_only, scatter_only = read_data(proj_list[i])
    sf = -np.log(primary_only + 1e-6)
    proj = -np.log(primary_only + scatter_only + 1e-6)
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(sf, cmap='gray', vmin=0, vmax=5.0)
    plt.title(f"Projection {i}")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(proj, cmap='gray', vmin=0, vmax=5.0)
    plt.title(f"Projection {i} with scatter")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.plot(sf[nv // 2, :], 'k--', label='primary')
    plt.plot(proj[nv // 2, :], 'r-.', label='primary + scatter')
    plt.legend()
    plt.tight_layout(pad=1.1)
    plt.savefig(os.path.join(plot_path, f'projection_{i}.png'))
