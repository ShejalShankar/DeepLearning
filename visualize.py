import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def voxel2points(voxel, threshold=0.5):

    # voxel is expected to be shape (D, D, D)
    D, M, N = voxel.shape
    # Binarize the voxel data
    filled_positions = np.where(voxel > threshold)

    # Convert to integer mask for easier neighbor checking
    bin_voxel = np.zeros_like(voxel, dtype=np.uint8)
    bin_voxel[filled_positions] = 1

    X, Y, Z = [], [], []
    for x, y, z in zip(*filled_positions):
        # Extract a 3x3x3 neighborhood around (x,y,z)
        x_min, x_max = max(x - 1, 0), min(x + 2, D)
        y_min, y_max = max(y - 1, 0), min(y + 2, M)
        z_min, z_max = max(z - 1, 0), min(z + 2, N)
        neighborhood = bin_voxel[x_min:x_max, y_min:y_max, z_min:z_max]

        # If at least one neighbor is empty, this is a surface voxel
        if np.sum(neighborhood) < neighborhood.size:
            X.append(x)
            Y.append(y)
            Z.append(z)
    return np.array(X), np.array(Y), np.array(Z)

def visualize_voxel(voxel, threshold=0.5, colormap='viridis', show_plot=True, save_path=None):
    X, Y, Z = voxel2points(voxel, threshold=threshold)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(X, Y, Z, c=Z, cmap=colormap, s=25, marker='.')

   
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    ax.set_box_aspect([1,1,1])  
    plt.title("3D Voxel Visualization")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)
