def off_to_voxel(off_path, grid_size=16):
    """
    Convert an .off file to a voxel grid using Open3D.
    """
    try:
        mesh = o3d.io.read_triangle_mesh(off_path)
        if mesh.is_empty():
            print(f"Mesh is empty for file: {off_path}")
            return None

        # Normalize mesh to fit within [0,1]
        bounds = mesh.get_max_bound() - mesh.get_min_bound()
        scale_factor = 1.0 / max(bounds)
        mesh.scale(scale_factor, center=mesh.get_center())
        translation = -mesh.get_min_bound() * scale_factor
        mesh.translate(translation)

        # Create voxel grid
        voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size=1.0 / grid_size)

        # Convert voxel grid to numpy
        voxel_numpy = np.zeros((grid_size, grid_size, grid_size), dtype=np.uint8)
        for voxel in voxel_grid.get_voxels():
            x, y, z = voxel.grid_index
            # Ensure indices are in bounds
            if 0 <= x < grid_size and 0 <= y < grid_size and 0 <= z < grid_size:
                voxel_numpy[x, y, z] = 1
        return voxel_numpy

    except Exception as e:
        print(f"Error processing {off_path}: {e}")
        return None


def load_dataset(data_dir, grid_size=16):
    """
    Loads the ModelNet dataset from .off files, converts to voxel grids.
    Returns np arrays: voxel_data, labels
    """
    voxel_data = []
    labels = []
    classes = ['chair']  
    for label, cls in enumerate(classes):
        class_dir = os.path.join(data_dir, cls)
        if os.path.isdir(class_dir):
            for split in ['train', 'test']:
                split_dir = os.path.join(class_dir, split)
                if not os.path.isdir(split_dir):
                    print(f"Split directory {split_dir} missing. Skipping...")
                    continue
                print(f"Processing class: {cls}, split: {split}")
                for file in os.listdir(split_dir):
                    if file.endswith(".off"):
                        file_path = os.path.join(split_dir, file)
                        voxel = off_to_voxel(file_path, grid_size)
                        if voxel is not None:
                            voxel_data.append(voxel)
                            labels.append(label)
    return np.array(voxel_data), np.array(labels)
