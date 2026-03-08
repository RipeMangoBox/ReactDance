import os
import numpy as np
import torch
from tqdm import tqdm

# Base input/output directories
base_input_dir = 'data_lazy/motion/rotmat'
base_output_dir = 'data_lazy/motion/rot6d'
os.makedirs(base_output_dir, exist_ok=True)


def rotmat_to_6d(rotmat):
    """
    Convert 3x3 rotation matrices to 6D representation by taking the first two columns.

    Args:
        rotmat: np.ndarray or torch.Tensor with shape (..., 3, 3).

    Returns:
        rot6d: same leading shape as `rotmat`, with the last two dims collapsed to 6,
               i.e. shape (..., 6).
    """
    return rotmat[..., :2].reshape(rotmat.shape[:-2] + (6,))


# Process all sub-directories (splits)
subdirs = ['train', 'test']
for subdir in subdirs:
    input_dir = os.path.join(base_input_dir, subdir)
    output_dir = os.path.join(base_output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all .npy motion files in the current split
    for filename in tqdm(os.listdir(input_dir), desc=f"{subdir} processing"):
        if filename.endswith('.npy'):
            # Load raw data: [trans(3), rotmat(55 * 9)]
            file_path = os.path.join(input_dir, filename)
            data = np.load(file_path)
            n = data.shape[0]  # number of frames
            data_tensor = torch.tensor(data, dtype=torch.float32)

            # Split translation and rotation matrices
            trans = data_tensor[:, :3]  # shape: (n, 3), kept here in case we want to concatenate later
            rotmat = data_tensor[:, 3:].reshape(n, 55, 3, 3)  # shape: (n, 55, 3, 3)

            # Convert rotmat(55 × 3 × 3) -> rot6d(55 × 6)
            rot6d = rotmat_to_6d(rotmat)  # shape: (n, 55, 6)
            # If you prefer flattened per-frame features, uncomment the following line:
            # rot6d = rot6d.reshape(n, -1)  # shape: (n, 55 * 6)

            # Choose what to save:
            # - Only 6D rotations:
            output_data = rot6d
            # - Or concatenate translation + flattened 6D rotations:
            # output_data = torch.cat([trans, rot6d.reshape(n, -1)], dim=1).numpy()

            # Save converted sequence
            output_path = os.path.join(output_dir, filename)
            np.save(output_path, output_data)
            # print(f"Converted and saved: {filename} -> {output_path}")