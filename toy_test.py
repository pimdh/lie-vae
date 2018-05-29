import numpy as np
from lie_vae.datasets import ToyDataset
from lie_vae.vae import ChairsVAE
from lie_vae.lie_tools import quaternions_to_group_matrix, group_matrix_to_eazyz, block_wigner_matrix_multiply
from torch.utils.data import DataLoader

dataset = ToyDataset()
loader = DataLoader(dataset, batch_size=64, shuffle=True)
matrix_dims, rep_copies = dataset[0][1].shape
degrees = int(np.round(np.sqrt(matrix_dims)))-1
model = ChairsVAE(
    latent_mode='so3',
    mean_mode='s2s2',
    decoder_mode='action',
    encode_mode='toy',
    deconv_mode='toy',
    rep_copies=rep_copies,
    degrees=degrees,
    rgb=False,
    single_id=True,
    deterministic=True,
    item_rep=dataset[0][1],
)

for i, (q, org, transformed) in enumerate(loader):
    mat = quaternions_to_group_matrix(q)

    # Test decodability: Supply correct transformation matrix to decoder
    # Ground truth original is already provided to decoder.
    reconstruction = model.decode(mat[None])[0]
    np.testing.assert_allclose(reconstruction, transformed, atol=1E-4)

    # Additional test: reconstruct original from transformed by applying inverse transform
    angles_inv = group_matrix_to_eazyz(mat.transpose(-2, -1))
    org_reconstructed = block_wigner_matrix_multiply(angles_inv, transformed, 3)
    np.testing.assert_allclose(org_reconstructed, org, atol=1E-4)

    if i == 10:
        break

print("All tests passed")
