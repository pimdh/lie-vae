from lie_vae.datasets import ToyDataset

ToyDataset.generate(n=1000000, degrees=3, rep_copies=1).save()
