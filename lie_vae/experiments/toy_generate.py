"""Generate toy dataset."""
import argparse
from lie_vae.experiments.datasets import ToyDataset

parser = argparse.ArgumentParser('Toy data generator')
parser.add_argument('num', type=int)
parser.add_argument('degrees', type=int)
parser.add_argument('rep_copies', type=int)
parser.add_argument('--path', default='data/toy.pickle')
args = parser.parse_args()

ToyDataset.generate(n=args.num, degrees=args.degrees, rep_copies=args.rep_copies).save(args.path)

print("Dataset generated")