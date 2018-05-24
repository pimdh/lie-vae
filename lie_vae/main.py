import torch
import os.path
from pprint import pprint
from tensorboardX import SummaryWriter
import argparse
from math import pi

from lie_vae.datasets import SelectedDataset, ObjectsDataset, ThreeObjectsDataset, \
    HumanoidDataset, ColorHumanoidDataset, SingleChairDataset, SphereCubeDataset
from lie_vae.experiments import UnsupervisedExperiment, SemiSupervisedExperiment
from lie_vae.vae import ChairsVAE
from lie_vae.utils import random_split
from lie_vae.beta_schedule import get_beta_schedule

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    args = parse_args()
    pprint(vars(args))
    if args.name is not None:
        args.log_dir = 'runs/'+args.name
        args.save_dir = 'outputs/'+args.name

    log = SummaryWriter(args.log_dir)

    if args.dataset == 'objects':
        dataset = ObjectsDataset()
    elif args.dataset == 'objects3':
        dataset = ThreeObjectsDataset()
    elif args.dataset == 'chairs':
        dataset = SelectedDataset()
    elif args.dataset == 'humanoid':
        dataset = HumanoidDataset()
    elif args.dataset == 'chumanoid':
        dataset = ColorHumanoidDataset()
    elif args.dataset == 'single':
        dataset = SingleChairDataset()
    elif args.dataset == 'spherecube':
        dataset = SphereCubeDataset()
    else:
        raise RuntimeError('Wrong dataset')
    if not len(dataset):
        raise RuntimeError('Dataset empty')

    model = ChairsVAE(
        content_dims=args.content_dims,
        latent_mode=args.latent_mode,
        mean_mode=args.mean_mode,
        decoder_mode=args.decoder_mode,
        deconv_mode=args.deconv_mode,
        rep_copies=args.rep_copies,
        degrees=args.degrees,
        deconv_hidden=args.deconv_hidden,
        batch_norm=args.batch_norm,
        rgb=dataset.rgb,
        single_id=dataset.single_id
    ).to(device)

    if args.continue_epoch > 0:
        model.load_state_dict(torch.load(os.path.join(
            args.save_dir, 'model.pickle')))

    num_test = min(int(len(dataset) * 0.2), 5000)
    split = [len(dataset)-num_test, num_test]
    train_dataset, test_dataset = random_split(dataset, split)

    optimizer = torch.optim.Adam(model.parameters())

    if args.experiment == 'unsupervised':
        exp_cls = UnsupervisedExperiment
        exp_kwargs = {}
    elif args.experiment == 'semi':
        exp_cls = SemiSupervisedExperiment
        exp_kwargs = {
            'num_labelled': args.semi_labelled,
            'lambda_supervised': args.semi_lambda}
    else:
        raise RuntimeError('Wrong experiment')

    experiment = exp_cls(
        log=log,
        model=model,
        optimizer=optimizer,
        beta_schedule=get_beta_schedule(args.beta_schedule, args.beta),
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        elbo_samples=args.elbo_samples,
        report_freq=args.report_freq,
        clip_grads=args.clip_grads,
        selective_clip=args.selective_clip,
        continuity_lamb=args.continuity,
        continuity_scale=2*pi/args.continuity_iscale,
        **exp_kwargs
    )

    for epoch in range(args.continue_epoch, args.epochs):
        experiment.train(epoch)

        if args.save_dir:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(model.state_dict(), os.path.join(
                args.save_dir, 'model.pickle'))
    log.close()


def parse_args():
    parser = argparse.ArgumentParser('VAE experiment')
    parser.add_argument('--dataset', default='chairs',
                        help='Data set to use, [chairs, objects, objects3,'
                             'spherecube, chumanoid, single]')
    parser.add_argument('--decoder_mode', default='action',
                        help='[action, mlp]')
    parser.add_argument('--latent_mode', default='so3',
                        help='[so3, so3f, normal]')
    parser.add_argument('--mean_mode', default='alg', help='For SO(3). Choose [q, alg, s2s2, s2s1]')
    parser.add_argument('--experiment', default='unsupervised',
                        help='[unsupervised, semi]')
    parser.add_argument('--deconv_mode', default='deconv',
                        help='Deconv mode [deconv, upsample]')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='Whether to use Batch Norm in conv')
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--beta_schedule', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--report_freq', type=int, default=1250)
    parser.add_argument('--degrees', type=int, default=6)
    parser.add_argument('--deconv_hidden', type=int, default=200)
    parser.add_argument('--content_dims', type=int, default=10,
                        help='The dims of the content latent code')
    parser.add_argument('--rep_copies', type=int, default=10,
                        help='The dims of the virtual signal on the Sphere, '
                             'i.e. the number of copies of the representation.')
    parser.add_argument('--clip_grads', type=float, default=1E-5)
    parser.add_argument('--selective_clip', action='store_true')
    parser.add_argument('--elbo_samples', type=int, default=1)
    parser.add_argument('--log_dir')
    parser.add_argument('--save_dir')
    parser.add_argument('--name')
    parser.add_argument('--continue_epoch', type=int, default=0)
    parser.add_argument('--semi_labelled', type=int, default=100,
                        help='Number of labelled samples')
    parser.add_argument('--semi_lambda', type=float, default=1.,
                        help='Relative strength of supervised loss')
    parser.add_argument('--semi_batch', type=int, default=1,
                        help='Number of labelled samples in each batch')
    parser.add_argument('--continuity', type=float,
                        help='Strength of continuity loss')
    parser.add_argument('--continuity_iscale', type=float, default=200,
                        help='Inverse algebra distance with which continuity'
                             'is measured. Distance is 2pi/iscale.')
    return parser.parse_args()


if __name__ == '__main__':
    main()
