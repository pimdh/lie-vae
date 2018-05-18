import torch
import numpy as np
from torch.utils.data import DataLoader
import os.path
from pprint import pprint
from tensorboardX import SummaryWriter
import argparse

from lie_vae.datasets import SelectedDataset, ObjectsDataset, ThreeObjectsDataset, \
    HumanoidDataset, ColorHumanoidDataset, SingleChairDataset, SphereCubeDataset
from lie_vae.vae import ChairsVAE
from lie_vae.utils import random_split, ConstantSchedule, LinearSchedule

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test(loader, model, elbo_samples=1):
    model.eval()
    losses = []
    for it, (item_label, rot_label, img_label) in enumerate(loader):
        img_label = img_label.to(device)
        recon, kl, kls = model.elbo(img_label, n=elbo_samples)
        if len(kls) == 2:
            kl0, kl1 = kls
        else:
            kl0, kl1 = kls[0], torch.tensor(0.)
        losses.append((recon.mean().item(), kl.mean().item(),
                       kl1.mean().item(), kl0.mean().item()))
    return np.mean(losses, 0)


def train(epoch, train_loader, test_loader, model, optimizer, log,
          beta_schedule, report_freq=1250, clip_grads=None,
          selective_clip=False, elbo_samples=1):
    losses = []
    for it, (item_label, rot_label, img_label) in enumerate(train_loader):
        model.train()
        img_label = img_label.to(device)
        recon, kl, kls = model.elbo(img_label, n=elbo_samples)
        if len(kls) == 2:
            kl0, kl1 = kls
        else:
            kl0, kl1 = kls[0], torch.tensor(0.)

        global_it = epoch * len(train_loader) + it + 1
        beta = beta_schedule(global_it)

        loss = (recon + beta * kl).mean()

        if torch.isnan(kl).sum():
            raise RuntimeError("NaN KL")

        optimizer.zero_grad()
        loss.backward()
        if clip_grads:
            if selective_clip:
                params = list(model.encoder.parameters()) \
                         + list(model.rep_group.parameters())
            else:
                params = model.parameters()
            torch.nn.utils.clip_grad_norm_(params, clip_grads)
        optimizer.step()

        losses.append((recon.mean().item(), kl.mean().item(),
                       kl1.mean().item(), kl0.mean().item()))

        if (it + 1) % report_freq == 0 or it + 1 == len(train_loader):
            train_recon, train_kl, train_kl0, train_kl1 = np.mean(losses[-report_freq:], 0)
            log.add_scalar('train_loss', train_recon + beta * train_kl, global_it)
            log.add_scalar('train_recon', train_recon, global_it)
            log.add_scalar('train_kl', train_kl, global_it)
            log.add_scalars('train_kls', {'kl0': train_kl0, 'kl1': train_kl1}, global_it)

            test_recon, test_kl, test_kl0, test_kl1 = test(test_loader, model)
            log.add_scalar('test_loss', test_recon + beta * test_kl, global_it)
            log.add_scalar('test_recon', test_recon, global_it)
            log.add_scalar('test_kl', test_kl, global_it)
            log.add_scalars('test_kls', {'kl0': test_kl0, 'kl1': test_kl1}, global_it)

            log.add_scalar('beta', beta, global_it)
            print('Epoch {} it {} train recon {:.4f} kl {:.4f} test recon {:.4f} kl {:.4f}'
                  .format(epoch, it+1, train_recon, train_kl, test_recon, test_kl))


def main():
    args = parse_args()
    pprint(vars(args))
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

    if args.beta_schedule is None:
        beta_schedule = ConstantSchedule(args.beta)
    elif args.beta_schedule == 'a':
        beta_schedule = LinearSchedule(0.001, 1, 60000, 200000)
    elif args.beta_schedule == 'b':
        beta_schedule = LinearSchedule(0.001, 0.1, 60000, 200000)
    elif args.beta_schedule == 'c':
        beta_schedule = LinearSchedule(0.001, 0.01, 60000, 200000)
    elif args.beta_schedule == 'd':
        beta_schedule = LinearSchedule(0.001, 10, 60000, 200000)
    elif args.beta_schedule == 'e':
        beta_schedule = LinearSchedule(0.001, 0.1, 60000, 120000)
    elif args.beta_schedule == 'f':
        beta_schedule = LinearSchedule(0.001, 1, 60000, 120000)
    elif args.beta_schedule == 'g':
        beta_schedule = LinearSchedule(0.001, 0.3, 60000, 120000)
    elif args.beta_schedule == 'h':
        beta_schedule = LinearSchedule(0.001, 0.3, 30000, 60000)
    elif args.beta_schedule == 'i':
        beta_schedule = LinearSchedule(0.001, 1, 30000, 60000)
    elif args.beta_schedule == 'j':
        beta_schedule = LinearSchedule(0.001, 3, 30000, 60000)
    elif args.beta_schedule == 'k':
        beta_schedule = LinearSchedule(0.001, 10, 30000, 60000)
    elif args.beta_schedule == 'l':
        beta_schedule = LinearSchedule(0.001, 30, 30000, 60000)
    elif args.beta_schedule == 'm':
        beta_schedule = LinearSchedule(0.001, 3, 60000, 120000)
    elif args.beta_schedule == 'n':
        beta_schedule = LinearSchedule(0.001, 10, 60000, 120000)
    elif args.beta_schedule == 'o':
        beta_schedule = LinearSchedule(0.001, 30, 60000, 120000)
    else:
        raise RuntimeError('Wrong beta schedule')

    num_test = min(int(len(dataset) * 0.2), 5000)
    split = [len(dataset)-num_test, num_test]
    train_dataset, test_dataset = random_split(dataset, split)
    train_loader = DataLoader(train_dataset, batch_size=64,
                              shuffle=True, num_workers=10)
    test_loader = DataLoader(test_dataset, batch_size=64,
                             shuffle=True, num_workers=10)

    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(args.continue_epoch, args.epochs):
        train(epoch, train_loader, test_loader, model, optimizer, log,
              report_freq=args.report_freq, clip_grads=args.clip_grads,
              beta_schedule=beta_schedule,
              elbo_samples=args.elbo_samples)
        if args.save_dir:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
            torch.save(model.state_dict(), os.path.join(
                args.save_dir, 'model.pickle'))
    log.close()


def parse_args():
    parser = argparse.ArgumentParser('Supervised experiment')
    parser.add_argument('--dataset', default='chairs',
                        help='Data set to use, [chairs, objects, objects3]')
    parser.add_argument('--decoder_mode', required=True,
                        help='[action, mlp]')
    parser.add_argument('--latent_mode', required=True,
                        help='[so3, normal]')
    parser.add_argument('--deconv_mode', default='deconv',
                        help='Deconv mode [deconv, upsample]')
    parser.add_argument('--batch_norm', type=int, default=1,
                        help='Whether to use Batch Norm in conv')
    parser.add_argument('--beta', type=float, default=1.)
    parser.add_argument('--beta_schedule', type=str)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--report_freq', type=int, default=1250)
    parser.add_argument('--degrees', type=int, default=6)
    parser.add_argument('--deconv_hidden', type=int, default=50)
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
    parser.add_argument('--continue_epoch', type=int, default=0)
    return parser.parse_args()


if __name__ == '__main__':
    main()
