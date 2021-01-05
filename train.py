import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, kl, kl_divergence, Independent
from itertools import chain
import os

from config import cifar_config
from model import *
from dataset.cifar import get_cifar

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0")


def is_parallel(model):
    return isinstance(model, torch.nn.parallel.DataParallel)

def gaussian_analytical_kl(mu, logvar):
    # KL(qz_x, N(0, 1))
    return 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), -1).mean()

def loss_vae_normal(x, encoder, decoder):
    batch_size = x.size(0)
    encoder_output = encoder(x)
    d = encoder_output.shape[1] // 2
    pz_loc = F.sigmoid(torch.zeros(batch_size, d).to(device))
    pz_scale = torch.ones(batch_size, d).to(device)
    pz = Independent(Normal(loc=pz_loc, scale=pz_scale),
                       reinterpreted_batch_ndims=1)
    qz_x_loc = encoder_output[:, :d]
    qz_x_log_scale = encoder_output[:, d:]
    qz_x = Independent(Normal(loc=qz_x_loc, scale=qz_x_log_scale ** 2),
                       reinterpreted_batch_ndims=1)
    z = qz_x.rsample()
    decoder_output = decoder(z)
    optimal_sigma_observed = ((x - decoder_output) ** 2).mean([0,1,2,3], keepdim=True).sqrt()
    px_z = Independent(Normal(loc=decoder_output, scale=optimal_sigma_observed),
                       reinterpreted_batch_ndims=3)
    elbo = (px_z.log_prob(x) - kl_divergence(qz_x, pz)).mean()
    return -elbo, decoder_output


def train_model(loss, batch_size, num_epochs, learning_rate):
    enc = Encoder(**cifar_config().encoder_arc).to(device)
    dec = Decoder(**cifar_config().decoder_arc).to(device)
    model = [enc, dec]
    gd = optim.Adam(
        chain(*[x.parameters() for x in model
                if (isinstance(x, nn.Module) or isinstance(x, nn.Parameter))]),
        learning_rate,
        weight_decay=1e-5
    )
    train_loader, test_loader = get_cifar(
        batch_size=batch_size, num_workers=32)
    train_losses = []
    test_results = []
    for cnt in range(num_epochs):
        for i, (batch, _) in enumerate(train_loader):
            total = len(train_loader)
            gd.zero_grad()
            batch = batch.to(device)
            loss_value, _ = loss(batch, enc, dec)
            loss_value.backward()
            train_losses.append(loss_value.item())
            if (i + 1) % 10 == 0:
                print('\rTrain loss:', train_losses[-1],
                      'Batch', i + 1, 'of', total, ' ' * 10, end='', flush=True)
            gd.step()
        test_elbo = 0.
        test_mse = 0.
        with torch.autograd.no_grad():
            for i, (batch, _) in enumerate(test_loader):
                batch = batch.to(device)
                batch_loss, recon = loss(batch, enc, dec)
                test_mse += (torch.nn.MSELoss()(recon, batch) - test_mse) / (i + 1)
                test_elbo += (batch_loss - test_elbo) / (i + 1)
        print('\nTest elbo after at epoch {}: {}'.format(cnt, test_elbo))
        print('Test mse after at epoch {}: {}'.format(cnt, test_mse))
        test_results.append((test_elbo, test_mse))

    enc.cpu()
    dec.cpu()
    torch.save(enc.state_dict(), "./ckpt/enc.pt")
    torch.save(dec.state_dict(), "./ckpt/dec.pt")
    with open('./log/log.txt', 'w') as f:
        for item in test_results:
            f.write("%s\n" % float(item))


if __name__ == "__main__":
    train_model(loss_vae_normal, 128, 200, 5e-4)
