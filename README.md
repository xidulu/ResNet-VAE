# ResNet-VAE
Can't find a satisfying VAE implementation on Github, so I decide to write one myself.


## Usage

`python train.py`


## Features

1. Closed form update for the variance of p(x | z)
2. Network structure borrowed from BigGAN
3. Parameterize the variance of q(z | x) by directly squaring the output of the network. (surprisingly stable compared with exp or softplus)
