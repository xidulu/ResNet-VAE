class cifar_config:
    latent_dim = 128
    ch = 64

    @property
    def encoder_arc(self):
        _encoder_arc = {
            "latent_dim": self.latent_dim,
            "in_channels": [3] + [item * self.ch for item in [16]],
            "out_channels": [item * self.ch for item in [16, 16]],
            "downsample": [True, True, False],
            "attention": [False, False, False, False]
        }
        return _encoder_arc

    @property
    def decoder_arc(self):
        _decoder_arc = {
            "latent_dim": self.latent_dim,
            "in_channels": [self.ch * item for item in [16,16]],
            "out_channels": [self.ch * item for item in [16,16]],
            "upsample": [True, True],
            "attention": [False, False],
            "bottom_width": 8
        }
        return _decoder_arc