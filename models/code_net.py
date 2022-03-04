import torch
import torch.nn as nn
from models.modules import Encode, Decode, UNet, reparameterize, initialize_weights, Jacobian


class CodeNet(nn.Module):
    def __init__(self, in_channels, latent_dim, average_depth):
        super(CodeNet, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.average_depth = average_depth
        self.stage_num = 3
        self.stages = tuple(f"stage{idx}" for idx in range(1, self.stage_num + 1))
        self.unet = UNet(self.in_channels)
        self.image_encode = Encode(self.in_channels, self.latent_dim)
        self.depth_encode = Encode(1, self.latent_dim)
        self.depth_decode = Decode(latent_dim=self.latent_dim)
        # self.jacobian = Jacobian(self.depth_decode)
        initialize_weights(self)

    def forward(self, img, depth, get_jacobian=False):
        f512, f256, f128, f64, std_pyr0, std_pyr1, std_pyr2 = self.unet(
            img)  # f** means image feature with n channels for concat
        image_mu, image_logvar = self.image_encode(img, f512, f256, f128, f64)
        depth_mu, depth_logvar = self.depth_encode(depth, f512, f256, f128, f64)
        image_code = reparameterize(image_mu, image_logvar)
        depth_code = reparameterize(depth_mu, depth_logvar)
        depth_rec = self.depth_decode(depth_code, f512, f256, f128, f64)
        depth_pred = self.depth_decode(image_code, f512, f256, f128, f64)
        jacobian = None
        if get_jacobian:
            jacobian = self.generate_jacobian(f512, f256, f128, f64)
        return {'image_mu': image_mu, 'image_logvar': image_logvar, 'depth_pred': depth_pred, 'depth_mu': depth_mu,
                'depth_logvar': depth_logvar, 'depth_rec': depth_rec, 'b': [std_pyr0, std_pyr1, std_pyr2],
                'jacobian': jacobian}

    def outputs_to_dict(self, outputs):
        for i, stage in enumerate(self.stages):
            outputs.update({stage: {'depth': torch.squeeze(self.average_depth - self.average_depth
                                                           / (outputs['depth_pred'][2 - i] + 1e-10), dim=1)}})
        return outputs

    def generate_jacobian(self, f512, f256, f128, f64):
        assert not self.training, "Training is not available for jacobian"
        self.jacobian = Jacobian(self.depth_decode)
        J = self.depth_decode.decoder_linear.weight
        J = J.view(512, 12, 16, 32).permute(3, 0, 1, 2)
        jacobians = self.jacobian(J, f512, f256, f128, f64)
        return jacobians
