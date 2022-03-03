import torch
from torch import nn
from torch.nn import functional as F
from typing import List, Callable, Union, Any, TypeVar, Tuple
from collections import OrderedDict

# from torch import tensor as Tensor

Tensor = TypeVar('torch.tensor')


def l1_depth_loss_term(depth_est, depth_gt, mask, keep_batch=False, stage=None):
    depth_loss = F.l1_loss(depth_est * mask, depth_gt *
                           mask, reduction='none')  # (B, H_stage, W_stage)
    depth_loss = torch.mean(depth_loss, (1, 2)) / \
                 torch.mean(mask, (1, 2))  # (B, )

    if not keep_batch:
        depth_loss = torch.mean(depth_loss)

    return depth_loss


def abs_rel_loss_term(depth_est, depth_gt, mask, eps=0.01, stage=None):
    abs_rel = mask * torch.abs(depth_est - depth_gt) / (depth_gt + eps)
    abs_rel = torch.mean(abs_rel, (1, 2)) / torch.mean(mask, (1, 2))  # (B, )
    abs_rel = torch.mean(abs_rel)

    return abs_rel


def compute_loss(outputs: dict, batch: dict, weights: tuple, loss_terms: tuple, term_weights: tuple,
                 keep_batch=False):
    """
    :param outputs:
        Outputs from model.
    :param batch:
    :param weights:
    :return:
    """
    depth_preds = outputs['depth_pred']
    depth_recs = outputs['depth_rec']
    uncertaintys = outputs['b']
    image_kld_loss = torch.mean(
        -0.5 * torch.sum(1 + outputs['image_logvar'] - outputs['image_mu'] ** 2 - outputs['image_logvar'].exp(), dim=1),
        dim=0)
    depth_kld_loss = torch.mean(
        -0.5 * torch.sum(1 + outputs['depth_logvar'] - outputs['depth_mu'] ** 2 - outputs['depth_logvar'].exp(), dim=1),
        dim=0)
    if type(outputs['depth_rec']) not in [list, tuple]:
        depth_preds = [outputs['depth_pred']]
        depth_recs = [outputs['depth_rec']]
    depth_pred_loss_total = 0
    depth_rec_loss_total = 0
    layer_scale = 4  # This scale is describe in CodeSLAM
    for depth_pred, depth_rec, uncertainty in zip(depth_preds, depth_recs, uncertaintys):
        b, _, h, w = depth_pred.shape
        labels_scaled = F.interpolate(batch['gt_norm'], (h, w), mode='bilinear')
        labels_scaled_mask = (labels_scaled < 1).float()
        depth_pred = depth_pred * labels_scaled_mask
        depth_rec = depth_rec * labels_scaled_mask
        uncertainty_mask = (uncertainty < -4).detach()
        uncertainty[uncertainty_mask] = -4.
        # This loss is DeepFactors's loss
        depth_pred_loss = F.l1_loss(depth_pred, labels_scaled) * layer_scale
        depth_rec_loss = torch.mean(
            (torch.abs((depth_rec - labels_scaled)) / uncertainty.exp() + uncertainty)) * layer_scale + 16
        depth_pred_loss_total += depth_pred_loss
        depth_rec_loss_total += depth_rec_loss

    total_losses = depth_pred_loss_total / len(depth_preds) + weights[0] * image_kld_loss + \
                   depth_rec_loss_total / len(depth_recs) + weights[0] * depth_kld_loss
    losses = {'depth_pred_loss': depth_pred_loss_total / len(depth_preds),
              'depth_rec_loss': depth_rec_loss_total / (len(depth_recs)),
              'image_kld_loss': -image_kld_loss, 'depth_kld_loss': -depth_kld_loss,
              'total_loss': total_losses}
    return total_losses, losses

# def unnormalize_depth(outputs: dict, batch: dict, average_depth: float):
#     outputs_depth = {}
#     for i, stage in enumerate(['stage1', 'stage2', 'stage3']):
#         outputs_depth[stage]['depth'] = average_depth - average_depth / (outputs['depth_pred'][i] + 1e-5)
#         batch['depth'][stage] = average_depth - average_depth / (outputs['depth_pred'][i] + 1e-5)


def eval_errors(outputs: dict, batch: dict, keep_batch=False) -> dict:
    """
    :param outputs:
        Outputs from model.
    :param batch:
    :return:
    """
    errors = {}
    for stage in ('stage1', 'stage2', 'stage3'):
        # type: torch.FloatTensor # (B, H_stage, W_stage)
        depth_est = outputs[stage]['depth']
        # type: torch.FloatTensor # (B, H_stage, W_stage)
        depth_gt = batch['depth'][stage]
        # type: torch.FloatTensor # (B, H_stage, W_stage)
        mask = batch['mask'][stage]
        assert depth_gt.size() == mask.size()

        batch_size = depth_est.size(0)

        if not keep_batch:
            errors[stage] = OrderedDict([
                ('abs_rel', 0.0),
                ('abs', 0.0),
                ('sq_rel', 0.0),
                ('rmse', 0.0),
                ('rmse_log', 0.0),
                ('a1', 0.0),
                ('a2', 0.0),
                ('a3', 0.0),
                ('d1', 0.0),
                ('d2', 0.0),
                ('d3', 0.0),
            ])
        else:
            errors[stage] = OrderedDict([
                ('abs_rel', []),
                ('abs', []),
                ('sq_rel', []),
                ('rmse', []),
                ('rmse_log', []),
                ('a1', []),
                ('a2', []),
                ('a3', []),
                ('d1', []),
                ('d2', []),
                ('d3', []),
            ])

        for gt, est, m in zip(depth_gt, depth_est, mask):
            gt = gt[m > 0.5]
            est = est[m > 0.5]
            abs_rel = torch.abs(gt - est) / gt
            a1 = (abs_rel < 0.01).to(torch.float32).mean()
            a2 = (abs_rel < 0.1 ** 2).to(torch.float32).mean()
            a3 = (abs_rel < 0.1 ** 3).to(torch.float32).mean()
            abs_rel = torch.mean(abs_rel)

            d_val = torch.max(gt / est, est / gt)
            d1 = (d_val < 1.25).to(torch.float32).mean()
            d2 = (d_val < 1.25 ** 2).to(torch.float32).mean()
            d3 = (d_val < 1.25 ** 3).to(torch.float32).mean()

            rmse = torch.sqrt(torch.mean((gt - est) ** 2))
            rmse_log = torch.sqrt(torch.mean(
                (torch.log(gt) - torch.log(est)) ** 2))
            sq_rel = torch.mean(((gt - est) ** 2) / gt)

            abs_abs = torch.mean(torch.abs(gt - est))

            if not keep_batch:
                errors[stage]['abs_rel'] += abs_rel
                errors[stage]['abs'] += abs_abs
                errors[stage]['sq_rel'] += sq_rel
                errors[stage]['rmse'] += rmse
                errors[stage]['rmse_log'] += rmse_log
                errors[stage]['a1'] += a1
                errors[stage]['a2'] += a2
                errors[stage]['a3'] += a3
                errors[stage]['d1'] += d1
                errors[stage]['d2'] += d2
                errors[stage]['d3'] += d3
            else:
                errors[stage]['abs_rel'].append(abs_rel)
                errors[stage]['abs'].append(abs_abs)
                errors[stage]['sq_rel'].append(sq_rel)
                errors[stage]['rmse'].append(rmse)
                errors[stage]['rmse_log'].append(rmse_log)
                errors[stage]['a1'].append(a1)
                errors[stage]['a2'].append(a2)
                errors[stage]['a3'].append(a3)
                errors[stage]['d1'].append(d1)
                errors[stage]['d2'].append(d2)
                errors[stage]['d3'].append(d3)

        if not keep_batch:
            for k in errors[stage]:
                errors[stage][k] = errors[stage][k] / batch_size
        else:
            for k in errors[stage]:
                errors[stage][k] = torch.stack(errors[stage][k])

    return errors


def reparameterize(mu, logvar):
    """
    Reparameterization trick to sample from N(mu, var) from
    N(0,1).
    :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    :return: (Tensor) [B x D]
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return eps * std + mu


def conv2d_block(in_channels, out_channels, kernel=3, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride, padding),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(inplace=True)
    )


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


class EncodeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EncodeConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=2 * out_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True))

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv2(x)


class DecodeConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecodeConv, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.conv2 = nn.Conv2d(3 * out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x1, x2):
        x = self.up(x1)
        x = self.conv1(x)
        element_wise_mul = torch.mul(x, x2)
        x = torch.cat([x, x2, element_wise_mul], dim=1)
        x = self.conv2(x)

        return x


class Encode(nn.Module):
    def __init__(self, in_channels, latent_dim=32):
        super(Encode, self).__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        self.encode0 = EncodeConv(in_channels, 64)  # [B, 64, 96, 128]
        self.encode1 = EncodeConv(64, 128)  # [B, 128, 48, 64]
        self.encode2 = EncodeConv(128, 256)  # [B, 256, 24, 32]
        self.encode3 = EncodeConv(256, 512)  # [B, 512, 12, 16]
        self.linear0 = nn.Sequential(
            nn.Linear(512 * 12 * 16, 512),
            nn.LeakyReLU(inplace=True)
        )
        self.linear1 = nn.Sequential(
            nn.Linear(512, 512),
            nn.LeakyReLU(inplace=True)
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_var = nn.Linear(512, latent_dim)

    def forward(self, x, *image_feature):
        f512, f256, f128, f64 = image_feature  # f** means image feature with n channels for concat

        e = self.encode0(x, f64)
        e = self.encode1(e, f128)
        e = self.encode2(e, f256)
        e = self.encode3(e, f512)
        e = torch.flatten(e, start_dim=1)
        e = self.linear0(e)
        e = self.linear1(e)
        mu = self.fc_mu(e)
        log_var = self.fc_var(e)
        return [mu, log_var]


class Decode(nn.Module):
    def __init__(self, latent_dim):
        super(Decode, self).__init__()
        self.latent_dim = latent_dim
        self.decoder_linear = nn.Linear(latent_dim, 512 * 16 * 12)
        self.conv0 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(512 * 3, 512, kernel_size=3, stride=1, padding=1)
        self.decode0 = DecodeConv(512, 256)
        self.decode1 = DecodeConv(256, 128)
        self.decode2 = DecodeConv(128, 64)
        self.dpt_pyr0 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=1)
        self.dpt_pyr1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.dpt_pyr2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, *image_feature):
        f512, f256, f128, f64 = image_feature  # f** means image feature with n channels for concat
        d0 = self.decoder_linear(x)
        d0 = d0.view(-1, 512, 12, 16)
        d0 = self.conv0(d0)  # [B, 512, 12, 16]
        element_wise_mul = torch.mul(d0, f512)
        x = torch.cat([d0, f512, element_wise_mul], dim=1)
        d0 = self.conv1(x)  # [B, 512, 12, 16]
        d256 = self.decode0(d0, f256)  # [B, 256, 24, 32]
        d128 = self.decode1(d256, f128)  # [B, 128, 48, 64]
        d64 = self.decode2(d128, f64)  # [B, 64, 96, 128]
        dpt_pyr0 = self.dpt_pyr0(d64, [d64.shape[0], 1, 192, 256])
        dpt_pyr1 = self.dpt_pyr1(d64)
        dpt_pyr2 = self.dpt_pyr2(d128)
        return [dpt_pyr0, dpt_pyr1, dpt_pyr2]


class UpAndConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpAndConv, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv1(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv2(x)


class UNet(nn.Module):
    def __init__(self, in_channels):
        super(UNet, self).__init__()
        # input [B, C, 192, 256]
        self.enc0 = conv2d_block(in_channels=in_channels, out_channels=64, kernel=7, stride=2,
                                 padding=3)  # [B, 64, 96, 128]
        self.enc1 = conv2d_block(in_channels=64, out_channels=128, kernel=3, stride=2,
                                 padding=1)  # [B, 128, 48, 64]
        self.enc2 = conv2d_block(in_channels=128, out_channels=256, kernel=3, stride=2,
                                 padding=1)  # [B, 256, 24, 32]
        self.enc3 = conv2d_block(in_channels=256, out_channels=512, kernel=3, stride=2,
                                 padding=1)  # [B, 512, 12, 16]

        self.dec0 = conv2d_block(in_channels=512, out_channels=512, kernel=3, stride=1,
                                 padding=1)  # [B, 256, 12, 16]
        self.dec1 = UpAndConv(in_channels=512, out_channels=256)
        self.dec2 = UpAndConv(in_channels=256, out_channels=128)
        self.dec3 = UpAndConv(in_channels=128, out_channels=64)
        self.std_pyr0 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=3, stride=2, padding=1)
        self.std_pyr1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.std_pyr2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x0 = self.enc0(x)  # [B, 64, 96, 128]
        x1 = self.enc1(x0)  # [B, 128, 48, 64]
        x2 = self.enc2(x1)  # [B, 256, 24, 32]
        x3 = self.enc3(x2)  # [B, 512, 12, 16]
        f512 = self.dec0(x3)  # [B, 512, 12, 16]
        f256 = self.dec1(f512, x2)  # [B, 256, 24, 32]
        f128 = self.dec2(f256, x1)  # [B, 128, 48, 64]
        f64 = self.dec3(f128, x0)  # [B, 64, 96, 128]
        std_pyr0 = self.std_pyr0(f64, [f64.shape[0], 1, 192, 256])  # [B, 1, 192, 256]
        std_pyr1 = self.std_pyr1(f64)  # [B, 1, 96, 128]
        std_pyr2 = self.std_pyr2(f128)  # [B, 1, 48, 64]
        return [f512, f256, f128, f64, std_pyr0, std_pyr1, std_pyr2]


class Jacobian(nn.Module):
    def __init__(self, decode):
        super(Jacobian, self).__init__()
        self.decode = decode

    def mulconcat(self, x, y):
        zero_features = torch.zeros_like(x)
        y = y.repeat(32, 1, 1, 1)
        element_wise_mul = torch.mul(x, y)
        x = torch.cat([x, zero_features, element_wise_mul], dim=1)
        return x

    def decodeconv(self, x, f, d):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = F.conv2d(x, d.conv1.weight, stride=1, padding=1)
        c = self.mulconcat(x, f)
        x = F.conv2d(c, d.conv2.weight, stride=1, padding=1)
        return x

    def forward(self, x, *image_feature):
        f512, f256, f128, f64 = image_feature  # f** means image feature with n channels for concat
        x = F.conv2d(x, self.decode.conv0.weight, stride=1, padding=1)
        c = self.mulconcat(x, f512)
        x = F.conv2d(c, self.decode.conv1.weight, stride=1, padding=1)
        x = self.decodeconv(x, f256, self.decode.decode0)
        x = self.decodeconv(x, f128, self.decode.decode1)
        x128 = x
        x = self.decodeconv(x, f64, self.decode.decode2)
        jacobian_dpt_pyr0 = F.conv_transpose2d(x, self.decode.dpt_pyr0.weight, stride=2, padding=1,
                                               output_padding=1).permute(1, 2, 3, 0)
        jacobian_dpt_pyr1 = F.conv2d(x, self.decode.dpt_pyr1.weight, stride=1, padding=1).permute(1, 2, 3, 0)
        jacobian_dpt_pyr2 = F.conv2d(x128, self.decode.dpt_pyr2.weight, stride=1, padding=1).permute(1, 2, 3, 0)
        return [jacobian_dpt_pyr0, jacobian_dpt_pyr1, jacobian_dpt_pyr2]
