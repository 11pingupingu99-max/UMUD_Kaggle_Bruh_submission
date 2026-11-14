"""R45T models: R22 MuscleNet (fascicles + axial orientation), ApoNet v2, legacy R12B apo.
SPDX-License-Identifier: GPL-3.0-or-later
"""
from pathlib import Path
import json
import os
import numpy as np
import cv2
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny

R22_VERSION = 'R22_v1.0.0'
APO2_VERSION = 'R45T_APO2_v1.0.0'


class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__(); g = min(8, co)
        self.net = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.GroupNorm(g, co), nn.SiLU(),
                                 nn.Conv2d(co, co, 3, padding=1, bias=False), nn.GroupNorm(g, co))
        self.skip = nn.Conv2d(ci, co, 1, bias=False) if ci != co else nn.Identity()

    def forward(self, x):
        return F.silu(self.net(x) + self.skip(x))


class MuscleNet(nn.Module):
    """Exact R22 architecture (module names identical, so R22 checkpoints load strictly).
    out_channels=3: [fascicle logit, axial cos2t, axial sin2t]; out_channels=1: aponeurosis logit."""
    def __init__(self, out_channels=3):
        super().__init__()
        self.encoder = convnext_tiny(weights=None, stochastic_depth_prob=.1).features
        self.stem_full = Block(1, 24)
        self.stem_half = nn.Sequential(nn.AvgPool2d(2), Block(24, 32))
        self.up16 = Block(768 + 384, 192)
        self.up8 = Block(192 + 192, 96)
        self.up4 = Block(96 + 96, 64)
        self.up2 = Block(64 + 32, 32)
        self.up1 = Block(32 + 24, 24)
        self.head = nn.Conv2d(24, out_channels, 1)
        nn.init.normal_(self.head.weight, std=.01)
        with torch.no_grad():
            bias = torch.tensor([-2., 1., 0.]) if out_channels == 3 else torch.full((out_channels,), -2.)
            self.head.bias.copy_(bias)
        self.register_buffer('rgb_mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('rgb_std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, x):
        full = self.stem_full(x); half = self.stem_half(full)
        z = (x.expand(-1, 3, -1, -1) - self.rgb_mean) / self.rgb_std
        features = []
        for i, layer in enumerate(self.encoder):
            z = layer(z)
            if i in (1, 3, 5, 7): features.append(z)

        def up(z, skip, block):
            z = F.interpolate(z, size=skip.shape[-2:], mode='bilinear', align_corners=False)
            return block(torch.cat((z, skip), 1))
        z = up(features[3], features[2], self.up16)
        z = up(z, features[1], self.up8); z = up(z, features[0], self.up4)
        z = up(z, half, self.up2); z = up(z, full, self.up1)
        return self.head(z)


def load_r22(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('version') != R22_VERSION:
        raise ValueError(f'Not an R22 checkpoint: {path}')
    model = MuscleNet(3); model.load_state_dict(state['model'], strict=True)
    return model.to(device).eval(), {k: state.get(k) for k in ('fold', 'epoch', 'identity')}


def new_apo2(pretrained_encoder=None):
    model = MuscleNet(1)
    if pretrained_encoder is not None:
        state = torch.load(pretrained_encoder, map_location='cpu', weights_only=True)
        model.encoder.load_state_dict(state['encoder'], strict=True)
    return model


def load_apo2(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('version') != APO2_VERSION:
        raise ValueError(f'Not an R45T apo checkpoint: {path}')
    model = MuscleNet(1)
    model.load_state_dict({k: v.float() for k, v in state['model'].items()}, strict=True)
    return model.to(device).eval(), {k: state.get(k) for k in ('fold', 'epoch', 'identity')}


class LegacyDinoApo(nn.Module):
    """Exact R12B architecture. Never downloads weights."""
    def __init__(self, size=518):
        super().__init__(); import timm
        self.enc = timm.create_model('vit_small_patch14_dinov2.lvd142m', pretrained=False, num_classes=0, img_size=size)
        self.grid = size // self.enc.patch_embed.patch_size[0]; self.size = size
        self.dec = nn.Sequential(nn.Conv2d(self.enc.embed_dim, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
                                 nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(), nn.Conv2d(64, 1, 1))

    def forward(self, x):
        t = self.enc.forward_features(x)[:, -self.grid * self.grid:]
        z = t.transpose(1, 2).reshape(x.shape[0], -1, self.grid, self.grid)
        return F.interpolate(self.dec(z), size=(self.size, self.size), mode='bilinear', align_corners=False)


def load_r12b(path, device='cpu'):
    d = torch.load(path, map_location='cpu', weights_only=True); sd = d.get('model', d.get('state_dict', d))
    m = LegacyDinoApo(); m.load_state_dict(sd, strict=True); return m.to(device).eval()


def _autocast(device):
    return torch.amp.autocast(device_type='cuda' if device == 'cuda' else 'cpu', dtype=torch.float16, enabled=device == 'cuda')


@torch.inference_mode()
def predict_r22(models, square, device='cpu', tta=True):
    """Mean over models (and horizontal flip) of fascicle probability and of the UNIT axial vector.
    Returns p, c, s (side x side, float32). For the flipped pass the map is flipped back and sin2t negated."""
    x = torch.from_numpy(np.ascontiguousarray(square[None, None], dtype=np.float32)).to(device)
    views = [(x, False)] + ([(torch.flip(x, (-1,)), True)] if tta else [])
    p = 0.; c = 0.; s = 0.; n = 0
    for model in models:
        for xin, flipped in views:
            with _autocast(device):
                out = model(xin)
            out = out.float()
            if flipped:
                out = torch.flip(out, (-1,))
            prob = torch.sigmoid(out[:, :1])
            v = F.normalize(out[:, 1:3], dim=1, eps=1e-6)
            if flipped:
                v = v * torch.tensor([1., -1.], device=v.device)[None, :, None, None]
            p = p + prob; c = c + v[:, :1]; s = s + v[:, 1:2]; n += 1
    to = lambda t: (t / n)[0, 0].cpu().numpy().astype(np.float32)
    return to(p), to(c), to(s)


@torch.inference_mode()
def predict_apo2(models, square, device='cpu', tta=True):
    x = torch.from_numpy(np.ascontiguousarray(square[None, None], dtype=np.float32)).to(device)
    views = [(x, False)] + ([(torch.flip(x, (-1,)), True)] if tta else [])
    p = 0.; n = 0
    for model in models:
        for xin, flipped in views:
            with _autocast(device):
                out = model(xin).float()
            if flipped:
                out = torch.flip(out, (-1,))
            p = p + torch.sigmoid(out[:, :1]); n += 1
    return (p / n)[0, 0].cpu().numpy().astype(np.float32)


@torch.inference_mode()
def predict_r12b(model, path, device='cpu'):
    """Exact R12B uint8 BGR->RGB anisotropic 518x518 contract; probability at native size."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None: raise ValueError(path)
    h, w = image.shape[:2]; x = cv2.resize(image, (518, 518), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32) / 255.
    x = (x - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    t = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)[None])).to(device)
    p = torch.sigmoid(model(t).float())[0, 0].cpu().numpy()
    return cv2.resize(p, (w, h), interpolation=cv2.INTER_LINEAR)


def atomic_torch(path, state):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.pt'); torch.save(state, tmp); os.replace(tmp, path)
