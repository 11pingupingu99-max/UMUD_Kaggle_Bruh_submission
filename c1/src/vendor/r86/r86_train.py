"""R86T: third fascicle model F86 for the FL ensemble (R22 + F71 + F86).
Same decoder, heads, targets, loss, augmentation and training loop as F71 (r71_train), but a ConvNeXt-Small encoder
(same stage widths 96/192/384/768, deeper third stage) and a new seed - a different model whose errors are less correlated
with R22/F71 (both ConvNeXt-Tiny). install() swaps the constructor/loader used inside r71_train.
SPDX-License-Identifier: GPL-3.0-or-later
"""
import torch
from torchvision.models import convnext_small
import r45_models
import r71_train

F86_VERSION = 'R86T_FASC_F86_v1.0.0'
F86_CONFIG = dict(r71_train.F71_CONFIG, seed=20261106)


class MuscleNetS(r45_models.MuscleNet):
    """MuscleNet with a ConvNeXt-Small encoder; forward() and all decoder modules are inherited unchanged."""
    def __init__(self, out_channels=3):
        super().__init__(out_channels)
        self.encoder = convnext_small(weights=None, stochastic_depth_prob=.2).features


def new_f86(pretrained_encoder=None):
    model = MuscleNetS(3)
    if pretrained_encoder is not None:
        state = torch.load(pretrained_encoder, map_location='cpu', weights_only=True)
        model.encoder.load_state_dict(state['encoder'], strict=True)
    return model


def load_f86(path, device='cpu'):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('version') != F86_VERSION:
        raise ValueError(f'Not an F86 checkpoint: {path}')
    model = MuscleNetS(3)
    model.load_state_dict({k: v.float() for k, v in state['model'].items()}, strict=True)
    return model.to(device).eval(), {k: state.get(k) for k in ('fold', 'epoch', 'identity', 'cfg')}


def install():
    """Make r71_train build, save and load F86 instead of F71 (functions are looked up at call time)."""
    r71_train.F71_VERSION = F86_VERSION
    r71_train.new_f71 = new_f86
    r71_train.load_f71 = load_f86
    return F86_VERSION


def pretrained_encoder(path, log=print):
    """ImageNet ConvNeXt-Small encoder weights (torchvision), saved once as {'encoder': features.state_dict()}."""
    from torchvision.models import ConvNeXt_Small_Weights
    m = convnext_small(weights=None)
    m.load_state_dict(ConvNeXt_Small_Weights.IMAGENET1K_V1.get_state_dict(progress=True, check_hash=True))
    r45_models.atomic_torch(path, {'encoder': m.features.state_dict()})
    log('zapisano enkoder ConvNeXt-Small (ImageNet):', path)
    return path
