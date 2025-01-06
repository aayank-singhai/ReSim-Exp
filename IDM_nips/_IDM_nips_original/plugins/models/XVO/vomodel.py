import torch
import torch.nn as nn
from timm.models.vision_transformer import _create_vision_transformer
from torchvision.models.feature_extraction import create_feature_extractor

from .flownet import FlowNetS
from .lstm import LSTM


class VOModel(nn.Module):
    def __init__(self):
        super(VOModel, self).__init__()
        '''
        Encoder
        '''
        self.encoder = FlowNetS()

        params = dict(img_size=(96, 160), patch_size=(12, 16), embed_dim=256, depth=4, num_heads=4, num_classes=0,
                      in_chans=102, class_token=False, global_pool='')
        self.transformer = _create_vision_transformer("vit_base_patch16_224_in21k", pretrained=False, **params)
        self.transformer = create_feature_extractor(self.transformer, return_nodes={"norm": "feature"})

        '''
        Decoder
        '''
        self.decoder = LSTM(20480)

    def forward(self, x):
        batch_size = x.size(0)
        with torch.no_grad():
            x = self.encoder(x).detach()

        x = self.transformer(x)["feature"]
        pose, A = self.decoder(x.view(batch_size, -1))

        return pose, A


class VOModelEncoder(nn.Module):
    def __init__(self):
        super(VOModelEncoder, self).__init__()
        '''
        Encoder
        '''
        self.encoder = FlowNetS()

        params = dict(img_size=(96, 160), patch_size=(12, 16), embed_dim=256, depth=4, num_heads=4, num_classes=0,
                      in_chans=102, class_token=False, global_pool='')
        self.transformer = _create_vision_transformer("vit_base_patch16_224_in21k", pretrained=False, **params)
        self.transformer = create_feature_extractor(self.transformer, return_nodes={"norm": "feature"})

    def forward(self, x):
        batch_size = x.size(0)
        with torch.no_grad():
            x = self.encoder(x)
        x = self.transformer(x)["feature"]
        x = x.view(batch_size, -1)

        return x
