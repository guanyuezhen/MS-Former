import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from .decoder import FPN
from .former import Block


class WCDNet(nn.Module):
    def __init__(self, patch_size=32, memory_length=128, depth=3):
        super(WCDNet, self).__init__()
        self.patch_size = patch_size
        self.decoder_channel = 128
        self.embedding_channel = 128
        self.memory_length = memory_length
        self.depth = depth
        # decoder
        channels = [64, 64, 128, 256, 512]
        self.context_encoder = timm.create_model('resnet18d', features_only=True, pretrained=True)
        self.fpn_net = FPN(channels, self.decoder_channel)
        # attention
        self.memory_tokens = nn.Embedding(self.memory_length, self.embedding_channel)
        self.pixel_feature_tokens = nn.Conv2d(self.decoder_channel, self.embedding_channel, kernel_size=1)
        self.attention = nn.ModuleList(
            [Block(dim=self.embedding_channel, num_heads=2, mlp_ratio=4) for i in range(self.depth)]
        )
        # mask
        self.mask_generation = nn.Conv2d(self.embedding_channel, 1, kernel_size=1)
        self.region_mask_generation = nn.ModuleList(
            [nn.Conv2d(self.embedding_channel, 1, kernel_size=1) for i in range(self.depth)]
        )

    def forward(self, x, gt=None):
        test_mode = gt is None
        size = x.size()[2:]
        # temporal difference information extraction
        t1 = x[:, 0:3, :, :]
        t2 = x[:, 3:6, :, :]
        t1_c1, t1_c2, t1_c3, t1_c4, t1_c5 = self.context_encoder(t1)
        t2_c1, t2_c2, t2_c3, t2_c4, t2_c5 = self.context_encoder(t2)
        #
        c5 = torch.abs(t1_c5 - t2_c5)
        c4 = torch.abs(t1_c4 - t2_c4)
        c3 = torch.abs(t1_c3 - t2_c3)
        c2 = torch.abs(t1_c2 - t2_c2)
        p_out = self.fpn_net(c2, c3, c4, c5)
        # attention
        pixel_feature_tokens = self.pixel_feature_tokens(p_out)
        B, C, H, W = pixel_feature_tokens.size()
        memory_tokens = self.memory_tokens.weight.unsqueeze(0).repeat(B, 1, 1)
        pixel_feature_tokens = pixel_feature_tokens.flatten(2).transpose(1, 2)
        region_mask = []
        for idx, a_block in enumerate(self.attention):
            pixel_feature_tokens, memory_tokens, mp = \
                a_block(pixel_feature_tokens, memory_tokens, H, W, self.patch_size)
            region_mask.append(self.region_mask_generation[idx](mp))

        pixel_feature_tokens = pixel_feature_tokens.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        # mask
        mask = self.mask_generation(pixel_feature_tokens)
        change_mask = F.interpolate(mask, size=size, mode='bilinear', align_corners=True)
        change_mask = torch.sigmoid(change_mask)
        change_mask_aux = F.adaptive_max_pool2d(mask, (size[0] // self.patch_size, size[1] // self.patch_size))
        change_mask_aux = torch.sigmoid(change_mask_aux)
        region_mask = torch.cat(region_mask, dim=1)
        region_mask = torch.sigmoid(region_mask)

        if test_mode:
            return change_mask

        return change_mask, change_mask_aux, region_mask


def get_model(patch_size, memory_length, depth):
    model = WCDNet(patch_size, memory_length, depth)

    return model