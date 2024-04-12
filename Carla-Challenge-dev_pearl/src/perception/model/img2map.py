import math
import copy
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Optional, List
from timm.models.registry import register_model
from timm.models.resnet import resnet26d


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class HybridInputEmbed(nn.Module):
    def __init__(
        self,
        backbone,
        img_size=224,
        patch_size=1,
        feature_size=None,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        self.backbone = backbone

        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size, img_size))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)

        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x)
        global_x = torch.mean(x, [2, 3], keepdim=False)[:, :, None]
        return x, global_x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        bs, _, h, w = x.shape
        not_mask = torch.ones((bs, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)

        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)

        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class Img2map(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_heads=8,
        embed_dim=256,
        enc_depth=6,
        dec_depth=6,
        dim_feedforward=2048,
        dim_map=20 * 20,
        dropout=0.1,
        with_center_sensor=False,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.with_center_sensor = with_center_sensor

        self.rgb_backbone = resnet26d(
            pretrained=True,
            in_chans=in_chans,
            features_only=True,
            out_indices=[4],
        )

        self.lidar_backbone = resnet26d(
            pretrained=False,
            in_chans=in_chans,
            features_only=True,
            out_indices=[4],
        )

        self.rgb_patch_embed = HybridInputEmbed(
            backbone=self.rgb_backbone,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.lidar_patch_embed = HybridInputEmbed(
            backbone=self.lidar_backbone,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.global_embed = nn.Parameter(torch.zeros(1, embed_dim, 5))
        self.view_embed = nn.Parameter(torch.zeros(1, embed_dim, 5, 1))
        self.query_embed = nn.Parameter(torch.zeros(dim_map, 1, embed_dim))
        self.position_encoding = PositionEmbeddingSine(embed_dim // 2, normalize=True)

        encoder_layer = TransformerEncoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        self.encoder = TransformerEncoder(encoder_layer, enc_depth, None)

        decoder_layer = TransformerDecoderLayer(embed_dim, num_heads, dim_feedforward, dropout)
        decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder = TransformerDecoder(decoder_layer, dec_depth, decoder_norm)

        self.traffic_pred_head = nn.Sequential(
            nn.Linear(embed_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, 7),
            nn.Sigmoid(),
        )

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.global_embed)
        nn.init.uniform_(self.view_embed)
        nn.init.uniform_(self.query_embed)

    def forward_features(
        self,
        front_image,
        left_image,
        right_image,
        front_center_image,
        lidar,
        measurements,
    ):
        features = []

        # Front view processing
        front_image_token, front_image_token_global = self.rgb_patch_embed(front_image)
        front_image_token = (
            front_image_token
            + self.view_embed[:, :, 0:1, :]
            + self.position_encoding(front_image_token)
        )

        front_image_token = front_image_token.flatten(2).permute(2, 0, 1)
        front_image_token_global = (
            front_image_token_global + self.view_embed[:, :, 0, :] + self.global_embed[:, :, 0:1]
        )
        front_image_token_global = front_image_token_global.permute(2, 0, 1)
        features.extend([front_image_token, front_image_token_global])

        # Left view processing
        left_image_token, left_image_token_global = self.rgb_patch_embed(left_image)
        left_image_token = (
            left_image_token
            + self.view_embed[:, :, 1:2, :]
            + self.position_encoding(left_image_token)
        )

        left_image_token = left_image_token.flatten(2).permute(2, 0, 1)
        left_image_token_global = (
            left_image_token_global + self.view_embed[:, :, 1, :] + self.global_embed[:, :, 1:2]
        )
        left_image_token_global = left_image_token_global.permute(2, 0, 1)

        # Right view processing
        right_image_token, right_image_token_global = self.rgb_patch_embed(right_image)

        right_image_token = (
            right_image_token
            + self.view_embed[:, :, 2:3, :]
            + self.position_encoding(right_image_token)
        )

        right_image_token = right_image_token.flatten(2).permute(2, 0, 1)
        right_image_token_global = (
            right_image_token_global + self.view_embed[:, :, 2, :] + self.global_embed[:, :, 2:3]
        )
        right_image_token_global = right_image_token_global.permute(2, 0, 1)

        features.extend(
            [
                left_image_token,
                left_image_token_global,
                right_image_token,
                right_image_token_global,
            ]
        )

        # lidar feature processing
        lidar_token, lidar_token_global = self.lidar_patch_embed(lidar)
        lidar_token = (
            lidar_token + self.view_embed[:, :, 4:5, :] + self.position_encoding(lidar_token)
        )
        lidar_token = lidar_token.flatten(2).permute(2, 0, 1)
        lidar_token_global = (
            lidar_token_global + self.view_embed[:, :, 4, :] + self.global_embed[:, :, 4:5]
        )
        lidar_token_global = lidar_token_global.permute(2, 0, 1)
        features.extend([lidar_token, lidar_token_global])

        features = torch.cat(features, 0)
        return features

    def forward(self, x):
        front_image = x["rgb"]
        left_image = x["rgb_left"]
        right_image = x["rgb_right"]
        front_center_image = x["rgb_center"]
        measurements = x["measurements"]
        lidar = x["lidar"]

        features = self.forward_features(
            front_image,
            left_image,
            right_image,
            front_center_image,
            lidar,
            measurements,
        )

        # batchsize
        bs = front_image.shape[0]

        pos_encode = self.position_encoding(torch.ones((bs, 1, 20, 20), device=x["rgb"].device))
        pos_encode = pos_encode.flatten(2).permute(2, 0, 1)

        memory = self.encoder(features)
        output = self.decoder(self.query_embed.repeat(1, bs, 1), memory, query_pos=pos_encode)[0]

        output = output.permute(1, 0, 2)  # permute to [batchsize, length, dim]

        traffic_feature = output[:, :400]

        # cat velocity to decoder output feature to predict grid velocity
        velocity = measurements[:, 6:7].unsqueeze(-1)
        velocity = velocity.repeat(1, 400, 32)
        traffic_feature_with_vel = torch.cat([traffic_feature, velocity], dim=2)

        traffic = self.traffic_pred_head(traffic_feature_with_vel)

        return traffic, traffic_feature


@register_model
def img2map_model(**kwargs):
    model = Img2map(enc_depth=6, dec_depth=6, embed_dim=256)
    return model
