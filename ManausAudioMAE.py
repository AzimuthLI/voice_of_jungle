import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
import timm
from timm.models.vision_transformer import Block
from timm.models.layers import to_2tuple


# Utils for positional embeddings
# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

class PatchEmbed_org(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_hw = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        #assert H == self.img_size[0] and W == self.img_size[1], \
        #    f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ViTEncoder(nn.Module):
    
    def __init__(self, backbone_name, contextual_depth=8):
        super().__init__()
        self.backbone_model = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        
        # desamble the backbone model
        self.patch_embed = getattr(self.backbone_model, 'patch_embed')
        self.cls_token = getattr(self.backbone_model, 'cls_token')
        self.pos_embed = getattr(self.backbone_model, 'pos_embed')
        self.blocks = getattr(self.backbone_model, 'blocks')
        self.norm = getattr(self.backbone_model, 'fc_norm')
        
        self.embed_dim = self.backbone_model.embed_dim
        self.num_patches = self.patch_embed.num_patches
        self.encoder_depth = len(self.blocks)
        self.contextual_depth=contextual_depth
        
        # Initialize weights
        self.initialize_weights()
        
    def initialize_weights(self):
        # Initialize the weights of each module using the backbone model's weights
        self._init_from_backbone(self.patch_embed, getattr(self.backbone_model, 'patch_embed'))
        self._init_from_backbone(self.cls_token, getattr(self.backbone_model, 'cls_token'))
        self._init_from_backbone(self.pos_embed, getattr(self.backbone_model, 'pos_embed'))
        self._init_from_backbone(self.blocks, getattr(self.backbone_model, 'blocks'))
        self._init_from_backbone(self.norm, getattr(self.backbone_model, 'fc_norm'))
    
    def _init_from_backbone(self, target, source):
        if isinstance(target, nn.Module):
            target.load_state_dict(source.state_dict())
        elif isinstance(target, torch.Tensor):
            target.data.copy_(source.data)
        else:
            raise TypeError("Unsupported type for weight initialization")
        
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
        
    def forward(self, x, mask_ratio=0):
        
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        
        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        
        # apply Transformer blocks
        if mask_ratio > 0:
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            
            return x, mask, ids_restore, None
        else:
            contextual_features = []
            for n, blk in enumerate(self.blocks):
                x = blk(x)
                if n >= self.contextual_depth:
                    contextual_features.append(self.norm(x))
            
            # -> [N, L=513, D=768]
            return torch.stack(contextual_features, dim=0).mean(dim=0)
        

class ViTDecoder(nn.Module):
    
    def __init__(self, image_size, patch_size, in_chans, num_patches, encoder_dim, decoder_dim, num_layers, num_heads, ff_dim, pos_trainable=False):
        super().__init__()
        
        self.num_patches = num_patches
        self.in_chans = in_chans
        self.image_size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        self.patch_size = patch_size if isinstance(patch_size, (list, tuple)) else (patch_size, patch_size)
        self.patch_hw = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        
        self.decoder_embed = nn.Linear(encoder_dim, decoder_dim, bias=True)
        
        self.mask_token = nn.Parameter(torch.randn(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_dim), requires_grad=pos_trainable)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(
                dim=decoder_dim, num_heads=num_heads, qkv_bias=True, norm_layer=nn.LayerNorm
            )
            for _ in range(num_layers)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed_flexible(self.decoder_pos_embed.shape[-1], self.patch_hw, cls_token=True)
        
        print('decoder_pos_embed', decoder_pos_embed.shape, int(self.num_patches**.5))
        print('self.decoder_pos_embed', self.decoder_pos_embed.shape)
        
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed
        
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        pred = self.decoder_pred(x)

        # remove cls token
        return pred, None, None #emb, emb_pixel
    
    
class ViTMAE(nn.Module):
    
    def __init__(self, encoder_backbone, decoder_config, mask_ratio=0.75):
        super().__init__()
        self.enc_model = ViTEncoder(encoder_backbone)
        
        decoder_config['encoder_dim'] = self.enc_model.embed_dim
        decoder_config['num_patches'] = self.enc_model.num_patches
        
        self.dec_model = ViTDecoder(**decoder_config)
        
        self.image_size = self.dec_model.image_size
        self.patch_size = self.dec_model.patch_size
        self.in_chans = self.dec_model.in_chans
        self.mask_ratio = mask_ratio
        
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        L = (H/p)*(W/p)
        """
        
        h = imgs.shape[2] // self.patch_size[0]
        w = imgs.shape[3] // self.patch_size[1]
        
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, self.patch_size[0], w, self.patch_size[1]))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, self.patch_size[0]*self.patch_size[1]*self.in_chans))
        
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        specs: (N, 1, H, W)
        """
        h = self.image_size[0] // self.patch_size[0]
        w = self.image_size[1] // self.patch_size[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, self.patch_size[0], self.patch_size[1], self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        specs = x.reshape(shape=(x.shape[0], 1, h*self.patch_size[0], w*self.patch_size[1]))
        return specs
    
    def forward_loss(self, imgs, pred, mask, norm_pix_loss=False):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss   
        
    def forward(self, x):
        encode_x, mask, ids_restore, _ = self.enc_model(x, self.mask_ratio)
        pred, _, _ = self.dec_model(encode_x, ids_restore)
        pred = pred[:, 1:, :]
        reconstruct_loss = self.forward_loss(x, pred, mask, norm_pix_loss=False)
        
        return reconstruct_loss, pred, mask, ids_restore