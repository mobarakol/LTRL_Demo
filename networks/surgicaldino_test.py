from __future__ import absolute_import, division, print_function

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import loralib

# from mmcv.cnn import ConvModule

import sys,os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from dinov2.eval.depth.ops import resize
from torch.nn.parameter import Parameter
from transformers import Dinov2Backbone

class forward_feature(torch.nn.Module):

    def __init__(self, input_transform="resize_concat", image_shape=(224,224), in_index=(0, 1, 2, 3), upsample=4, align_corners=False,):
        super().__init__()
        self.input_transform = input_transform
        self.image_shape = image_shape
        self.in_index = in_index
        self.upsample = upsample
        self.align_corners = align_corners

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if "concat" in self.input_transform:
            inputs = [inputs[i] for i in self.in_index]
            if "resize" in self.input_transform:
                inputs = [
                    resize(
                        input=x,
                        size=[s * self.upsample for s in inputs[0].shape[2:]],
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for x in inputs
                ]
            # inputs = torch.cat(inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # accept lists (for cls token)
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if len(x) == 2:
                x, cls_token = x[0], x[1]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            else:
                # x = x[0]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                inputs[i] = x
        x = self._transform_inputs(inputs)
        # feats = self.bn(x)
        return x
    
    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        # print(output.shape)
        
        return output

class _IncreLoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            w_linear_q: loralib.SVDLinear,
            w_linear_v: loralib.SVDLinear,
    ):
        super().__init__()
        self.qkv = qkv
        self.w_linear_q = w_linear_q
        self.w_linear_v = w_linear_v
        self.dim = qkv.in_features

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.w_linear_q(x)
        new_v = self.w_linear_v(x)
        # qkv[:, :, :, : self.dim] += new_q
        # qkv[:, :, :, -self.dim:] += new_v
        
        qkv[:, :, : self.dim] = new_q
        qkv[:, :, -self.dim:] = new_v
        return qkv
    
class IncreDino(nn.Module):
    """Applies low-rank adaptation to a Dinov2 model's image encoder.

    Args:
        backbone_size: size of pretrained Dinov2 choice from: "small", "base", "large", "giant"
        r: rank of LoRA
        image_shape: input image shape, h,w need to be multiplier of 14, default:(224,280)
        output_shape: output image shape, h,w need to be multiplier of 32, default:(256,320)
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.
    """

    def __init__(self, backbone_size = "base", r=4, image_shape=(224,280), output_shape=(256,320), lora_layer=None, pretrained_path=None):
        super(IncreDino, self).__init__()

        assert r > 0
        self.r = r
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        self.backbone_size = backbone_size
        self.intermediate_layers = {
            "small": [2, 5, 8, 10, 12],
            "base": [2, 5, 8, 10, 12],
            "large": [4, 11, 17, 23],
            "giant": [9, 19, 29, 39],
        }
        self.embedding_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        self.out_indices = self.intermediate_layers[self.backbone_size]
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        self.backbone_name = f"facebook/dinov2-{self.backbone_size}"
        self.num_ch_enc = np.array([self.embedding_dim, self.embedding_dim, self.embedding_dim,
                                    self.embedding_dim, self.embedding_dim])
        
        dinov2 = Dinov2Backbone.from_pretrained("facebook/dinov2-base", out_indices=self.out_indices)
        self.image_shape = image_shape
        self.output_shape = output_shape
        
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(dinov2.encoder.layer)))  # Only apply lora to the image encoder by default

        # Apply IncreLoRA on mlps
        for t_layer_i, blk in enumerate(dinov2.encoder.layer):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            
            q_in_features = blk.attention.attention.query.in_features
            v_in_features = blk.attention.attention.value.in_features
            blk.attention.attention.query = loralib.Linear(q_in_features, q_in_features, 
                                                    r=self.r, lora_alpha=self.r, merge_weights=False)
            blk.attention.attention.value = loralib.Linear(v_in_features, v_in_features, 
                                                    r=self.r, lora_alpha=self.r, merge_weights=False)

        # load the pretrained weight again
        if pretrained_path is not None:
            pretrained_path = os.path.join(pretrained_path, "dinov2-{}.pth".format(self.backbone_size))
            encoder_dict = torch.load(pretrained_path)
            dinov2.load_state_dict(encoder_dict, strict=False)
            print("load pretrained weight from {}".format(pretrained_path))
        
        loralib.mark_only_lora_as_trainable(dinov2)
        self.dinov2 = dinov2
        
        # The output feature numbers and shape
        self.inchannels = [self.embedding_dim,self.embedding_dim,self.embedding_dim,self.embedding_dim, self.embedding_dim]
        self.channels = self.embedding_dim*10
        self.in_index = (0, 1, 2, 3, 4)
        self.input_transform="resize_concat"
            
        self.forward_feature = forward_feature(image_shape=self.image_shape, 
                                              input_transform=self.input_transform,
                                              in_index=self.in_index)

    def forward(self, pixel_values):
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=self.image_shape, mode="bilinear", align_corners=True)
        feature = self.dinov2(pixel_values).feature_maps
        pred = self.forward_feature(feature)
        output = []
        feature_length = len(pred)
        
        for i in range(feature_length):
            rescale_shape = [int(self.output_shape[0] / 2**(i+1)), int(self.output_shape[1] / 2**(i+1))]
            # output.append(torch.nn.functional.interpolate(pred[feature_length-1-i], size=rescale_shape, mode="bilinear", align_corners=True))
            output.append(torch.nn.functional.interpolate(pred[i], size=rescale_shape, mode="bilinear", align_corners=True))

        return output
    
class AdaDino(nn.Module):
    """Applies low-rank adaptation to a Dinov2 model's image encoder.

    Args:
        backbone_size: size of pretrained Dinov2 choice from: "small", "base", "large", "giant"
        r: rank of LoRA
        image_shape: input image shape, h,w need to be multiplier of 14, default:(224,280)
        output_shape: output image shape, h,w need to be multiplier of 32, default:(256,320)
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.
    """

    def __init__(self, backbone_size = "base", r=4, image_shape=(224,280), output_shape=(256,320), lora_layer=None, pretrained_path=None):
        super(AdaDino, self).__init__()

        assert r > 0
        self.r = r
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        self.backbone_size = backbone_size
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        self.intermediate_layers = {
            "small": [3, 5, 7, 9, 11],
            "base": [3, 5, 7, 9, 11],
            "large": [4, 11, 17, 23],
            "giant": [9, 19, 29, 39],
        }
        self.embedding_dims = {
            "small": 384,
            "base": 768,
            "large": 1024,
            "giant": 1536,
        }
        self.backbone_arch = self.backbone_archs[self.backbone_size]
        self.n = self.intermediate_layers[self.backbone_size]
        self.embedding_dim = self.embedding_dims[self.backbone_size]
        self.backbone_name = f"dinov2_{self.backbone_arch}"
        self.num_ch_enc = np.array([self.embedding_dim, self.embedding_dim, self.embedding_dim,
                                    self.embedding_dim, self.embedding_dim]) * 2
        
        dinov2 = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.backbone_name)
        self.image_shape = image_shape
        self.output_shape = output_shape
        
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(dinov2.blocks)))  # Only apply lora to the image encoder by default

        # Apply IncreLoRA on mlps
        for t_layer_i, blk in enumerate(dinov2.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            
            # qkv_in_features = blk.attn.qkv.in_features
            # blk.attn.qkv = loralib.MergedLinear(qkv_in_features, 3*qkv_in_features, r=self.r, 
            #                                     lora_alpha=self.r, merge_weights=False, enable_lora=[True, False, True])
            
            mlp_in_features = blk.mlp.fc1.in_features
            mlp_hidden_features = blk.mlp.fc1.out_features
            mlp_out_features = blk.mlp.fc2.out_features
            blk.mlp.fc1 = loralib.SVDLinear(mlp_in_features, mlp_hidden_features, r=self.r, lora_alpha=self.r, merge_weights=False)
            blk.mlp.fc2 = loralib.SVDLinear(mlp_hidden_features, mlp_out_features, r=self.r, lora_alpha=self.r, merge_weights=False)

        # load the pretrained weight again
        if pretrained_path is not None:
            pretrained_path = os.path.join(pretrained_path, "dinov2_{}.pth".format(self.backbone_arch))
            encoder_dict = torch.load(pretrained_path)
            dinov2.load_state_dict(encoder_dict, strict=False)
            print("load pretrained weight from {}\n".format(pretrained_path))
        
        loralib.mark_only_lora_as_trainable(dinov2)
        self.dinov2 = dinov2
        
        # The output feature numbers and shape
        self.in_index = (0, 1, 2, 3, 4)
        self.input_transform="resize_concat"
            
        self.forward_feature = forward_feature(image_shape=self.image_shape, 
                                              input_transform=self.input_transform,
                                              in_index=self.in_index)

    def forward(self, pixel_values):
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=self.image_shape, mode="bilinear", align_corners=True)
        feature = self.dinov2.get_intermediate_layers(pixel_values, n=self.n, reshape = True, return_class_token = True, norm=False)
        pred = self.forward_feature(feature)
        output = []
        feature_length = len(pred)
        
        for i in range(feature_length):
            rescale_shape = [int(self.output_shape[0] / 2**(i+1)), int(self.output_shape[1] / 2**(i+1))]
            # output.append(torch.nn.functional.interpolate(pred[feature_length-1-i], size=rescale_shape, mode="bilinear", align_corners=True))
            output.append(torch.nn.functional.interpolate(pred[i], size=rescale_shape, mode="bilinear", align_corners=True))

        return output