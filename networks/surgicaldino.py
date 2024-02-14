from __future__ import absolute_import, division, print_function

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# from mmcv.cnn import ConvModule

import sys,os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lib'))

from dinov2.eval.depth.ops import resize
from torch.nn.parameter import Parameter

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
                x = x[0]
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

class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            lora_a_q: nn.Module,
            lora_b_q: nn.Module,
            lora_a_v: nn.Module,
            lora_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.lora_a_q = lora_a_q
        self.lora_b_q = lora_b_q
        self.lora_a_v = lora_a_v
        self.lora_b_v = lora_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,3*org_C
        new_q = self.lora_b_q(self.lora_a_q(x))
        new_v = self.lora_b_v(self.lora_a_v(x))
        # qkv[:, :, :, : self.dim] += new_q
        # qkv[:, :, :, -self.dim:] += new_v
        
        qkv[:, :, : self.dim] += new_q
        qkv[:, :, -self.dim:] += new_v
        return qkv
    
class SurgicalDino(nn.Module):
    """Applies low-rank adaptation to a Dinov2 model's image encoder.

    Args:
        backbone_size: size of pretrained Dinov2 choice from: "small", "base", "large", "giant"
        r: rank of LoRA
        image_shape: input image shape, h,w need to be multiplier of 14, default:(224,280)
        output_shape: output image shape, h,w need to be multiplier of 32, default:(256,320)
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.
    """

    def __init__(self, backbone_size = "base", r=4, image_shape=(224,280), output_shape=(256,320), lora_layer=None):
        super(SurgicalDino, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        self.num_ch_enc = np.array([1536, 1536, 1536, 1536, 1536])
        self.backbone_size = backbone_size
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14",
            "large": "vitl14",
            "giant": "vitg14",
        }
        self.intermediate_layers = {
            "small": [2, 4, 7, 9, 11],
            "base": [2, 4, 7, 9, 11],
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
        dinov2 = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=self.backbone_name)
        self.image_shape = image_shape
        self.output_shape = output_shape
        
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(dinov2.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_lora_As = []  # These are linear layers
        self.w_lora_Bs = []
        # lets freeze first
        for param in dinov2.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(dinov2.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_lora_q = nn.Linear(self.dim, r, bias=False)
            w_b_lora_q = nn.Linear(r, self.dim, bias=False)
            w_a_lora_v = nn.Linear(self.dim, r, bias=False)
            w_b_lora_v = nn.Linear(r, self.dim, bias=False)
            self.w_lora_As.append(w_a_lora_q)
            self.w_lora_Bs.append(w_b_lora_q)
            self.w_lora_As.append(w_a_lora_v)
            self.w_lora_Bs.append(w_b_lora_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_lora_q,
                w_b_lora_q,
                w_a_lora_v,
                w_b_lora_v,
            )
        self.reset_parameters()
        self.dinov2 = dinov2
        # The output feature numbers and shape
        self.inchannels = [self.embedding_dim,self.embedding_dim,self.embedding_dim,self.embedding_dim, self.embedding_dim]
        self.channels = self.embedding_dim*10
        self.in_index = (0, 1, 2, 3, 4)
        self.input_transform="resize_concat"
            
        self.forward_feature = forward_feature(image_shape=self.image_shape, 
                                              input_transform=self.input_transform,
                                              in_index=self.in_index)
    
    def save_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_lora_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_lora_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_lora_Bs[i].weight for i in range(num_layer)}

        merged_dict = {**a_tensors, **b_tensors}
        torch.save(merged_dict, filename)

        print('saved lora parameters to %s.' % filename)

    def load_parameters(self, filename: str, device: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename, map_location=device)

        for i, w_A_linear in enumerate(self.w_lora_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_lora_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        print('loaded lora parameters from %s.' % filename)

    def reset_parameters(self) -> None:
        for w_A in self.w_lora_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_lora_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, pixel_values):
        pixel_values = torch.nn.functional.interpolate(pixel_values, size=self.image_shape, mode="bilinear", align_corners=True)
        feature = self.dinov2.get_intermediate_layers(pixel_values, n=self.n, reshape = True, return_class_token = True, norm=False)
        pred = self.forward_feature(feature)
        
        for i in range(len(pred)):
            rescale_shape = [int(self.output_shape[0] / 2**(i+1)), int(self.output_shape[1] / 2**(i+1))]
            pred[i] = torch.nn.functional.interpolate(pred[i], size=rescale_shape, mode="bilinear", align_corners=True)
        # print(pred.shape)
        return pred
    