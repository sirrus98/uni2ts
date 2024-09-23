from functools import partial

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from hydra.utils import instantiate
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution
from torch.utils._pytree import tree_map

from uni2ts.common.torch_util import mask_fill, packed_attention_mask
from uni2ts.model.moirai.convert import _convert


class EncoderModel(nn.Module):
    def __init__(self, pretrained_model,patch_size_int=128):
        super().__init__()
        self.model = pretrained_model
        self.patch_size_int = patch_size_int
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def forward(self, x):
        past_target = torch.as_tensor(x).T.unsqueeze(0).to(dtype=torch.float32).to(self.device)
        past_observed_target = torch.ones_like(past_target, dtype=torch.bool).to(self.device)
        # 1s if the value is padding, 0s otherwise. Shape: (batch, time)
        past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)[:, :, 0].to(self.device)
        # past_is_pad = torch.zeros_like(past_target, dtype=torch.bool).squeeze(-1)

        target, observed_mask, sample_id, time_id, variate_id, prediction_mask = _convert(
            context_length=21000,
            patch_size=self.patch_size_int,
            past_target=past_target,
            past_observed_target=past_observed_target,
            past_is_pad=past_is_pad, )

        loc, scale = self.model.scaler(target, observed_mask * ~prediction_mask.unsqueeze(
            -1),
                                        sample_id,
                                        variate_id, )
        patch_size = torch.tensor([self.patch_size_int]).repeat(1, target.shape[1]).to(self.device)
        scaled_target = (target - loc) / scale
        print(scaled_target.device)
        reprs = self.model.in_proj(scaled_target, patch_size)

        masked_reprs = mask_fill(reprs, prediction_mask,
                                 self.model.mask_encoding.weight)
        encoded = self.model.encoder(
            masked_reprs,
            packed_attention_mask(sample_id),
            time_id=time_id,
            var_id=variate_id,
        )

        return encoded