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
import lightning as L

class EncoderModel(nn.Module):
    def __init__(self, pretrained_model, patch_size_int=128):
        super().__init__()
        self.model = pretrained_model
        self.patch_size_int = patch_size_int

    def forward(self, x):
        target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size = x

        print('target', target.shape)
        print('observed_mask', observed_mask.shape)
        print('sample_id', sample_id.shape)
        print('time_id', time_id.shape)
        print('variate_id', variate_id.shape)
        print('prediction_mask', prediction_mask.shape)

        loc, scale = self.model.scaler(target, observed_mask * ~prediction_mask.unsqueeze(-1),
                                        sample_id,
                                        variate_id, )

        print('loc', loc.shape)
        print('scale', scale.shape)

        scaled_target = (target - loc) / scale
        print(patch_size.shape)

        reprs = self.model.in_proj(scaled_target, patch_size)
        masked_reprs = mask_fill(reprs, prediction_mask,self.model.mask_encoding.weight)
        encoded = self.model.encoder(
            masked_reprs,
            packed_attention_mask(sample_id),
            time_id=time_id,
            var_id=variate_id,
        )

        return encoded

    def forward(self, x):
        target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size = x

        print('target', target.shape)
        print('observed_mask', observed_mask.shape)
        print('sample_id', sample_id.shape)
        print('time_id', time_id.shape)
        print('variate_id', variate_id.shape)
        print('prediction_mask', prediction_mask.shape)

        loc, scale = self.model.scaler(target, observed_mask * ~prediction_mask.unsqueeze(-1),
                                        sample_id,
                                        variate_id, )

        print('loc', loc.shape)
        print('scale', scale.shape)

        scaled_target = (target - loc) / scale
        # print(patch_size.shape)

        reprs = self.model.in_proj(scaled_target, patch_size)
        masked_reprs = mask_fill(reprs, prediction_mask,self.model.mask_encoding.weight)
        encoded = self.model.encoder(
            masked_reprs,
            packed_attention_mask(sample_id),
            time_id=time_id,
            var_id=variate_id,
        )

        return encoded

class LightningWrapper(L.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.linear(384,2)

    def training_step(self, batch, batch_idx):
        data_zip, label = batch
        z = self.encoder(data_zip)

        logits = self.fc(z)


        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer