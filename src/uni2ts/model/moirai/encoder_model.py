from functools import partial

import torch
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin
from hydra.utils import instantiate
from jaxtyping import Bool, Float, Int
from torch import nn
from torch.distributions import Distribution
from torch.utils._pytree import tree_map
from einops import rearrange

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

        # print('target', target.shape)
        # print('observed_mask', observed_mask.shape)
        # print('sample_id', sample_id.shape)
        # print('time_id', time_id.shape)
        # print('variate_id', variate_id.shape)
        # print('prediction_mask', prediction_mask.shape)

        loc, scale = self.model.scaler(target, observed_mask * ~prediction_mask.unsqueeze(-1),
                                       sample_id,
                                       variate_id, )

        # print('loc', loc.shape)
        # print('scale', scale.shape)
        dmin = -512
        dmax = 511

        scaled_target = (target - loc) / scale
        # scaled_target = (target - dmin) / (dmax - dmin)
        # print(patch_size.shape)

        reprs = self.model.in_proj(scaled_target, patch_size)
        masked_reprs = mask_fill(reprs, prediction_mask, self.model.mask_encoding.weight)
        encoded = self.model.encoder(
            masked_reprs,
            packed_attention_mask(sample_id),
            time_id=time_id,
            var_id=variate_id,
        )

        return encoded

    # def forward(self, x):
    #     target, observed_mask, sample_id, time_id, variate_id, prediction_mask, patch_size = x
    #
    #     print('target', target.shape)
    #     print('observed_mask', observed_mask.shape)
    #     print('sample_id', sample_id.shape)
    #     print('time_id', time_id.shape)
    #     print('variate_id', variate_id.shape)
    #     print('prediction_mask', prediction_mask.shape)
    #
    #     loc, scale = self.model.scaler(target, observed_mask * ~prediction_mask.unsqueeze(-1),
    #                                     sample_id,
    #                                     variate_id, )
    #
    #     print('loc', loc.shape)
    #     print('scale', scale.shape)
    #
    #     scaled_target = (target - loc) / scale
    #     print(patch_size.shape)
    #
    #     reprs = self.model.in_proj(scaled_target, patch_size)
    #     masked_reprs = mask_fill(reprs, prediction_mask,self.model.mask_encoding.weight)
    #     encoded = self.model.encoder(
    #         masked_reprs,
    #         packed_attention_mask(sample_id),
    #         time_id=time_id,
    #         var_id=variate_id,
    #     )
    #
    #     return encoded


class LightningWrapper(L.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def process_unmasked_data(self, data, mask, time_id):
        unmasked_data = []

        for batch_n in range(data.shape[0]):
            dt_masked = data[batch_n][mask[batch_n]]
            t_id_last = time_id[batch_n][mask[batch_n]][-1] + 1
            reformatted = rearrange(dt_masked, '(s t) f -> s t f', t=t_id_last)
            pooled = reformatted.mean(dim=0)

            unmasked_data.append(pooled)

        return torch.concat(unmasked_data, dim=0)

    def prepare_unmasked_data_label(self, z, data_zip, label):
        sample_id = data_zip[2]
        time_id = data_zip[3]
        mask = sample_id.to(dtype=torch.bool)
        unmasked_z = self.process_unmasked_data(z, mask, time_id)
        unmasked_label = label[mask[:, :label.shape[1]]]

        return unmasked_z, unmasked_label

    def training_step(self, batch, batch_idx):
        data_zip, label = batch
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(data_zip)
        unmasked_z, unmasked_label = self.prepare_unmasked_data_label(z, data_zip, label)
        logits = self.fc2(self.relu(self.fc1(unmasked_z)))
        loss = self.criterion(logits, unmasked_label.long())

        self.log('loss', loss, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        data_zip, label = batch
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(data_zip)
        unmasked_z, unmasked_label = self.prepare_unmasked_data_label(z, data_zip, label)
        logits = self.fc2(self.relu(self.fc1(unmasked_z)))

        return data_zip, unmasked_z, unmasked_label, logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)

        return optimizer
