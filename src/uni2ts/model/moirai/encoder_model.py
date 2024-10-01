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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


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

        scaled_target = (target - loc) / scale

        reprs = self.model.in_proj(scaled_target, patch_size)
        masked_reprs = mask_fill(reprs, prediction_mask, self.model.mask_encoding.weight)
        encoded = self.model.encoder(
            masked_reprs,
            packed_attention_mask(sample_id),
            time_id=time_id,
            var_id=variate_id,
        )

        return encoded, scaled_target

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


class Pretraining(L.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        self.fc1 = nn.Linear(384, 256)
        self.fc2_1 = nn.Linear(256, 128)
        # self.fc1_2 = nn.Linear(384, 256)
        self.fc2_2 = nn.Linear(256, 128)
        # self.fc1_3 = nn.Linear(384, 256)
        self.fc2_3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.criterion = nn.MSELoss(reduction='mean')

    def process_unmasked_data(self, data, mask, time_id):
        unmasked_data = []

        for batch_n in range(data.shape[0]):
            dt_masked = data[batch_n][mask[batch_n]]
            t_id_last = time_id[batch_n][mask[batch_n]][-1] + 1
            reformatted = rearrange(dt_masked, '(s t) f -> s t f', t=t_id_last)
            pooled = reformatted.mean(dim=0)

            unmasked_data.append(pooled)

        return torch.concat(unmasked_data, dim=0)

    def generate_symmetrical_mask(self, sample_id, prediction_mask, ratio = 0.1):
        # Find indices where sample_id_tensor equals 1
        valid_indices = torch.nonzero(sample_id == 1)
        total_valid_indices = len(valid_indices)

        # Randomly select half of the valid indices
        mask_size = int(total_valid_indices * ratio)

        # Randomly select 'mask1_size' number of indices
        selected_random_indices_1 = torch.randperm(total_valid_indices)[:mask_size]
        selected_random_indices_2 = torch.randperm(total_valid_indices)[-mask_size:]

        # Create boolean masks for splitting the valid indices
        mask1_selector = torch.zeros(total_valid_indices, dtype=torch.bool)
        mask2_selector = torch.zeros(total_valid_indices, dtype=torch.bool)
        mask1_selector[selected_random_indices_1] = True
        mask2_selector[selected_random_indices_2] = True
        # mask2_selector = ~mask1_selector

        # Initialize output masks with the same shape as full_mask
        mask1 = torch.zeros_like(prediction_mask, dtype=torch.bool)
        mask2 = torch.zeros_like(prediction_mask, dtype=torch.bool)

        # Apply the selected and complementary indices to their respective masks
        mask1[valid_indices[mask1_selector][:, 0], valid_indices[mask1_selector][:, 1]] = True
        mask2[valid_indices[mask2_selector][:, 0], valid_indices[mask2_selector][:, 1]] = True

        return mask1, mask2

    def prepare_unmasked_data_label(self, z, data_zip, label):
        sample_id = data_zip[2]
        time_id = data_zip[3]
        mask = sample_id.to(dtype=torch.bool)
        unmasked_z = self.process_unmasked_data(z, mask, time_id)
        unmasked_label = label[mask[:, :label.shape[1]]]

        return unmasked_z, unmasked_label

    def prepare_unmasked_data_channel_label(self, z, data_zip, label):
        sample_id = data_zip[2]
        mask = sample_id.to(dtype=torch.bool)
        unmasked_z = z[mask]
        unmasked_label = label[mask]

        return unmasked_z, unmasked_label

    def std_norm(self, x, batch_n):
        # print(range(batch_n.max()+1))
        # print(batch_n)
        for i in range(batch_n.max() + 1):
            # print(i)
            idx = batch_n == i
            mean = torch.mean(x[idx])
            std = torch.std(x[idx])
            x[idx] = (x[idx] - mean) / (std + 1e-6)
        return x

    def training_step(self, batch, batch_idx):
        data_zip, label, batch_n = batch
        mask_1, mask_2 = self.generate_symmetrical_mask(data_zip[2], data_zip[5])
        data_zip[5] = mask_1
        z_1, scaled_target = self.encoder(data_zip)
        z_1 = z_1[mask_1]
        _x_1_amp = self.fc2_1(self.relu(self.fc1(z_1)))
        _x_1_pha = self.fc2_2(self.relu(self.fc1(z_1)))
        _x_1 = self.fc2_3(self.relu(self.fc1(z_1)))
        batch_n_1 = batch_n[mask_1]
        data_zip[5] = mask_2
        z_2, scaled_target = self.encoder(data_zip)
        z_2 = z_2[mask_2]
        _x_2_amp = self.fc2_1(self.relu(self.fc1(z_2)))
        _x_2_pha = self.fc2_2(self.relu(self.fc1(z_2)))
        _x_2 = self.fc2_3(self.relu(self.fc1(z_2)))
        batch_n_2 = batch_n[mask_2]
        # print(batch_n_1.shape, batch_n_2.shape)

        pred_concat_amp = torch.cat((_x_1_amp, _x_2_amp), dim=0)
        pred_concat_pha = torch.cat((_x_1_pha, _x_2_pha), dim=0)
        pred_concat = torch.cat((_x_1, _x_2))
        # print(pred_concat.shape)
        target_concat = torch.cat((scaled_target[mask_1], scaled_target[mask_2]), dim=0)
        # print(target_concat.shape)
        batch_n = torch.cat((batch_n_1, batch_n_2), dim=0)
        # print(batch_n.shape)

        target_fft = torch.fft.fft(target_concat, dim=-1)
        target_amplitude = torch.abs(target_fft)
        target_amplitude = self.std_norm(target_amplitude, batch_n)
        target_phase = torch.angle(target_fft)
        target_phase = self.std_norm(target_phase, batch_n)

        # reconstructed_fft = torch.fft.fft(pred_concat, dim = -1)
        # reconstructed_amplitude = torch.abs(reconstructed_fft)
        # reconstructed_amplitude = self.std_norm(reconstructed_amplitude, batch_n)
        # reconstructed_phase = torch.angle(reconstructed_fft)
        # reconstructed_phase = self.std_norm(reconstructed_phase, batch_n)

        reconstruction_loss = torch.nn.functional.mse_loss(target_concat, pred_concat)
        amplitude_loss = torch.nn.functional.smooth_l1_loss(target_amplitude, pred_concat_amp)
        phase_loss = torch.nn.functional.smooth_l1_loss(target_phase, pred_concat_pha)

        loss = amplitude_loss + phase_loss + reconstruction_loss
        self.log('amp_loss', amplitude_loss, prog_bar=True)
        self.log('phase_loss', phase_loss, prog_bar=True)
        self.log('recon_loss', reconstruction_loss, prog_bar=True)

        self.log('loss', loss, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        data_zip, label = batch
        self.encoder.eval()
        with torch.no_grad():
            z = self.encoder(data_zip)
        unmasked_z, unmasked_label = self.prepare_unmasked_data_channel_label(z, data_zip, label)
        logits = self.fc2(self.relu(self.fc1(unmasked_z)))

        return data_zip, unmasked_z, unmasked_label, logits

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=5e-5, betas=(0.9, 0.99))

        # Scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)

        # Lightning expects a dictionary if you want to return both the optimizer and scheduler
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss'  # Optional: metric to monitor for scheduling
        }


class LinearProbing(L.LightningModule):
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

    def prepare_unmasked_data_channel_label(self, z, data_zip, label):
        sample_id = data_zip[2]
        mask = sample_id.to(dtype=torch.bool)
        unmasked_z = z[mask]
        unmasked_label = label[mask]

        return unmasked_z, unmasked_label

    def training_step(self, batch, batch_idx):
        data_zip, label, _ = batch
        self.encoder.eval()
        with torch.no_grad():
            z, _ = self.encoder(data_zip)
        unmasked_z, unmasked_label = self.prepare_unmasked_data_channel_label(z, data_zip, label)
        logits = self.fc2(self.relu(self.fc1(unmasked_z)))
        loss = self.criterion(logits, unmasked_label.long())

        self.log('loss', loss, prog_bar=True)

        return loss

    def predict_step(self, batch, batch_idx):
        data_zip, label, _ = batch
        self.encoder.eval()
        with torch.no_grad():
            z, _ = self.encoder(data_zip)
        unmasked_z, unmasked_label = self.prepare_unmasked_data_channel_label(z, data_zip, label)
        logits = self.fc2(self.relu(self.fc1(unmasked_z)))

        return data_zip, unmasked_z, unmasked_label, logits

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        return optimizer
