import torch
from jaxtyping import Bool, Float, Int
from typing import Any, Generator, Optional
from einops import rearrange, reduce, repeat
def _convert(
        self,
        patch_size: int,
        past_target: Float[torch.Tensor, "batch past_time tgt"],
        past_observed_target: Bool[torch.Tensor, "batch past_time tgt"],
        past_is_pad: Bool[torch.Tensor, "batch past_time"],
        future_target: Optional[Float[torch.Tensor, "batch future_time tgt"]] = None,
        future_observed_target: Optional[
            Bool[torch.Tensor, "batch future_time tgt"]
        ] = None,
        future_is_pad: Optional[Bool[torch.Tensor, "batch future_time"]] = None,
        feat_dynamic_real: Optional[Float[torch.Tensor, "batch time feat"]] = None,
        observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch time feat"]
        ] = None,
        past_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
        past_observed_feat_dynamic_real: Optional[
            Float[torch.Tensor, "batch past_time past_feat"]
        ] = None,
) -> tuple[
    Float[torch.Tensor, "batch combine_seq patch"],  # target
    Bool[torch.Tensor, "batch combine_seq patch"],  # observed_mask
    Int[torch.Tensor, "batch combine_seq"],  # sample_id
    Int[torch.Tensor, "batch combine_seq"],  # time_id
    Int[torch.Tensor, "batch combine_seq"],  # variate_id
    Bool[torch.Tensor, "batch combine_seq"],  # prediction_mask
]:
    batch_shape = past_target.shape[:-2]
    device = past_target.device

    target = []
    observed_mask = []
    sample_id = []
    time_id = []
    variate_id = []
    prediction_mask = []
    dim_count = 0

    past_seq_id, future_seq_id = self._generate_time_id(
        patch_size, past_observed_target
    )

    if future_target is None:
        future_target = torch.zeros(
            batch_shape
            + (
                self.hparams.prediction_length,
                past_target.shape[-1],
            ),
            dtype=past_target.dtype,
            device=device,
        )
    target.extend(
        [
            torch.nn.functional.pad(
                rearrange(
                    self._patched_seq_pad(patch_size, past_target, -2, left=True),
                    "... (seq patch) dim -> ... (dim seq) patch",
                    patch=patch_size,
                ),
                (0, self.max_patch_size - patch_size),
            ),
            torch.nn.functional.pad(
                rearrange(
                    self._patched_seq_pad(
                        patch_size, future_target, -2, left=False
                    ),
                    "... (seq patch) dim -> ... (dim seq) patch",
                    patch=patch_size,
                ),
                (0, self.max_patch_size - patch_size),
            ),
        ]
    )
    if future_observed_target is None:
        future_observed_target = torch.ones(
            batch_shape
            + (
                self.hparams.prediction_length,
                past_observed_target.shape[-1],
            ),
            dtype=torch.bool,
            device=device,
        )
    observed_mask.extend(
        [
            torch.nn.functional.pad(
                rearrange(
                    self._patched_seq_pad(
                        patch_size, past_observed_target, -2, left=True
                    ),
                    "... (seq patch) dim -> ... (dim seq) patch",
                    patch=patch_size,
                ),
                (0, self.max_patch_size - patch_size),
            ),
            torch.nn.functional.pad(
                rearrange(
                    self._patched_seq_pad(
                        patch_size, future_observed_target, -2, left=False
                    ),
                    "... (seq patch) dim -> ... (dim seq) patch",
                    patch=patch_size,
                ),
                (0, self.max_patch_size - patch_size),
            ),
        ]
    )
    if future_is_pad is None:
        future_is_pad = torch.zeros(
            batch_shape + (self.hparams.prediction_length,),
            dtype=torch.long,
            device=device,
        )
    sample_id.extend(
        [
            repeat(
                reduce(
                    (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True, value=1
                            )
                            == 0
                    ).int(),
                    "... (seq patch) -> ... seq",
                    "max",
                    patch=patch_size,
                ),
                "... seq -> ... (dim seq)",
                dim=past_target.shape[-1],
            ),
            repeat(
                reduce(
                    (
                            self._patched_seq_pad(
                                patch_size, future_is_pad, -1, left=False, value=1
                            )
                            == 0
                    ).int(),
                    "... (seq patch) -> ... seq",
                    "max",
                    patch=patch_size,
                ),
                "... seq -> ... (dim seq)",
                dim=past_target.shape[-1],
            ),
        ]
    )
    time_id.extend(
        [past_seq_id] * past_target.shape[-1]
        + [future_seq_id] * past_target.shape[-1]
    )
    variate_id.extend(
        [
            repeat(
                torch.arange(past_target.shape[-1], device=device) + dim_count,
                f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                past=self.context_token_length(patch_size),
            ),
            repeat(
                torch.arange(past_target.shape[-1], device=device) + dim_count,
                f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                future=self.prediction_token_length(patch_size),
            ),
        ]
    )
    dim_count += past_target.shape[-1]
    prediction_mask.extend(
        [
            torch.zeros(
                batch_shape
                + (self.context_token_length(patch_size) * past_target.shape[-1],),
                dtype=torch.bool,
                device=device,
            ),
            torch.ones(
                batch_shape
                + (
                    self.prediction_token_length(patch_size)
                    * past_target.shape[-1],
                ),
                dtype=torch.bool,
                device=device,
            ),
        ]
    )

    if feat_dynamic_real is not None:
        if observed_feat_dynamic_real is None:
            raise ValueError(
                "observed_feat_dynamic_real must be provided if feat_dynamic_real is provided"
            )

        target.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size,
                            feat_dynamic_real[
                            ..., : self.hparams.context_length, :
                            ],
                            -2,
                            left=True,
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size,
                            feat_dynamic_real[
                            ..., self.hparams.context_length:, :
                            ],
                            -2,
                            left=False,
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
            ]
        )
        observed_mask.extend(
            [
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size,
                            observed_feat_dynamic_real[
                            ..., : self.hparams.context_length, :
                            ],
                            -2,
                            left=True,
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
                torch.nn.functional.pad(
                    rearrange(
                        self._patched_seq_pad(
                            patch_size,
                            observed_feat_dynamic_real[
                            ..., self.hparams.context_length:, :
                            ],
                            -2,
                            left=False,
                        ),
                        "... (seq patch) dim -> ... (dim seq) patch",
                        patch=patch_size,
                    ),
                    (0, self.max_patch_size - patch_size),
                ),
            ]
        )
        sample_id.extend(
            [
                repeat(
                    reduce(
                        (
                                self._patched_seq_pad(
                                    patch_size, past_is_pad, -1, left=True
                                )
                                == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=feat_dynamic_real.shape[-1],
                ),
                torch.ones(
                    batch_shape
                    + (
                        self.prediction_token_length(patch_size)
                        * feat_dynamic_real.shape[-1],
                    ),
                    dtype=torch.long,
                    device=device,
                ),
            ]
        )
        time_id.extend(
            [past_seq_id] * feat_dynamic_real.shape[-1]
            + [future_seq_id] * feat_dynamic_real.shape[-1]
        )
        variate_id.extend(
            [
                repeat(
                    torch.arange(feat_dynamic_real.shape[-1], device=device)
                    + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                ),
                repeat(
                    torch.arange(feat_dynamic_real.shape[-1], device=device)
                    + dim_count,
                    f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                    future=self.prediction_token_length(patch_size),
                ),
            ]
        )
        dim_count += feat_dynamic_real.shape[-1]
        prediction_mask.extend(
            [
                torch.zeros(
                    batch_shape
                    + (
                        self.context_token_length(patch_size)
                        * feat_dynamic_real.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                ),
                torch.zeros(
                    batch_shape
                    + (
                        self.prediction_token_length(patch_size)
                        * feat_dynamic_real.shape[-1],
                    ),
                    dtype=torch.bool,
                    device=device,
                ),
            ]
        )

    if past_feat_dynamic_real is not None:
        if past_observed_feat_dynamic_real is None:
            raise ValueError(
                "past_observed_feat_dynamic_real must be provided if past_feat_dynamic_real is provided"
            )
        target.append(
            torch.nn.functional.pad(
                rearrange(
                    self._patched_seq_pad(
                        patch_size, past_feat_dynamic_real, -2, left=True
                    ),
                    "... (seq patch) dim -> ... (dim seq) patch",
                    patch=patch_size,
                ),
                (0, self.max_patch_size - patch_size),
            )
        )
        observed_mask.append(
            torch.nn.functional.pad(
                rearrange(
                    self._patched_seq_pad(
                        patch_size, past_observed_feat_dynamic_real, -2, left=True
                    ),
                    "... (seq patch) dim -> ... (dim seq) patch",
                    patch=patch_size,
                ),
                (0, self.max_patch_size - patch_size),
            )
        )
        sample_id.append(
            repeat(
                reduce(
                    (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True
                            )
                            == 0
                    ).int(),
                    "... (seq patch) -> ... seq",
                    "max",
                    patch=patch_size,
                ),
                "... seq -> ... (dim seq)",
                dim=past_feat_dynamic_real.shape[-1],
            )
        )
        time_id.extend([past_seq_id] * past_feat_dynamic_real.shape[-1])

        variate_id.append(
            repeat(
                torch.arange(past_feat_dynamic_real.shape[-1], device=device)
                + dim_count,
                f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                past=self.context_token_length(patch_size),
            )
        )
        dim_count += past_feat_dynamic_real.shape[-1]
        prediction_mask.append(
            torch.zeros(
                batch_shape
                + (
                    self.context_token_length(patch_size)
                    * past_feat_dynamic_real.shape[-1],
                ),
                dtype=torch.bool,
                device=device,
            )
        )

    target = torch.cat(target, dim=-2)
    observed_mask = torch.cat(observed_mask, dim=-2)
    sample_id = torch.cat(sample_id, dim=-1)
    time_id = torch.cat(time_id, dim=-1)
    variate_id = torch.cat(variate_id, dim=-1)
    prediction_mask = torch.cat(prediction_mask, dim=-1)
    return (
        target,
        observed_mask,
        sample_id,
        time_id,
        variate_id,
        prediction_mask,
    )