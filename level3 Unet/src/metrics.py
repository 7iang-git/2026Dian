import numpy as np
import torch
from skimage.metrics import structural_similarity


def calculate_batch_metrics(predicted, target):
    predicted = torch.clamp(predicted.detach(), 0.0, 1.0)
    target = torch.clamp(target.detach(), 0.0, 1.0)

    l1_values = torch.abs(predicted - target).flatten(1).mean(dim=1)

    mse_values = (
        (predicted - target).pow(2).flatten(1).mean(dim=1)
    )

    psnr_values = 10.0 * torch.log10(
        1.0 / torch.clamp(mse_values, min=1e-12)
    )

    predicted_np = (
        predicted.cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )

    target_np = (
        target.cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )

    ssim_values = []

    for index in range(predicted_np.shape[0]):
        try:
            value = structural_similarity(
                target_np[index],
                predicted_np[index],
                channel_axis=-1,
                data_range=1.0,
            )
        except TypeError:
            value = structural_similarity(
                target_np[index],
                predicted_np[index],
                multichannel=True,
                data_range=1.0,
            )

        ssim_values.append(value)

    return {
        "l1": l1_values.mean().item(),
        "psnr": psnr_values.mean().item(),
        "ssim": float(np.mean(ssim_values)),
    }