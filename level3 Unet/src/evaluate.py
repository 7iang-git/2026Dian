from pathlib import Path
import csv
import re

import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import PairedImageDataset
from model import UNet


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
METADATA_DIR = DATA_ROOT / "metadata"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "baseline_epoch1"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"

BATCH_SIZE = 2
NUM_WORKERS = 0


def calculate_batch_metrics(predicted, target):
    predicted = torch.clamp(predicted, 0.0, 1.0)
    target = torch.clamp(target, 0.0, 1.0)

    batch_size = predicted.shape[0]

    l1_values = torch.abs(predicted - target).flatten(1).mean(dim=1)

    mse_values = (
        (predicted - target).pow(2).flatten(1).mean(dim=1)
    )

    psnr_values = 10.0 * torch.log10(
        1.0 / torch.clamp(mse_values, min=1e-12)
    )

    predicted_np = (
        predicted.detach()
        .cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )

    target_np = (
        target.detach()
        .cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )

    ssim_values = []

    for index in range(batch_size):
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
        "count": batch_size,
        "l1_sum": l1_values.sum().item(),
        "psnr_sum": psnr_values.sum().item(),
        "ssim_sum": float(np.sum(ssim_values)),
    }


def evaluate_checkpoint(checkpoint_path, data_loader, device):
    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    total_count = 0
    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0

    with torch.no_grad():
        for input_images, target_images in data_loader:
            input_images = input_images.to(device)
            target_images = target_images.to(device)

            predicted_images = model(input_images)

            batch_metrics = calculate_batch_metrics(
                predicted_images,
                target_images,
            )

            total_count += batch_metrics["count"]
            total_l1 += batch_metrics["l1_sum"]
            total_psnr += batch_metrics["psnr_sum"]
            total_ssim += batch_metrics["ssim_sum"]

    return {
        "samples": total_count,
        "l1": total_l1 / total_count,
        "psnr": total_psnr / total_count,
        "ssim": total_ssim / total_count,
    }


def get_epoch(checkpoint_path):
    match = re.search(
        r"unet_epoch_(\d+)\.pth",
        checkpoint_path.name,
    )

    if match is None:
        return -1

    return int(match.group(1))


def save_csv(path, rows, fieldnames):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_validation_metrics(rows):
    epochs = [int(row["epoch"]) for row in rows]
    l1_values = [float(row["l1"]) for row in rows]
    psnr_values = [float(row["psnr"]) for row in rows]
    ssim_values = [float(row["ssim"]) for row in rows]

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(16, 5),
    )

    axes[0].plot(epochs, l1_values, marker="o")
    axes[0].set_title("Validation L1")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("L1")
    axes[0].grid(True)

    axes[1].plot(epochs, psnr_values, marker="o")
    axes[1].set_title("Validation PSNR")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("PSNR")
    axes[1].grid(True)

    axes[2].plot(epochs, ssim_values, marker="o")
    axes[2].set_title("Validation SSIM")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("SSIM")
    axes[2].grid(True)

    figure.tight_layout()

    figure.savefig(
        METRICS_DIR / "validation_metrics.png",
        dpi=150,
    )

    plt.close(figure)


def main():
    METRICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("使用设备:", device)

    checkpoint_paths = sorted(
        CHECKPOINT_DIR.glob("unet_epoch_*.pth"),
        key=get_epoch,
    )

    if not checkpoint_paths:
        raise FileNotFoundError(
            f"没有找到模型文件: {CHECKPOINT_DIR}"
        )

    val_dataset = PairedImageDataset(
        csv_path=METADATA_DIR / "val.csv",
        data_root=DATA_ROOT,
        mode="val",
        image_size=256,
        normalize=True,
        return_meta=False,
    )

    test_dataset = PairedImageDataset(
        csv_path=METADATA_DIR / "test.csv",
        data_root=DATA_ROOT,
        mode="test",
        image_size=256,
        normalize=True,
        return_meta=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    validation_rows = []

    for checkpoint_path in checkpoint_paths:
        epoch = get_epoch(checkpoint_path)

        print(
            f"评估验证集: epoch {epoch}"
        )

        metrics = evaluate_checkpoint(
            checkpoint_path,
            val_loader,
            device,
        )

        validation_rows.append(
            {
                "epoch": epoch,
                "checkpoint": checkpoint_path.name,
                "samples": metrics["samples"],
                "l1": f"{metrics['l1']:.8f}",
                "psnr": f"{metrics['psnr']:.8f}",
                "ssim": f"{metrics['ssim']:.8f}",
            }
        )

    save_csv(
        METRICS_DIR / "validation_metrics.csv",
        validation_rows,
        [
            "epoch",
            "checkpoint",
            "samples",
            "l1",
            "psnr",
            "ssim",
        ],
    )

    plot_validation_metrics(validation_rows)

    best_row = max(
        validation_rows,
        key=lambda row: float(row["ssim"]),
    )

    final_row = validation_rows[-1]

    test_checkpoint_names = []

    for row in [best_row, final_row]:
        if row["checkpoint"] not in test_checkpoint_names:
            test_checkpoint_names.append(row["checkpoint"])

    test_rows = []

    for checkpoint_name in test_checkpoint_names:
        checkpoint_path = CHECKPOINT_DIR / checkpoint_name
        epoch = get_epoch(checkpoint_path)

        print(
            f"评估测试集: epoch {epoch}"
        )

        metrics = evaluate_checkpoint(
            checkpoint_path,
            test_loader,
            device,
        )

        selection = "best_val_ssim"

        if checkpoint_name == final_row["checkpoint"]:
            selection = "final"

        test_rows.append(
            {
                "selection": selection,
                "epoch": epoch,
                "checkpoint": checkpoint_name,
                "samples": metrics["samples"],
                "l1": f"{metrics['l1']:.8f}",
                "psnr": f"{metrics['psnr']:.8f}",
                "ssim": f"{metrics['ssim']:.8f}",
            }
        )

    save_csv(
        METRICS_DIR / "test_metrics.csv",
        test_rows,
        [
            "selection",
            "epoch",
            "checkpoint",
            "samples",
            "l1",
            "psnr",
            "ssim",
        ],
    )

    print("验证集指标:", METRICS_DIR / "validation_metrics.csv")
    print("测试集指标:", METRICS_DIR / "test_metrics.csv")
    print("指标曲线:", METRICS_DIR / "validation_metrics.png")
    print("验证集 SSIM 最佳模型:", best_row["checkpoint"])


if __name__ == "__main__":
    main()
    