from __future__ import annotations

"""
改进版训练脚本（配合 model_v2.py 和 losses_v2.py）。

与 baseline train.py 的区别：
1. 模型换成 UNetV2（残差块 + 注意力门控）；
2. 训练损失换成 CombinedLoss（L1 + Sobel 边缘损失，可选 VGG 感知损失）；
3. 验证/测试仍使用纯 L1，保证 PSNR/SSIM 与 baseline 实验可直接对比；
4. 输出目录独立为 outputs/experiment_v2_50epoch，不会覆盖 baseline 结果。

其余流程（数据加载、指标计算、checkpoint、曲线、可视化）与 baseline 保持一致。
"""

import csv
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import PairedImageDataset
from losses_v2 import CombinedLoss
from metrics import calculate_batch_metrics
from model_v2 import UNetV2


# ============================================================
# 路径配置
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
METADATA_DIR = os.path.join(PROJECT_ROOT, "data", "metadata")

EXPERIMENT_NAME = "experiment_v2_50epoch"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", EXPERIMENT_NAME)
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
VISUALIZE_DIR = os.path.join(OUTPUT_DIR, "visualize", "samples")

LOG_FILE = os.path.join(OUTPUT_DIR, "loss.csv")
EPOCH_METRICS_FILE = os.path.join(OUTPUT_DIR, "epoch_metrics.csv")
STEP_METRICS_FILE = os.path.join(OUTPUT_DIR, "step_metrics.csv")
TEST_METRICS_FILE = os.path.join(OUTPUT_DIR, "test_metrics.csv")
STEP_METRICS_PLOT = os.path.join(OUTPUT_DIR, "step_metrics.png")
EPOCH_METRICS_PLOT = os.path.join(OUTPUT_DIR, "epoch_metrics.png")
BEST_CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")


# ============================================================
# 训练参数
# ============================================================

IMAGE_SIZE = 256
BATCH_SIZE = 6
NUM_WORKERS = 0

LEARNING_RATE = 0.0001
NUM_EPOCHS = 40

MAX_TRAIN_BATCHES = None
MAX_VAL_BATCHES = None

# 每 20 个 step 记录一次指标
METRICS_INTERVAL = 20

# 组合损失权重。edge_weight 是新增的边缘约束，先给 0.5 保守起步；
# 开启 VGG 感知损失需要 torchvision + 预训练权重，默认关闭。
L1_WEIGHT = 1.0
EDGE_WEIGHT = 0.5
VGG_WEIGHT = 0.1
USE_VGG_LOSS = False


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_directories():
    for path in (OUTPUT_DIR, CHECKPOINT_DIR, VISUALIZE_DIR):
        if not os.path.exists(path):
            os.makedirs(path)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用 GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("使用 CPU 训练")

    return device


def create_dataloaders():
    train_dataset = PairedImageDataset(
        csv_path=os.path.join(METADATA_DIR, "train.csv"),
        data_root=DATA_ROOT,
        mode="train",
        image_size=IMAGE_SIZE,
        normalize=True,
        return_meta=False
    )

    val_dataset = PairedImageDataset(
        csv_path=os.path.join(METADATA_DIR, "val.csv"),
        data_root=DATA_ROOT,
        mode="val",
        image_size=IMAGE_SIZE,
        normalize=True,
        return_meta=False
    )

    test_dataset = PairedImageDataset(
        csv_path=os.path.join(METADATA_DIR, "test.csv"),
        data_root=DATA_ROOT,
        mode="test",
        image_size=IMAGE_SIZE,
        normalize=True,
        return_meta=False
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory
    )

    print("训练集数量:", len(train_dataset))
    print("验证集数量:", len(val_dataset))
    print("测试集数量:", len(test_dataset))
    print("训练 batch 数量:", len(train_loader))
    print("验证 batch 数量:", len(val_loader))
    print("测试 batch 数量:", len(test_loader))

    return train_loader, val_loader, test_loader


def calculate_metrics(predicted_images, target_images):
    """计算一个 batch 的 L1、PSNR、SSIM 和样本数。"""
    result = calculate_batch_metrics(predicted_images, target_images)
    result["count"] = predicted_images.shape[0]
    return result


def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device,
    epoch,
    global_step,
    step_metrics_history
):
    """训练一个 epoch，使用 CombinedLoss，记录各损失分量。"""
    model.train()

    total_loss = 0.0
    batch_count = 0

    start_time = time.time()

    for batch_index, data in enumerate(train_loader):
        input_images, target_images = data

        input_images = input_images.to(device)
        target_images = target_images.to(device)

        optimizer.zero_grad()

        predicted_images = model(input_images)

        loss, components = criterion(
            predicted_images,
            target_images
        )

        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()
        batch_count = batch_count + 1
        global_step = global_step + 1

        if batch_index % 5 == 0:
            print(
                "Epoch {}，Train Batch {}/{}，Loss: {:.6f}（L1: {:.6f}，Edge: {:.6f}，VGG: {:.6f}）".format(
                    epoch,
                    batch_index + 1,
                    len(train_loader),
                    loss.item(),
                    components["l1"],
                    components["edge"],
                    components["vgg"]
                )
            )

        # 每 METRICS_INTERVAL 步记录一次指标，观察训练过程
        if global_step % METRICS_INTERVAL == 0:
            metrics = calculate_metrics(
                predicted_images,
                target_images
            )

            step_metrics_history.append(
                {
                    "step": global_step,
                    "epoch": epoch,
                    "batch": batch_index + 1,
                    "loss": loss.item(),
                    "l1": components["l1"],
                    "edge": components["edge"],
                    "vgg": components["vgg"],
                    "psnr": metrics["psnr"],
                    "ssim": metrics["ssim"]
                }
            )

            print(
                "Step {}，Loss: {:.6f}，PSNR: {:.4f}，SSIM: {:.4f}".format(
                    global_step,
                    loss.item(),
                    metrics["psnr"],
                    metrics["ssim"]
                )
            )

        if MAX_TRAIN_BATCHES is not None:
            if batch_count >= MAX_TRAIN_BATCHES:
                break

    if batch_count == 0:
        raise RuntimeError("训练数据为空，无法计算平均 Loss。")

    average_loss = total_loss / batch_count
    elapsed_time = time.time() - start_time

    print(
        "Epoch {} 训练完成，平均 Loss: {:.6f}，耗时: {:.2f} 秒".format(
            epoch,
            average_loss,
            elapsed_time
        )
    )

    return average_loss, global_step


def validate(model, data_loader, eval_criterion, device, epoch, data_name):
    """
    在验证集或测试集上评估。

    使用纯 L1 作为评估损失，与 baseline 的指标口径一致。
    """
    model.eval()

    total_loss = 0.0
    total_l1 = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    total_count = 0

    start_time = time.time()

    with torch.no_grad():
        for batch_index, data in enumerate(data_loader):
            input_images, target_images = data

            input_images = input_images.to(device)
            target_images = target_images.to(device)

            predicted_images = model(input_images)

            loss = eval_criterion(
                predicted_images,
                target_images
            )

            metrics = calculate_metrics(
                predicted_images,
                target_images
            )

            current_count = metrics["count"]

            total_loss = total_loss + loss.item() * current_count
            total_l1 = total_l1 + metrics["l1"] * current_count
            total_psnr = total_psnr + metrics["psnr"] * current_count
            total_ssim = total_ssim + metrics["ssim"] * current_count
            total_count = total_count + current_count

            if MAX_VAL_BATCHES is not None:
                if batch_index + 1 >= MAX_VAL_BATCHES:
                    break

    if total_count == 0:
        raise RuntimeError(
            "{} 数据为空，无法计算评估指标。".format(data_name)
        )

    result = {
        "loss": total_loss / total_count,
        "l1": total_l1 / total_count,
        "psnr": total_psnr / total_count,
        "ssim": total_ssim / total_count,
        "samples": total_count
    }

    elapsed_time = time.time() - start_time

    print(
        "Epoch {} {}完成，Loss: {:.6f}，PSNR: {:.4f}，SSIM: {:.4f}，耗时: {:.2f} 秒".format(
            epoch,
            data_name,
            result["loss"],
            result["psnr"],
            result["ssim"],
            elapsed_time
        )
    )

    return result


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    validation_metrics,
    device,
    checkpoint_path
):
    checkpoint = {
        "epoch": epoch,
        "model": "UNetV2",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": validation_metrics["loss"],
        "val_psnr": validation_metrics["psnr"],
        "val_ssim": validation_metrics["ssim"],
        "device": str(device)
    }

    torch.save(checkpoint, checkpoint_path)
    print("模型已保存:", checkpoint_path)


def save_epoch_metrics(history):
    with open(EPOCH_METRICS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "train_loss", "val_loss", "val_psnr", "val_ssim"]
        )
        writer.writeheader()
        writer.writerows(history)

    print("Epoch 指标已保存:", EPOCH_METRICS_FILE)


def save_step_metrics(metrics_history):
    if len(metrics_history) == 0:
        print("没有达到指标记录间隔，未生成 step 指标文件。")
        return

    with open(STEP_METRICS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["step", "epoch", "batch", "loss", "l1", "edge", "vgg", "psnr", "ssim"]
        )
        writer.writeheader()
        writer.writerows(metrics_history)

    print("Step 指标已保存:", STEP_METRICS_FILE)


def save_test_metrics(test_rows):
    with open(TEST_METRICS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["model", "epoch", "samples", "loss", "psnr", "ssim"]
        )
        writer.writeheader()
        writer.writerows(test_rows)

    print("测试集指标已保存:", TEST_METRICS_FILE)


def plot_metrics(step_metrics_history, epoch_history):
    if len(step_metrics_history) > 0:
        steps = [item["step"] for item in step_metrics_history]
        step_losses = [item["loss"] for item in step_metrics_history]
        step_psnr = [item["psnr"] for item in step_metrics_history]
        step_ssim = [item["ssim"] for item in step_metrics_history]

        figure, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].plot(steps, step_losses)
        axes[0].set_title("Training Loss (combined)")
        axes[0].set_xlabel("Step")
        axes[0].set_ylabel("Loss")
        axes[0].grid(True)

        axes[1].plot(steps, step_psnr)
        axes[1].set_title("Training PSNR")
        axes[1].set_xlabel("Step")
        axes[1].set_ylabel("PSNR")
        axes[1].grid(True)

        axes[2].plot(steps, step_ssim)
        axes[2].set_title("Training SSIM")
        axes[2].set_xlabel("Step")
        axes[2].set_ylabel("SSIM")
        axes[2].grid(True)

        figure.tight_layout()
        figure.savefig(STEP_METRICS_PLOT, dpi=150)
        plt.close(figure)

        print("Step 指标曲线已保存:", STEP_METRICS_PLOT)

    if len(epoch_history) > 0:
        epochs = [item["epoch"] for item in epoch_history]
        train_losses = [item["train_loss"] for item in epoch_history]
        val_losses = [item["val_loss"] for item in epoch_history]
        val_psnr = [item["val_psnr"] for item in epoch_history]
        val_ssim = [item["val_ssim"] for item in epoch_history]

        figure, axes = plt.subplots(1, 3, figsize=(16, 5))

        axes[0].plot(epochs, train_losses, label="train")
        axes[0].plot(epochs, val_losses, label="validation")
        axes[0].set_title("Loss (eval = L1)")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(epochs, val_psnr)
        axes[1].set_title("Validation PSNR")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("PSNR")
        axes[1].grid(True)

        axes[2].plot(epochs, val_ssim)
        axes[2].set_title("Validation SSIM")
        axes[2].set_xlabel("Epoch")
        axes[2].set_ylabel("SSIM")
        axes[2].grid(True)

        figure.tight_layout()
        figure.savefig(EPOCH_METRICS_PLOT, dpi=150)
        plt.close(figure)

        print("Epoch 指标曲线已保存:", EPOCH_METRICS_PLOT)


def tensor_to_image(tensor):
    """[C,H,W] 且范围 [0,1] 的张量转为 [H,W,C] 的 uint8 数组。"""
    array = tensor.clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    return (array * 255.0).astype(np.uint8)


def save_visualize_samples(model, loader, device, num_samples=5):
    """训练结束后保存几张 input / prediction / GT 的三联对比图。"""
    from PIL import Image

    model.eval()
    saved = 0

    with torch.no_grad():
        for data in loader:
            input_images, target_images = data
            input_images = input_images.to(device)

            predicted_images = model(input_images)

            for index in range(input_images.shape[0]):
                if saved >= num_samples:
                    return

                input_image = tensor_to_image(input_images[index])
                predicted_image = tensor_to_image(predicted_images[index])
                target_image = tensor_to_image(target_images[index])

                combined = np.concatenate(
                    [input_image, predicted_image, target_image],
                    axis=1
                )

                save_path = os.path.join(
                    VISUALIZE_DIR,
                    "sample_{}.png".format(saved + 1)
                )

                Image.fromarray(combined).save(save_path)
                saved = saved + 1


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def evaluate_test_checkpoints(
    model,
    test_loader,
    eval_criterion,
    device,
    best_checkpoint_path,
    final_checkpoint_path
):
    """使用最佳验证模型和最终模型自动评估测试集。"""
    test_rows = []
    checkpoint_paths = [best_checkpoint_path]
    checkpoint_names = ["best_validation_ssim"]

    if final_checkpoint_path != best_checkpoint_path:
        checkpoint_paths.append(final_checkpoint_path)
        checkpoint_names.append("final_epoch")

    for index in range(len(checkpoint_paths)):
        checkpoint = load_checkpoint(model, checkpoint_paths[index], device)
        epoch = checkpoint["epoch"]

        metrics = validate(
            model=model,
            data_loader=test_loader,
            eval_criterion=eval_criterion,
            device=device,
            epoch=epoch,
            data_name="测试集"
        )

        test_rows.append(
            {
                "model": checkpoint_names[index],
                "epoch": epoch,
                "samples": metrics["samples"],
                "loss": metrics["loss"],
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"]
            }
        )

    save_test_metrics(test_rows)


def main():
    set_random_seed(42)
    create_directories()

    device = get_device()

    train_loader, val_loader, test_loader = create_dataloaders()

    model = UNetV2(
        in_channels=3,
        out_channels=3,
        base_channels=32
    )
    model = model.to(device)

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("UNetV2 可训练参数量:", parameter_count)

    # 训练用组合损失，评估用纯 L1（与 baseline 口径一致，方便对比）
    criterion = CombinedLoss(
        l1_weight=L1_WEIGHT,
        edge_weight=EDGE_WEIGHT,
        vgg_weight=VGG_WEIGHT,
        use_vgg=USE_VGG_LOSS,
        in_channels=3,
        device=device
    )

    eval_criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    step_metrics_history = []
    global_step = 0
    best_val_ssim = -1.0
    final_checkpoint_path = ""

    print("开始训练（改进版）")
    print("实验名称:", EXPERIMENT_NAME)
    print("训练 epoch 数:", NUM_EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Image size:", IMAGE_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("损失权重: L1={}, Edge={}, VGG={}（use_vgg={}）".format(
        L1_WEIGHT, EDGE_WEIGHT, VGG_WEIGHT, USE_VGG_LOSS
    ))
    print("指标记录间隔:", METRICS_INTERVAL, "steps")

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            global_step=global_step,
            step_metrics_history=step_metrics_history
        )

        validation_metrics = validate(
            model=model,
            data_loader=val_loader,
            eval_criterion=eval_criterion,
            device=device,
            epoch=epoch,
            data_name="验证集"
        )

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation_metrics["loss"],
            "val_psnr": validation_metrics["psnr"],
            "val_ssim": validation_metrics["ssim"]
        }
        history.append(epoch_record)

        final_checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            "unetv2_epoch_{}.pth".format(epoch)
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            validation_metrics=validation_metrics,
            device=device,
            checkpoint_path=final_checkpoint_path
        )

        # 按验证集 SSIM 选 best model，与 baseline 的选择标准一致
        if validation_metrics["ssim"] > best_val_ssim:
            best_val_ssim = validation_metrics["ssim"]

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                validation_metrics=validation_metrics,
                device=device,
                checkpoint_path=BEST_CHECKPOINT_PATH
            )

            print(
                "当前最佳模型已更新，验证集 SSIM: {:.6f}".format(best_val_ssim)
            )

        # 每个 epoch 落盘一次，避免中途停止丢失记录
        save_epoch_metrics(history)
        save_step_metrics(step_metrics_history)
        plot_metrics(step_metrics_history, history)

    # 训练结束后评估测试集并保存可视化
    evaluate_test_checkpoints(
        model=model,
        test_loader=test_loader,
        eval_criterion=eval_criterion,
        device=device,
        best_checkpoint_path=BEST_CHECKPOINT_PATH,
        final_checkpoint_path=final_checkpoint_path
    )

    # 可视化用 best model 而不是最后的模型：
    # 训练后期可能过拟合，验证 SSIM 下降，best model 才是代表实验结果的权重
    load_checkpoint(model, BEST_CHECKPOINT_PATH, device)
    print("可视化已切换到 best model")

    save_visualize_samples(model, test_loader, device, num_samples=5)

    print("训练和测试评估完成")


if __name__ == "__main__":
    main()
