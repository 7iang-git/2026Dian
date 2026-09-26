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
from skimage.metrics import structural_similarity
from torch.utils.data import DataLoader

from dataset import PairedImageDataset
from model import UNet


# ============================================================
# 项目路径配置
# ============================================================

# 当前文件位于：项目根目录/src/train.py。
# 先取得 src 目录，再向上一级得到项目根目录。
# 同时兼容 Windows 和 Linux
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# 原始图片和 metadata 所在的 data 目录。
# CSV 中的图片路径是相对路径，例如：
# 20250211/dataset/input/xxxx.jpg
DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data"
)

# train.csv、val.csv、test.csv 所在的目录。
METADATA_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata"
)

# 所有训练结果统一保存到 outputs 目录。
# 如果需要进行新的实验，可以把 baseline_epoch1 改成新的实验名称。
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "experiment_50epoch"
)

CHECKPOINT_DIR = os.path.join(
    OUTPUT_DIR,
    "checkpoints"
)

LOG_FILE = os.path.join(
    OUTPUT_DIR,
    "loss.csv"
)

STEP_METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "step_metrics.csv"
)

EPOCH_METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "epoch_metrics.csv"
)

TEST_METRICS_FILE = os.path.join(
    OUTPUT_DIR,
    "test_metrics.csv"
)

STEP_METRICS_PLOT = os.path.join(
    OUTPUT_DIR,
    "step_metrics.png"
)

EPOCH_METRICS_PLOT = os.path.join(
    OUTPUT_DIR,
    "epoch_metrics.png"
)

BEST_CHECKPOINT_PATH = os.path.join(
    CHECKPOINT_DIR,
    "best_model.pth"
)


# ============================================================
# 训练参数
# ============================================================

IMAGE_SIZE = 256
BATCH_SIZE = 6
NUM_WORKERS = 0

LEARNING_RATE = 0.0001

# 正式训练可以先使用 50 个 epoch。
# 之前的 1 个 epoch 只用于检查流程是否正常。
NUM_EPOCHS = 40

# 设置为 None 表示使用完整数据集。
# 调试代码时可以临时设置为整数，例如 10。
MAX_TRAIN_BATCHES = None
MAX_VAL_BATCHES = None

# 每隔多少个训练 step 记录一次 Loss、PSNR 和 SSIM。
METRICS_INTERVAL = 20


def set_random_seed(seed):
    """
    设置随机种子，使实验尽量可以重复。

    训练集在 train 模式下会进行随机裁剪，因此固定随机种子
    可以让不同实验更容易比较。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_directories():
    """创建训练结果目录。目录已经存在时不重复创建。"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)


def get_device():
    """优先使用 CUDA GPU，否则使用 CPU。"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("使用 GPU 训练")
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print("使用 CPU 训练")

    return device


def create_dataloaders():
    """
    创建训练集、验证集和测试集的数据加载器。

    训练集使用 train 模式，会进行随机裁剪。
    验证集和测试集使用确定性的缩放与填充，不进行随机裁剪。
    """
    train_csv = os.path.join(
        METADATA_DIR,
        "train.csv"
    )

    val_csv = os.path.join(
        METADATA_DIR,
        "val.csv"
    )

    test_csv = os.path.join(
        METADATA_DIR,
        "test.csv"
    )

    train_dataset = PairedImageDataset(
        csv_path=train_csv,
        data_root=DATA_ROOT,
        mode="train",
        image_size=IMAGE_SIZE,
        normalize=True,
        return_meta=False
    )

    val_dataset = PairedImageDataset(
        csv_path=val_csv,
        data_root=DATA_ROOT,
        mode="val",
        image_size=IMAGE_SIZE,
        normalize=True,
        return_meta=False
    )

    test_dataset = PairedImageDataset(
        csv_path=test_csv,
        data_root=DATA_ROOT,
        mode="test",
        image_size=IMAGE_SIZE,
        normalize=True,
        return_meta=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    print("训练集数量:", len(train_dataset))
    print("验证集数量:", len(val_dataset))
    print("测试集数量:", len(test_dataset))
    print("训练 batch 数量:", len(train_loader))
    print("验证 batch 数量:", len(val_loader))
    print("测试 batch 数量:", len(test_loader))

    return train_loader, val_loader, test_loader


def calculate_metrics(predicted_images, target_images):
    """
    计算一个 batch 的 L1、PSNR 和 SSIM。

    输入图像已经被 Dataset 归一化到 [0, 1]，因此 PSNR 的
    data range 使用 1.0。

    PSNR 越大越好，SSIM 越接近 1 越好，L1 越小越好。
    """
    predicted_images = torch.clamp(
        predicted_images.detach(),
        0.0,
        1.0
    )

    target_images = torch.clamp(
        target_images.detach(),
        0.0,
        1.0
    )

    batch_size = predicted_images.shape[0]

    # 计算每张图像的平均绝对误差。
    l1_values = torch.abs(
        predicted_images - target_images
    ).reshape(batch_size, -1).mean(dim=1)

    # 计算每张图像的均方误差。
    mse_values = (
        (predicted_images - target_images).pow(2)
        .reshape(batch_size, -1)
        .mean(dim=1)
    )

    # PSNR = 10 * log10(MAX^2 / MSE)。
    # 当前像素已经归一化到 [0, 1]，所以 MAX=1。
    psnr_values = 10.0 * torch.log10(
        1.0 / torch.clamp(mse_values, min=1e-12)
    )

    # skimage 的 SSIM 需要接收 H x W x C 格式的 numpy 数组。
    predicted_numpy = (
        predicted_images.cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )

    target_numpy = (
        target_images.cpu()
        .permute(0, 2, 3, 1)
        .numpy()
    )

    ssim_values = []

    for index in range(batch_size):
        try:
            ssim_value = structural_similarity(
                target_numpy[index],
                predicted_numpy[index],
                channel_axis=-1,
                data_range=1.0
            )
        except TypeError:
            # 兼容旧版本 scikit-image。
            ssim_value = structural_similarity(
                target_numpy[index],
                predicted_numpy[index],
                multichannel=True,
                data_range=1.0
            )

        ssim_values.append(ssim_value)

    return {
        "count": batch_size,
        "l1": l1_values.mean().item(),
        "psnr": psnr_values.mean().item(),
        "ssim": float(np.mean(ssim_values))
    }


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
    """
    训练一个 epoch。

    global_step 用于跨 epoch 连续记录训练步数。
    例如 METRICS_INTERVAL=20 时，会在第 20、40、60... 步记录指标。
    """
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

        loss = criterion(
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
                "Epoch {}，Train Batch {}/{}，Loss: {:.6f}".format(
                    epoch,
                    batch_index + 1,
                    len(train_loader),
                    loss.item()
                )
            )

        # 每隔固定 step 计算一次 PSNR 和 SSIM。
        # 这些指标只使用当前 batch，主要用于观察训练过程。
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


def validate(
    model,
    data_loader,
    criterion,
    device,
    epoch,
    data_name
):
    """
    在验证集或测试集上评估模型。

    这里使用完整数据集的样本数量进行加权平均，避免最后一个
    batch 数量较少而影响整体指标。
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

            loss = criterion(
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


def create_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    validation_metrics,
    device
):
    """整理 checkpoint 内容，避免重复编写字典。"""
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": validation_metrics["loss"],
        "val_psnr": validation_metrics["psnr"],
        "val_ssim": validation_metrics["ssim"],
        "device": str(device)
    }


def save_checkpoint(
    model,
    optimizer,
    epoch,
    train_loss,
    validation_metrics,
    device,
    checkpoint_path
):
    """保存一个模型 checkpoint。"""
    checkpoint = create_checkpoint(
        model,
        optimizer,
        epoch,
        train_loss,
        validation_metrics,
        device
    )

    torch.save(
        checkpoint,
        checkpoint_path
    )

    print("模型已保存:", checkpoint_path)


def save_loss_log(history):
    """保存每个 epoch 的 Loss、PSNR 和 SSIM。"""
    with open(
        LOG_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "epoch",
                "train_loss",
                "val_loss",
                "val_psnr",
                "val_ssim"
            ]
        )

        for item in history:
            writer.writerow(
                [
                    item["epoch"],
                    item["train_loss"],
                    item["val_loss"],
                    item["val_psnr"],
                    item["val_ssim"]
                ]
            )

    print("Epoch 指标已保存:", LOG_FILE)


def save_step_metrics(metrics_history):
    """保存每隔 METRICS_INTERVAL 个 step 记录的训练指标。"""
    if len(metrics_history) == 0:
        print("没有达到指标记录间隔，未生成 step 指标文件。")
        return

    with open(
        STEP_METRICS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "step",
                "epoch",
                "batch",
                "loss",
                "psnr",
                "ssim"
            ]
        )

        writer.writeheader()
        writer.writerows(metrics_history)

    print("Step 指标已保存:", STEP_METRICS_FILE)


def save_epoch_metrics(history):
    """单独保存验证集的 epoch 指标，方便后续分析。"""
    with open(
        EPOCH_METRICS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_loss",
                "val_psnr",
                "val_ssim"
            ]
        )

        writer.writeheader()
        writer.writerows(history)

    print("Epoch 指标已保存:", EPOCH_METRICS_FILE)


def save_test_metrics(test_rows):
    """保存测试集最终评估结果。"""
    with open(
        TEST_METRICS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model",
                "epoch",
                "samples",
                "loss",
                "psnr",
                "ssim"
            ]
        )

        writer.writeheader()
        writer.writerows(test_rows)

    print("测试集指标已保存:", TEST_METRICS_FILE)


def plot_metrics(
    step_metrics_history,
    epoch_history
):
    """
    绘制两张曲线图：
    1. step_metrics.png：每 20 steps 的训练指标
    2. epoch_metrics.png：每个 epoch 的验证指标
    """
    if len(step_metrics_history) > 0:
        steps = []
        step_losses = []
        step_psnr = []
        step_ssim = []

        for item in step_metrics_history:
            steps.append(item["step"])
            step_losses.append(item["loss"])
            step_psnr.append(item["psnr"])
            step_ssim.append(item["ssim"])

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(16, 5)
        )

        axes[0].plot(steps, step_losses)
        axes[0].set_title("Training Loss")
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
        epochs = []
        train_losses = []
        val_losses = []
        val_psnr = []
        val_ssim = []

        for item in epoch_history:
            epochs.append(item["epoch"])
            train_losses.append(item["train_loss"])
            val_losses.append(item["val_loss"])
            val_psnr.append(item["val_psnr"])
            val_ssim.append(item["val_ssim"])

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(16, 5)
        )

        axes[0].plot(epochs, train_losses, label="train")
        axes[0].plot(epochs, val_losses, label="validation")
        axes[0].set_title("Loss")
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


def load_checkpoint(model, checkpoint_path, device):
    """读取 checkpoint 中的模型参数。"""
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return checkpoint


def evaluate_test_checkpoints(
    model,
    test_loader,
    criterion,
    device,
    best_checkpoint_path,
    final_checkpoint_path
):
    """使用最佳验证模型和最终模型自动评估测试集。"""
    test_rows = []
    checkpoint_paths = []
    checkpoint_names = []

    checkpoint_paths.append(best_checkpoint_path)
    checkpoint_names.append("best_validation_ssim")

    if final_checkpoint_path != best_checkpoint_path:
        checkpoint_paths.append(final_checkpoint_path)
        checkpoint_names.append("final_epoch")

    for index in range(len(checkpoint_paths)):
        checkpoint_path = checkpoint_paths[index]
        model_name = checkpoint_names[index]

        checkpoint = load_checkpoint(
            model,
            checkpoint_path,
            device
        )

        epoch = checkpoint["epoch"]

        metrics = validate(
            model=model,
            data_loader=test_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            data_name="测试集"
        )

        test_rows.append(
            {
                "model": model_name,
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

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32
    )

    model = model.to(device)

    # L1 对图像到图像任务通常比单纯 MSE 更不容易产生过度模糊。
    criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    history = []
    step_metrics_history = []
    global_step = 0
    best_val_ssim = -1.0
    final_checkpoint_path = ""

    print("开始训练")
    print("训练 epoch 数:", NUM_EPOCHS)
    print("Batch size:", BATCH_SIZE)
    print("Image size:", IMAGE_SIZE)
    print("Learning rate:", LEARNING_RATE)
    print("指标记录间隔:", METRICS_INTERVAL, "steps")
    print("数据根目录:", DATA_ROOT)
    print("Metadata 目录:", METADATA_DIR)

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
            criterion=criterion,
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
            "unet_epoch_{}.pth".format(epoch)
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

        # 使用验证集 SSIM 选择最佳模型。
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
                "当前最佳模型已更新，验证集 SSIM: {:.6f}".format(
                    best_val_ssim
                )
            )

        # 每个 epoch 保存一次，避免程序中途停止时丢失全部记录。
        save_loss_log(history)
        save_epoch_metrics(history)
        save_step_metrics(step_metrics_history)
        plot_metrics(
            step_metrics_history,
            history
        )

    # 训练结束后自动使用最佳模型和最终模型评估测试集。
    evaluate_test_checkpoints(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        best_checkpoint_path=BEST_CHECKPOINT_PATH,
        final_checkpoint_path=final_checkpoint_path
    )

    print("训练和测试评估完成")


if __name__ == "__main__":
    main()
