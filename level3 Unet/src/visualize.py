import os

import numpy as np
import torch
from PIL import Image

from dataset import PairedImageDataset
from model import UNet


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_ROOT = os.path.join(PROJECT_ROOT, "data")

CSV_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "metadata",
    "val.csv"
)

CHECKPOINT_PATH = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "experiment_50epoch",
    "checkpoints",
    "unet_epoch_30.pth"
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "outputs",
    "experiment_50epoch",
    "visualize",
    "samples"
)

IMAGE_SIZE = 256
NUM_SAMPLES = 5


def tensor_to_image(tensor):
    tensor = tensor.detach().cpu()
    tensor = torch.clamp(tensor, 0.0, 1.0)

    array = tensor.numpy()
    array = np.transpose(array, (1, 2, 0))
    array = array * 255.0
    array = array.astype(np.uint8)

    return array


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("使用设备:", device)

    dataset = PairedImageDataset(
        csv_path=CSV_PATH,
        data_root=DATA_ROOT,
        mode="val",
        image_size=IMAGE_SIZE,
        normalize=True,
        return_meta=True
    )

    model = UNet(
        in_channels=3,
        out_channels=3,
        base_channels=32
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    sample_count = min(NUM_SAMPLES, len(dataset))

    for index in range(sample_count):
        input_tensor, target_tensor, metadata = dataset[index]

        input_batch = input_tensor.unsqueeze(0)
        input_batch = input_batch.to(device)

        with torch.no_grad():
            predicted_tensor = model(input_batch)

        predicted_tensor = predicted_tensor[0]

        input_image = tensor_to_image(input_tensor)
        predicted_image = tensor_to_image(predicted_tensor)
        target_image = tensor_to_image(target_tensor)

        combined_image = np.concatenate(
            [
                input_image,
                predicted_image,
                target_image
            ],
            axis=1
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            "sample_{}.png".format(index + 1)
        )

        Image.fromarray(combined_image).save(
            output_path
        )

        print(
            "样本 {} 已保存: {}".format(
                index + 1,
                output_path
            )
        )

        print(
            "预测图像像素范围: {:.4f} - {:.4f}".format(
                predicted_tensor.min().item(),
                predicted_tensor.max().item()
            )
        )

        print(
            "预测图像平均像素值: {:.4f}".format(
                predicted_tensor.mean().item()
            )
        )
        print(
            "输入图像平均像素值: {:.4f}".format(
                input_tensor.mean().item()
            )
        )

        print(
            "GT 图像平均像素值: {:.4f}".format(
                target_tensor.mean().item()
            )
        )

        print(
            "GT 图像像素范围: {:.4f} - {:.4f}".format(
                target_tensor.min().item(),
                target_tensor.max().item()
            )
        )

        print(
            "输入图片路径: {}".format(
                metadata["input_path"]
            )
        )


if __name__ == "__main__":
    main()
