from __future__ import annotations

"""
整体处理逻辑：
- train 模式：先对 input / output 使用完全相同的边缘填充，再做同步的随机裁剪，得到 256×256
- val/test 模式：保持比例缩放（短边缩放到 image_size），再边缘填充到 256×256
- input 和 output 始终使用完全相同的空间变换（保证像素级对齐）
- 原始图片不会被修改（只读取，不写入）
- 不依赖 torchvision（只使用 PIL + numpy + torch）
CSV 文件格式（必须包含表头）：
    batch,input_path,output_path,width,height[,split]
"""

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# 允许传入 str 或 pathlib.Path 类型的路径
PathLike = Union[str, Path]


class PairedImageDataset(Dataset):
    """
    成对图像数据集（用于手写去除任务）。

    CSV 字段：
        batch,input_path,output_path,width,height[,split]

    每个样本返回：
        input_tensor:  [3, image_size, image_size]
        target_tensor: [3, image_size, image_size]
    """

    def __init__(
        self,
        csv_path: PathLike,
        data_root: PathLike = ".",
        mode: str = "train",
        image_size: int = 256,
        normalize: bool = True,
        return_meta: bool = False,
    ) -> None:
        """
        参数：
            csv_path:   标注 CSV 文件路径
            data_root:  CSV 中相对路径的根目录
            mode:       "train" / "val" / "test"，决定空间处理方式
            image_size: 输出图像边长（方形），如 256
            normalize:  是否将像素从 [0,255] 缩放到 [0,1]
            return_meta: 是否额外返回元数据（路径、原始尺寸等）
        """
        # ---------- 参数校验 ----------
        if mode not in {"train", "val", "test"}:
            raise ValueError("mode must be one of: train, val, test")

        if image_size <= 0:
            raise ValueError("image_size must be positive")

        # ---------- 保存配置 ----------
        # 将路径统一转为绝对路径，避免后续因为工作目录变化而出错
        self.csv_path = Path(csv_path).resolve()
        self.data_root = Path(data_root).resolve()
        self.mode = mode
        self.image_size = image_size
        self.normalize = normalize
        self.return_meta = return_meta

        # CSV 文件必须存在
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        # ---------- 加载 CSV 并建立样本列表 ----------
        self.samples: List[Dict[str, Any]] = []
        self._load_csv()

    def _load_csv(self) -> None:
        """读取 CSV 文件，解析每一行，并校验文件和路径有效性。"""
        # CSV 至少需要包含这两个字段
        required_columns = {"input_path", "output_path"}

        # 使用 utf-8-sig 以兼容 Excel 导出的带 BOM 文件
        with self.csv_path.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            reader = csv.DictReader(file)

            # 必须有表头，否则无法按列名解析
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {self.csv_path}")

            # 检查必需列是否齐全
            missing_columns = required_columns - set(reader.fieldnames)
            if missing_columns:
                raise ValueError(
                    f"CSV is missing columns: {sorted(missing_columns)}"
                )

            # start=2 表示 CSV 真实行号（含表头）从第 2 行开始
            for line_number, row in enumerate(reader, start=2):
                input_value = (row.get("input_path") or "").strip()
                output_value = (row.get("output_path") or "").strip()

                # 路径不能为空
                if not input_value or not output_value:
                    raise ValueError(
                        f"Empty input_path or output_path at CSV line "
                        f"{line_number}"
                    )

                # 解析为绝对路径（相对路径基于 data_root）
                input_path = self._resolve_path(input_value)
                output_path = self._resolve_path(output_value)

                # 图片必须存在，尽早失败而不是在训练中途报错
                if not input_path.exists():
                    raise FileNotFoundError(
                        f"Input image not found: {input_path}"
                    )

                if not output_path.exists():
                    raise FileNotFoundError(
                        f"Output image not found: {output_path}"
                    )

                # 保存样本信息；保留相对路径用于日志/调试
                self.samples.append(
                    {
                        "batch": row.get("batch", ""),
                        "input_path": input_path,
                        "output_path": output_path,
                        "input_relative_path": input_value,
                        "output_relative_path": output_value,
                    }
                )

        # 空数据集直接报错，避免训练时出现莫名奇妙的问题
        if not self.samples:
            raise ValueError(f"CSV contains no samples: {self.csv_path}")

    def _resolve_path(self, path_value: str) -> Path:
        """
        将 CSV 中的路径解析为绝对路径：
        - 若已是绝对路径，直接使用
        - 若是相对路径，则基于 data_root 拼接
        """
        path = Path(path_value)

        if path.is_absolute():
            return path.resolve()

        return (self.data_root / path).resolve()

    @staticmethod
    def _load_rgb(path: Path) -> Image.Image:
        """
        读取图片并强制转换为 RGB 三通道。
        使用 with 上下文管理器保证文件句柄及时释放。
        """
        with Image.open(path) as image:
            return image.convert("RGB")

    @staticmethod
    def _image_to_array(image: Image.Image) -> np.ndarray:
        """PIL Image -> numpy uint8 数组（H, W, 3）。"""
        return np.asarray(image, dtype=np.uint8)

    @staticmethod
    def _pad_with_edge(
        array: np.ndarray,
        target_height: int,
        target_width: int,
    ) -> np.ndarray:
        """
        使用边缘复制（edge padding）将数组填充到指定大小：
        - 若某一边已经 >= target，则不填充该方向
        - 填充量在两侧均匀分配（多出的 1 像素放在右侧/底部）
        - 这有助于在裁剪时避免出现纯黑边
        """
        height, width = array.shape[:2]

        # 需要填充的像素数（不会为负）
        pad_height = max(0, target_height - height)
        pad_width = max(0, target_width - width)

        # 上下左右平均分配
        top = pad_height // 2
        bottom = pad_height - top
        left = pad_width // 2
        right = pad_width - left

        # 已经满足尺寸，直接返回
        if pad_height == 0 and pad_width == 0:
            return array

        # mode="edge"：用边界像素复制填充；最后一个轴（通道）不填充
        return np.pad(
            array,
            (
                (top, bottom),
                (left, right),
                (0, 0),
            ),
            mode="edge",
        )

    def _random_crop_pair(
        self,
        input_array: np.ndarray,
        target_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        train 模式的空间处理：
        1. 若原图小于 image_size，先边缘填充到至少 image_size
        2. 在填充后的图上做同步的随机裁剪，得到 image_size × image_size

        注意：input 和 target 使用同一个 (top, left)，保证严格对齐。
        """
        height, width = input_array.shape[:2]

        # 填充后的目标尺寸：保证每个方向都 >= image_size
        padded_height = max(height, self.image_size)
        padded_width = max(width, self.image_size)

        input_array = self._pad_with_edge(
            input_array,
            padded_height,
            padded_width,
        )

        target_array = self._pad_with_edge(
            target_array,
            padded_height,
            padded_width,
        )

        # 可裁剪的最大偏移量
        max_top = padded_height - self.image_size
        max_left = padded_width - self.image_size

        # 随机选择裁剪起点；这里用 torch.randint 以便和 PyTorch 随机种子对齐
        if max_top > 0:
            top = int(torch.randint(0, max_top + 1, size=(1,)).item())
        else:
            top = 0

        if max_left > 0:
            left = int(torch.randint(0, max_left + 1, size=(1,)).item())
        else:
            left = 0

        bottom = top + self.image_size
        right = left + self.image_size

        # input/target 共用同一裁剪窗口，确保像素对齐
        input_crop = input_array[top:bottom, left:right]
        target_crop = target_array[top:bottom, left:right]

        return input_crop, target_crop

    def _resize_and_pad_pair(
        self,
        input_array: np.ndarray,
        target_array: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        val/test 模式的空间处理：
        1. 计算统一缩放比 scale = min(image_size/w, image_size/h)，
           保证缩放后仍能完整放进 image_size × image_size
        2. 对 input 和 target 使用同一 scale 缩放（保持比例）
        3. 再用边缘填充补到 image_size × image_size
        """
        height, width = input_array.shape[:2]

        # 取长边的缩放比例作为统一比例（短边缩得更小）
        scale = min(
            self.image_size / width,
            self.image_size / height,
        )

        # 计算缩放后尺寸，至少 1 像素，且不超过 image_size
        new_width = max(
            1,
            min(self.image_size, int(round(width * scale))),
        )
        new_height = max(
            1,
            min(self.image_size, int(round(height * scale))),
        )

        # 兼容不同 Pillow 版本的 Resampling 枚举
        if hasattr(Image, "Resampling"):
            resampling = Image.Resampling.BILINEAR
        else:
            resampling = Image.BILINEAR

        # input 与 target 使用完全相同的缩放参数
        input_image = Image.fromarray(input_array).resize(
            (new_width, new_height),
            resampling,
        )

        target_image = Image.fromarray(target_array).resize(
            (new_width, new_height),
            resampling,
        )

        input_resized = np.asarray(input_image, dtype=np.uint8)
        target_resized = np.asarray(target_image, dtype=np.uint8)

        # 用边缘填充补齐到 image_size × image_size
        input_resized = self._pad_with_edge(
            input_resized,
            self.image_size,
            self.image_size,
        )

        target_resized = self._pad_with_edge(
            target_resized,
            self.image_size,
            self.image_size,
        )

        return input_resized, target_resized

    def _to_tensor(self, array: np.ndarray) -> torch.Tensor:
        """
        numpy (H, W, 3) uint8 -> torch.Tensor (3, H, W) float32。
        若 normalize=True，将像素值从 [0,255] 归一化到 [0,1]。
        """
        array = array.astype(np.float32)

        if self.normalize:
            array = array / 255.0

        # transpose 后得到 (3, H, W)；copy() 保证内存连续
        tensor = torch.from_numpy(array.transpose(2, 0, 1).copy())
        return tensor

    def __len__(self) -> int:
        """返回数据集样本数量，DataLoader 依赖此方法。"""
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ):
        """
        获取第 index 个样本。

        返回值：
        - return_meta=False: (input_tensor, target_tensor)
        - return_meta=True:  (input_tensor, target_tensor, metadata)
        """
        sample = self.samples[index]

        # 读取 input 与 target（均转换为 RGB 三通道）
        input_image = self._load_rgb(sample["input_path"])
        target_image = self._load_rgb(sample["output_path"])

        # 尺寸必须严格一致，否则无法做像素级对齐
        if input_image.size != target_image.size:
            raise ValueError(
                "Input and target dimensions do not match: "
                f"{sample['input_relative_path']} "
                f"{input_image.size} vs {target_image.size}"
            )

        # 转成 numpy 数组（H, W, 3）uint8
        input_array = self._image_to_array(input_image)
        target_array = self._image_to_array(target_image)

        # 记录原始尺寸（用于元数据/评估时还原）
        original_width, original_height = input_image.size

        # 根据 mode 应用不同的空间处理
        if self.mode == "train":
            # 训练：随机裁剪（含边缘填充）
            input_array, target_array = self._random_crop_pair(
                input_array,
                target_array,
            )
        else:
            # 验证/测试：保持比例缩放 + 边缘填充，保证可复现
            input_array, target_array = self._resize_and_pad_pair(
                input_array,
                target_array,
            )

        # 转成张量
        input_tensor = self._to_tensor(input_array)
        target_tensor = self._to_tensor(target_array)

        # 不返回元数据时直接给出 (input, target)
        if not self.return_meta:
            return input_tensor, target_tensor

        # 组装元数据
        metadata = {
            "batch": sample["batch"],
            "input_path": str(sample["input_relative_path"]),
            "output_path": str(sample["output_relative_path"]),
            "original_width": original_width,
            "original_height": original_height,
            "mode": self.mode,
        }

        return input_tensor, target_tensor, metadata

