import os
import torch
import numpy as np
from pathlib import Path

class MemmapDataset:
    """
    Quản lý dữ liệu nhị phân (.bin uint16) dùng chung cho cả LSTM và GPT.
    Sử dụng np.memmap để đọc tức thì từ ổ đĩa mà không tốn RAM.
    """
    def __init__(self, bin_path: str, block_size: int = 256):
        self.bin_path = Path(bin_path)
        if not self.bin_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file nhị phân: {bin_path}")
        
        self.block_size = block_size
        # Đọc dữ liệu dạng mảng số nguyên 16-bit
        self.data = np.memmap(self.bin_path, dtype=np.uint16, mode='r')
        self.total_tokens = len(self.data)
        print(f"[Dataset] Đã nạp {self.bin_path.name}: {self.total_tokens:,} tokens")

    def __len__(self):
        return max(0, self.total_tokens - self.block_size)

    def get_batch(self, batch_size: int, device: str = 'cpu'):
        """
        Lấy ngẫu nhiên một batch gồm (X, Y):
        X: chuỗi token độ dài block_size
        Y: chuỗi token dịch phải 1 bước (Next-token prediction)
        """
        max_idx = len(self)
        if max_idx <= 0:
            raise ValueError(f"Dữ liệu ({self.total_tokens} tokens) quá ngắn so với block_size ({self.block_size})")

        # Chọn ngẫu nhiên batch_size vị trí bắt đầu
        ix = np.random.randint(0, max_idx, size=batch_size)
        
        # Gom các đoạn token
        x_list = [np.array(self.data[i : i + self.block_size], dtype=np.int64) for i in ix]
        y_list = [np.array(self.data[i + 1 : i + self.block_size + 1], dtype=np.int64) for i in ix]

        x = torch.from_numpy(np.stack(x_list))
        y = torch.from_numpy(np.stack(y_list))

        if device != 'cpu':
            # Pinned memory / direct transfer nếu dùng GPU
            x = x.pin_memory().to(device, non_blocking=True) if torch.cuda.is_available() else x.to(device)
            y = y.pin_memory().to(device, non_blocking=True) if torch.cuda.is_available() else y.to(device)

        return x, y

def get_dataloader(bin_path: str, block_size: int = 256):
    """Hàm tiện ích nạp nhanh dataset."""
    return MemmapDataset(bin_path=bin_path, block_size=block_size)

