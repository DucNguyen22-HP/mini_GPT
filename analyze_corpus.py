import os
import json
import numpy as np
from pathlib import Path
from collections import Counter

def analyze_dataset(data_dir: str = "data/processed", raw_dir: str = "data/raw", tok_path: str = "data/tokenizer.json"):
    """
    Phân tích và trích xuất các chỉ số thống kê của Corpus & Tokenizer 
    để đưa thẳng vào Báo cáo theo chuẩn ACL.
    """
    proc_path = Path(data_dir)
    tok_file = Path(tok_path)
    
    print("=" * 60)
    print("           BÁO CÁO THỐNG KÊ CORPUS & TOKENIZER (ACL FORMAT)      ")
    print("=" * 60)

    # 1. Thống kê file thô
    raw_files = list(Path(raw_dir).glob("*.txt"))
    raw_size_bytes = sum(f.stat().st_size for f in raw_files)
    raw_size_mb = raw_size_bytes / (1024 * 1024)
    print(f"\n1. Dữ liệu thô (Raw text):")
    print(f"   - Số file văn bản thô: {len(raw_files)}")
    print(f"   - Dung lượng thô: {raw_size_mb:.2f} MB ({raw_size_bytes:,} bytes)")

    # 2. Thống kê các tập dữ liệu nhị phân (.bin uint16)
    splits = ["train", "val", "test"]
    split_tokens = {}
    total_tokens = 0

    print(f"\n2. Thống kê phân chia tập dữ liệu (Splits):")
    for s in splits:
        bin_file = proc_path / f"{s}.bin"
        if bin_file.exists():
            data = np.fromfile(bin_file, dtype=np.uint16)
            n_tokens = len(data)
            split_tokens[s] = n_tokens
            total_tokens += n_tokens
            file_mb = bin_file.stat().st_size / (1024 * 1024)
            print(f"   - Tập {s:<5}: {n_tokens:>10,} tokens ({file_mb:.2f} MB)")
        else:
            print(f"   - Tập {s:<5}: [Chưa tạo]")

    # 3. Tỷ lệ nén (Compression Ratio)
    if total_tokens > 0 and raw_size_bytes > 0:
        bytes_per_token = raw_size_bytes / total_tokens
        compression_ratio = raw_size_bytes / (total_tokens * 2)  # uint16 là 2 bytes/token
        print(f"\n3. Hiệu suất Tokenizer (Compression Efficiency):")
        print(f"   - Tỷ lệ byte/token: {bytes_per_token:.2f} bytes/token")
        print(f"   - Hệ số nén so với UTF-8 thô: {compression_ratio:.2f}x")

    # 4. Thống kê Tokenizer
    if tok_file.exists():
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(str(tok_file))
            vocab = tok.get_vocab()
            print(f"\n4. Cấu trúc Tokenizer:")
            print(f"   - Thuật toán: Byte-Level BPE")
            print(f"   - Kích thước từ vựng (Vocab size): {len(vocab):,}")
            print(f"   - Special tokens: <|endoftext|>, <|pad|>")
        except Exception as e:
            print(f"\n4. Tokenizer: Không thể đọc chi tiết ({e})")

    # 5. Xuất bảng Markdown cho Báo cáo ACL
    print("\n" + "-" * 60)
    print("BẢNG DỮ LIỆU ĐỊNH DẠNG MARKDOWN (DÙNG CHO BÁO CÁO):")
    print("-" * 60)
    print("| Tập dữ liệu | Số lượng Tokens | Dung lượng (.bin) | Tỷ lệ (%) |")
    print("|:------------|----------------:|------------------:|----------:|")
    for s, count in split_tokens.items():
        pct = (count / total_tokens * 100) if total_tokens > 0 else 0
        mb = (count * 2) / (1024 * 1024)
        print(f"| {s.capitalize():<11} | {count:>15,} | {mb:>14.2f} MB | {pct:>8.1f}% |")
    print(f"| **Tổng cộng** | **{total_tokens:>11,}** | **{(total_tokens * 2) / (1024 * 1024):>10.2f} MB** | **100.0%** |")
    print("=" * 60)

if __name__ == "__main__":
    analyze_dataset()

