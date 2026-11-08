import os
import sys
import argparse
import unicodedata
from pathlib import Path

def download_wikipedia_vi(output_path: Path, target_mb: float = 60.0):
    """
    Tải tập dữ liệu Wikipedia tiếng Việt thật từ Hugging Face qua stream.
    Chuẩn hóa NFC và lưu trực tiếp thành văn bản sạch.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[!] Chưa cài đặt thư viện 'datasets'. Vui lòng chạy: pip install datasets")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_bytes = target_mb * 1024 * 1024

    print(f"[*] Bắt đầu tải Wikipedia tiếng Việt thật từ Hugging Face...")
    print(f"[*] Mục tiêu dung lượng: {target_mb:.1f} MB văn bản thuần")
    print(f"[*] File lưu: {output_path}")

    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
        doc_count = 0
        written_bytes = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for item in ds:
                text = item.get("text", "")
                if not text:
                    continue

                # Chuẩn hóa bảng mã Unicode NFC
                norm_text = unicodedata.normalize("NFC", text).strip()
                if len(norm_text) < 120:
                    continue

                # Ghi bài viết vào file, ngăn cách bằng 2 dòng trống
                entry = norm_text + "\n\n"
                f.write(entry)
                doc_count += 1

                if doc_count % 500 == 0:
                    written_bytes = output_path.stat().st_size
                    current_mb = written_bytes / (1024 * 1024)
                    print(f"    -> Đã tải {doc_count:,} bài viết | Dung lượng: {current_mb:.2f} MB / {target_mb:.1f} MB ({(current_mb/target_mb)*100:.1f}%)")
                    if written_bytes >= target_bytes:
                        break

        total_mb = output_path.stat().st_size / (1024 * 1024)
        print("\n" + "=" * 60)
        print("TẢI DỮ LIỆU THẬT HOÀN TẤT!")
        print(f"- Tổng số bài viết Wikipedia: {doc_count:,}")
        print(f"- Dung lượng văn bản thuần: {total_mb:.2f} MB")
        print(f"- Lưu tại: {output_path}")
        print("=" * 60)
        return True

    except Exception as e:
        print(f"[!] Lỗi trong quá trình tải: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Tải dữ liệu Wikipedia tiếng Việt thật (Thành viên 1)")
    parser.add_argument("--output", type=str, default="data/raw/wikipedia_vi.txt", help="Đường dẫn lưu file")
    parser.add_argument("--size_mb", type=float, default=50.0, help="Dung lượng mục tiêu (MB)")
    args = parser.parse_args()

    download_wikipedia_vi(Path(args.output), target_mb=args.size_mb)

if __name__ == "__main__":
    main()
