import os
import sys
import argparse
import unicodedata
from pathlib import Path

def download_wikipedia_vi(output_path: Path, target_mb: float = 120.0):
    """
    Tải các shard Parquet của Wikipedia tiếng Việt từ Hugging Face về máy
    qua hf_hub_download rồi đọc cục bộ (tránh lỗi WinError 10038 khi stream trên Windows).
    Chuẩn hóa NFC và lưu trực tiếp thành văn bản sạch.
    """
    try:
        from huggingface_hub import hf_hub_download
        import pyarrow.parquet as pq
    except ImportError:
        print("[!] Thiếu thư viện. Vui lòng chạy: pip install huggingface_hub pyarrow")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_bytes = target_mb * 1024 * 1024
    shards = [f"20231101.vi/train-{i:05d}-of-00004.parquet" for i in range(4)]

    print("[*] Bắt đầu tải Wikipedia tiếng Việt thật từ Hugging Face (Parquet shard)...")
    print(f"[*] Mục tiêu dung lượng: {target_mb:.1f} MB văn bản thuần")
    print(f"[*] File lưu: {output_path}")

    try:
        doc_count = 0
        written_bytes = 0

        with open(output_path, "w", encoding="utf-8") as f:
            for shard in shards:
                print(f"[*] Đang tải shard: {shard} ...")
                local_parquet = hf_hub_download(
                    repo_id="wikimedia/wikipedia",
                    repo_type="dataset",
                    filename=shard,
                )
                parquet_file = pq.ParquetFile(local_parquet)

                for batch in parquet_file.iter_batches(batch_size=500, columns=["text"]):
                    for text in batch.column("text").to_pylist():
                        if not text:
                            continue

                        # Chuẩn hóa bảng mã Unicode NFC
                        norm_text = unicodedata.normalize("NFC", text).strip()
                        if len(norm_text) < 120:
                            continue

                        entry = norm_text + "\n\n"
                        f.write(entry)
                        written_bytes += len(entry.encode("utf-8"))
                        doc_count += 1

                        if doc_count % 5000 == 0:
                            current_mb = written_bytes / (1024 * 1024)
                            print(
                                f"    -> Đã xử lý {doc_count:,} bài viết | "
                                f"Dung lượng: {current_mb:.2f} MB / {target_mb:.1f} MB "
                                f"({(current_mb / target_mb) * 100:.1f}%)"
                            )

                        if written_bytes >= target_bytes:
                            break
                    if written_bytes >= target_bytes:
                        break
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
    parser.add_argument("--size_mb", type=float, default=120.0, help="Dung lượng mục tiêu (MB)")
    args = parser.parse_args()

    download_wikipedia_vi(Path(args.output), target_mb=args.size_mb)

if __name__ == "__main__":
    main()
