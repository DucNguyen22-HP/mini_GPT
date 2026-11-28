import json
import argparse
import unicodedata
from pathlib import Path

CANDIDATE_DATASETS = [
    ("bkai-foundation-models/vi-alpaca", "train"),
    ("vilm/vietnamese-alpaca", "train"),
    ("5CD-AI/Vietnamese-395k-open-hermes", "train"),
]


def norm_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text or "").strip()


def extract_pair(item: dict):
    """Trích xuất (instruction, output) từ các schema phổ biến trên Hugging Face."""
    if "instruction" in item and "output" in item:
        inst = norm_nfc(item.get("instruction", ""))
        inp = norm_nfc(item.get("input", ""))
        out = norm_nfc(item.get("output", ""))
        if inp:
            inst = f"{inst}\n{inp}"
        return inst, out
    if "conversations" in item and isinstance(item["conversations"], list):
        convs = item["conversations"]
        if len(convs) >= 2:
            inst = norm_nfc(convs[0].get("value", ""))
            out = norm_nfc(convs[1].get("value", ""))
            return inst, out
    return "", ""


def generate_instruction_dataset(output_path: Path, num_samples: int = 2000, max_chars: int = 600):
    """Tải tập hỏi-đáp tiếng Việt thật từ Hugging Face, chuẩn hóa NFC và xuất jsonl."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("[!] Chưa cài đặt thư viện 'datasets'. Vui lòng chạy: pip install datasets")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    samples = []
    seen = set()

    for repo_id, split in CANDIDATE_DATASETS:
        print(f"[*] Đang thử tải từ Hugging Face: {repo_id} ({split})...")
        try:
            ds = load_dataset(repo_id, split=split, streaming=True)
            for item in ds:
                inst, out = extract_pair(item)
                if not inst or not out:
                    continue
                if len(inst) + len(out) > max_chars or len(out) < 10:
                    continue
                if inst in seen:
                    continue
                seen.add(inst)
                samples.append({
                    "instruction": inst,
                    "output": out,
                    "formatted_text": f"### Câu hỏi: {inst}\n### Trả lời: {out}<|endoftext|>",
                })
                if len(samples) >= num_samples:
                    break
            if samples:
                print(f"[+] Đã lấy {len(samples)} mẫu từ {repo_id}")
                break
        except Exception as e:
            print(f"[!] Không tải được {repo_id}: {e}")

    if not samples:
        print("[!] Không tải được dataset nào từ Hugging Face.")
        return

    with open(output_path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[+] Đã lưu {len(samples)} mẫu instruction thật tại: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Tạo tập dữ liệu Instruction-Tuning tiếng Việt (Thành viên 1)")
    parser.add_argument("--output", type=str, default="data/instructions.jsonl", help="Đường dẫn file jsonl")
    parser.add_argument("--num_samples", type=int, default=2000, help="Số lượng mẫu hỏi-đáp")
    args = parser.parse_args()

    generate_instruction_dataset(Path(args.output), num_samples=args.num_samples)


if __name__ == "__main__":
    main()

