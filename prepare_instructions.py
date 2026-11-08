import json
import argparse
from pathlib import Path

SAMPLE_INSTRUCTIONS = [
    {
        "instruction": "Thủ đô của Việt Nam là gì?",
        "output": "Thủ đô của Việt Nam là thành phố Hà Nội."
    },
    {
        "instruction": "Kể tên 3 danh lam thắng cảnh nổi tiếng của Việt Nam.",
        "output": "Ba danh lam thắng cảnh nổi tiếng của Việt Nam là Vịnh Hạ Long, Phố cổ Hội An và Quần thể danh thắng Tràng An."
    },
    {
        "instruction": "Học sâu (Deep Learning) là gì?",
        "output": "Học sâu là một nhánh của học máy dựa trên các mạng nơ-ron nhân tạo nhiều tầng, có khả năng tự động học các đặc trưng phức tạp từ dữ liệu thô."
    },
    {
        "instruction": "Mô hình ngôn ngữ tự hồi quy hoạt động như thế nào?",
        "output": "Mô hình ngôn ngữ tự hồi quy dự đoán token tiếp theo trong chuỗi dựa trên phân phối xác suất có điều kiện của toàn bộ các token đã xuất hiện trước đó."
    },
    {
        "instruction": "Viết một câu thơ lục bát về quê hương.",
        "output": "Quê hương là cánh diều chao, Ngọt ngào câu hát dạt dào lời ru."
    }
]

def generate_instruction_dataset(output_path: Path, num_samples: int = 1000):
    """
    Tạo hoặc chuẩn bị tập mẫu hỏi-đáp tiếng Việt phục vụ Instruction-tuning.
    Khuôn mẫu chuẩn:
    ### Câu hỏi: {instruction}
    ### Trả lời: {output}<|endoftext|>
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Đang xuất tập dữ liệu instruction-tuning ra {output_path}...")
    
    samples = []
    # Nhân bản hoặc mở rộng tập mẫu
    for i in range(num_samples):
        base = SAMPLE_INSTRUCTIONS[i % len(SAMPLE_INSTRUCTIONS)]
        samples.append({
            "instruction": base["instruction"],
            "output": base["output"],
            "formatted_text": f"### Câu hỏi: {base['instruction']}\n### Trả lời: {base['output']}<|endoftext|>"
        })

    with open(output_path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[+] Đã tạo {len(samples)} mẫu instruction tại: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Tạo tập dữ liệu Instruction-Tuning (Thành viên 1)")
    parser.add_argument("--output", type=str, default="data/instructions.jsonl", help="Đường dẫn file jsonl")
    parser.add_argument("--num_samples", type=int, default=2000, help="Số lượng mẫu hỏi-đáp")
    args = parser.parse_args()

    generate_instruction_dataset(Path(args.output), num_samples=args.num_samples)

if __name__ == "__main__":
    main()

