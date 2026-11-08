import os
import sys
import argparse
import unicodedata
import numpy as np
from pathlib import Path

# Special tokens
SPECIAL_TOKENS = ["<|endoftext|>", "<|pad|>"]
VOCAB_SIZE = 16384

SAMPLE_VIETNAMESE_TEXT = [
    "Ngày xửa ngày xưa, ở một ngôi làng nhỏ ven sông Đáy, có một chàng trai nghèo nhưng hiền lành và chăm chỉ.",
    "Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương, thuộc khu vực Đông Nam Á.",
    "Hà Nội là thủ đô của nước Cộng hòa Xã hội Chủ nghĩa Việt Nam, trung tâm văn hóa và chính trị của cả nước.",
    "Trí tuệ nhân tạo (AI) là ngành khoa học máy tính nghiên cứu và phát triển các hệ thống có khả năng thực hiện các nhiệm vụ đòi hỏi trí thông minh của con người.",
    "Mô hình ngôn ngữ lớn (Large Language Model - LLM) dựa trên kiến trúc Transformer đã tạo ra bước đột phá vượt bậc trong xử lý ngôn ngữ tự nhiên.",
    "Truyện Kiều của đại thi hào Nguyễn Du là một kiệt tác của văn học trung đại Việt Nam, viết bằng thể thơ lục bát.",
    "Trăm năm trong cõi người ta, chữ tài chữ mệnh khéo là ghét nhau. Trải qua một cuộc bể dâu, những điều trông thấy mà đau đớn lòng.",
    "Khoa học dữ liệu và học máy đang mở ra nhiều cơ hội ứng dụng trong y tế, giáo dục, giao thông và tài chính.",
    "Thành phố Hồ Chí Minh là trung tâm kinh tế, tài chính và công nghệ năng động bậc nhất của Việt Nam.",
    "Học sâu (Deep Learning) sử dụng các mạng nơ-ron nhiều tầng để tự động học các biểu diễn đặc trưng phức tạp từ dữ liệu."
]

def normalize_text(text: str) -> str:
    """Chuẩn hóa bảng mã Unicode sang dạng NFC và làm sạch khoảng trắng thừa."""
    if not isinstance(text, str):
        return ""
    # Chuẩn hóa về NFC
    text = unicodedata.normalize("NFC", text)
    # Xử lý xuống dòng thống nhất
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip()

def clean_and_split_corpus(raw_dir: Path, min_length: int = 40):
    """Đọc toàn bộ file txt trong raw_dir, làm sạch và gom thành danh sách văn bản."""
    documents = []
    txt_files = list(raw_dir.glob("*.txt"))
    
    if not txt_files:
        print(f"[!] Không tìm thấy file .txt nào trong {raw_dir}")
        return documents

    for txt_path in txt_files:
        print(f"[*] Đang đọc: {txt_path.name}")
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        
        # Tách theo bài viết (nếu dùng delimiter) hoặc theo đoạn văn bản
        paragraphs = content.split("\n\n")
        for p in paragraphs:
            norm_p = normalize_text(p)
            if len(norm_p) >= min_length:
                documents.append(norm_p)
                
    print(f"[+] Tổng số đoạn/văn bản thu được: {len(documents)}")
    return documents

def generate_mock_data(raw_dir: Path, num_copies: int = 500):
    """Tạo tập văn bản mẫu để chạy thử pipeline ngay lập tức."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    sample_file = raw_dir / "sample_corpus.txt"
    print(f"[*] Đang tạo dữ liệu mẫu tại: {sample_file}")
    
    with open(sample_file, "w", encoding="utf-8") as f:
        for i in range(num_copies):
            for sentence in SAMPLE_VIETNAMESE_TEXT:
                f.write(sentence + f" (Mục số {i})\n\n")
    print(f"[+] Đã tạo xong {sample_file} (~{os.path.getsize(sample_file) / 1024:.2f} KB)")
    return sample_file

def train_tokenizer(texts_for_training: list, save_path: Path, vocab_size: int = VOCAB_SIZE):
    """Huấn luyện Byte-Level BPE Tokenizer chuẩn trên văn bản tiếng Việt."""
    try:
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import ByteLevel
        from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    except ImportError:
        print("[!] Chưa cài đặt thư viện 'tokenizers'. Vui lòng chạy: pip install tokenizers")
        sys.exit(1)

    print(f"[*] Đang huấn luyện BPE Tokenizer với vocab_size = {vocab_size}...")
    tokenizer = Tokenizer(BPE(unk_token=None))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet()
    )

    tokenizer.train_from_iterator(texts_for_training, trainer=trainer)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(save_path))
    print(f"[+] Tokenizer đã được lưu tại: {save_path}")
    return tokenizer

def encode_and_save_bin(tokenizer, documents: list, split_ratios=(0.9, 0.05, 0.05), output_dir: Path = Path("data/processed")):
    """Mã hóa toàn bộ văn bản sang uint16 và lưu thành train.bin, val.bin, test.bin."""
    output_dir.mkdir(parents=True, exist_ok=True)
    eot_id = tokenizer.token_to_id("<|endoftext|>")
    
    n_docs = len(documents)
    n_train = int(n_docs * split_ratios[0])
    n_val = int(n_docs * split_ratios[1])
    
    splits = {
        "train": documents[:n_train],
        "val": documents[n_train:n_train + n_val],
        "test": documents[n_train + n_val:]
    }

    stats = {}
    for split_name, docs in splits.items():
        print(f"[*] Đang mã hóa tập {split_name} ({len(docs)} bài viết)...")
        all_ids = []
        for doc in docs:
            encoded = tokenizer.encode(doc)
            all_ids.extend(encoded.ids)
            all_ids.append(eot_id)  # Phân cách các bài viết bằng <|endoftext|>

        arr = np.array(all_ids, dtype=np.uint16)
        bin_path = output_dir / f"{split_name}.bin"
        arr.tofile(bin_path)
        stats[split_name] = len(arr)
        print(f"[+] Đã lưu {bin_path.name}: {len(arr):,} tokens ({bin_path.stat().st_size / (1024*1024):.2f} MB)")

    return stats

def main():
    parser = argparse.ArgumentParser(description="Pipeline tiền xử lý dữ liệu tiếng Việt & Tokenizer (Thành viên 1)")
    parser.add_argument("--demo", action="store_true", help="Tạo dữ liệu mẫu nhanh để bàn giao test pipeline")
    parser.add_argument("--raw_dir", type=str, default="data/raw", help="Thư mục chứa văn bản thô .txt")
    parser.add_argument("--processed_dir", type=str, default="data/processed", help="Thư mục xuất .bin")
    parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE, help="Kích thước từ vựng BPE")
    args = parser.parse_args()

    raw_path = Path(args.raw_dir)
    proc_path = Path(args.processed_dir)
    tok_path = Path("data/tokenizer.json")

    # 1. Thu thập / làm sạch dữ liệu
    if args.demo or not any(raw_path.glob("*.txt")):
        print("[!] Thư mục raw trống hoặc đang chạy cờ --demo. Khởi tạo dữ liệu mẫu...")
        generate_mock_data(raw_path)

    documents = clean_and_split_corpus(raw_path)
    if not documents:
        print("[!] Không có văn bản hợp lệ để xử lý!")
        return

    # 2. Huấn luyện Tokenizer
    tokenizer = train_tokenizer(documents, tok_path, vocab_size=args.vocab_size)

    # 3. Mã hóa & xuất các file .bin
    stats = encode_and_save_bin(tokenizer, documents, output_dir=proc_path)

    print("\n" + "="*50)
    print("HOÀN THÀNH BÀN GIAO CHO THÀNH VIÊN 2 & 3:")
    print(f"- Tokenizer: {tok_path}")
    for name, tokens in stats.items():
        print(f"- {name}.bin: {tokens:,} tokens")
    print("="*50)

if __name__ == "__main__":
    main()

