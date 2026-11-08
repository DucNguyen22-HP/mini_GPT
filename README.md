# Bài Tập Lớn Môn Học Sâu - Đề Tài 9
## Xây dựng mô hình ngôn ngữ nhỏ (mini-GPT) từ đầu cho tiếng Việt

### 1. Thành viên nhóm & Phân công nhiệm vụ
- **Thành viên 1 (Trưởng nhóm): Nguyễn Đức Vũ Bảo** (MSSV: `[Điền MSSV]`)
  - *Nhiệm vụ chính:* Xây dựng Data Pipeline, chuẩn hóa Unicode NFC, huấn luyện BPE Tokenizer 16k, cài đặt Baseline 1 (N-gram LM), chuẩn bị dữ liệu Instruction-tuning.
- **Thành viên 2:** `[Họ và tên TV2]` (MSSV: `[Điền MSSV]`)
  - *Nhiệm vụ chính:* Tự cài đặt kiến trúc GPT Decoder-only (Attention, MLP, Pre-LayerNorm, Weight Tying), xây dựng pipeline `train.py` (AdamW, Cosine LR, AMP FP16, GradAccum), huấn luyện mô hình chính (10--30M).
- **Thành viên 3:** `[Họ và tên TV3]` (MSSV: `[Điền MSSV]`)
  - *Nhiệm vụ chính:* Cài đặt Baseline 2 (LSTM LM), pipeline đánh giá Perplexity/BPB, thực nghiệm Scaling Law (khảo sát kích thước mô hình vs loss), xây dựng thuật toán giải mã sinh văn bản (Temperature, Top-$k$, Top-$p$).

---

### 2. Cấu trúc thư mục dự án
```text
BTLDeeplearning/
├── data/
│   ├── raw/                  # Chứa văn bản thô tiếng Việt (Wikipedia dump / txt)
│   ├── processed/            # Dữ liệu nhị phân train.bin, val.bin, test.bin (uint16)
│   ├── tokenizer.json        # File BPE Tokenizer 16k đã huấn luyện
│   └── instructions.jsonl    # Dữ liệu hỏi-đáp phục vụ Instruction-tuning mở rộng
├── models/
│   └── ngram.py              # Baseline 1: Mô hình N-gram LM (Laplace smoothing)
├── proposal.tex              # Mã nguồn LaTeX bản Proposal 2 trang chuẩn ACL
├── references.bib            # Trích dẫn BibTeX cho 3 bài báo nền tảng
├── prepare_data.py           # Tiền xử lý dữ liệu, train Tokenizer & xuất .bin
├── download_corpus.py        # Tải ngữ liệu Wikipedia tiếng Việt
├── dataset.py                # DataLoader np.memmap dùng chung cho cả nhóm
├── analyze_corpus.py         # Trích xuất bảng thống kê ngữ liệu chuẩn bài báo ACL
├── prepare_instructions.py   # Tạo tập dữ liệu Instruction-tuning
├── requirements.txt          # Danh sách thư viện cần thiết
└── .gitignore                # Bỏ qua file nhị phân nặng, checkpoint mô hình
```

---

### 3. Hướng dẫn cài đặt và Chạy thử nghiệm (Tuần 1)

#### Bước 1: Cài đặt môi trường
```bash
pip install -r requirements.txt
```

#### Bước 2: Chuẩn bị dữ liệu và huấn luyện Tokenizer
Chạy chế độ thử nghiệm nhanh:
```bash
python prepare_data.py --demo
```
*(Hoặc tải ngữ liệu Wikipedia thật qua `python download_corpus.py` rồi chạy `python prepare_data.py`)*.

#### Bước 3: Đánh giá Baseline 1 (N-gram Language Model)
```bash
python models/ngram.py --n 3 --prompt "Việt Nam là một"
```

#### Bước 4: Trích xuất bảng số liệu thống kê cho Báo cáo
```bash
python analyze_corpus.py
```

---

### 4. Mốc nộp bài Tuần 1: Proposal
Bản Proposal 2 trang theo định dạng ACL được lưu tại mã nguồn `proposal.tex` và trích dẫn tại `references.bib`.

