# PROPOSAL BÀI TẬP LỚN MÔN HỌC SÂU
## ĐỀ TÀI 9: XÂY DỰNG MÔ HÌNH NGÔN NGỮ NHỎ (MINI-GPT) TỪ ĐẦU CHO TIẾNG VIỆT
**Định dạng: ACL Conference | Giới hạn: Đúng 2 trang nội dung**

**Nhóm sinh viên:**
- **Thành viên 1 (Trưởng nhóm): Nguyễn Đức Vũ Bảo** (MSSV: `[Điền MSSV]`)  
  *Phụ trách:* Xây dựng Data Pipeline, Unicode NFC, huấn luyện BPE Tokenizer 16k, Baseline 1 (N-gram LM), dữ liệu Instruction-tuning.
- **Thành viên 2:** `[Họ và tên TV2]` (MSSV: `[Điền MSSV]`)  
  *Phụ trách:* Tự cài đặt kiến trúc GPT (Attention, MLP, Pre-LayerNorm, Weight Tying), xây dựng pipeline `train.py` (AdamW, Cosine LR, AMP FP16, GradAccum), huấn luyện mô hình chính (10--30M).
- **Thành viên 3:** `[Họ và tên TV3]` (MSSV: `[Điền MSSV]`)  
  *Phụ trách:* Cài đặt Baseline 2 (LSTM LM), pipeline đánh giá Perplexity/BPB, thực nghiệm Scaling Law (khảo sát kích thước mô hình vs loss), xây dựng thuật toán giải mã sinh văn bản (Temperature, Top-$k$, Top-$p$).

---

### 1. Đặt vấn đề và Mục tiêu bài toán (Problem Statement)
- **Bối cảnh & Động lực:** Các mô hình ngôn ngữ lớn (LLMs) dựa trên kiến trúc Transformer decoder đã trở thành tiêu chuẩn trong xử lý ngôn ngữ tự nhiên. Tuy nhiên, việc hiểu sâu sắc bản chất kiến trúc ở mức mã nguồn (từ embedding, cơ chế causal attention, đến gradient accumulation và decoding) đòi hỏi phải tự cài đặt từ đầu mà không dựa vào các framework cấp cao có sẵn.
- **Mục tiêu:** Xây dựng một mô hình ngôn ngữ tự hồi quy kiểu GPT cỡ nhỏ (10–30 triệu tham số) từ đầu bằng PyTorch trên tập ngữ liệu tiếng Việt, đối chuẩn trực tiếp với 2 baseline truyền thống (N-gram và LSTM), đồng thời khảo sát định luật mở rộng quy mô (Scaling Law) và các chiến lược lấy mẫu (Sampling).
- **Đầu vào (Input):** Một đoạn văn bản mở đầu tiếng Việt (Prompt: $x_1, x_2, \dots, x_t$).
- **Đầu ra (Output):** Phân phối xác suất có điều kiện của token kế tiếp $P(x_{t+1} \mid x_{\le t})$ và chuỗi văn bản hoàn chỉnh được sinh tiếp tự hồi quy.

---

### 2. Các công trình nghiên cứu chính (Related Work & 3 Bài báo nền tảng)
Dự án nghiên cứu và kế thừa trực tiếp từ 3 bài báo kinh điển:
1. **Vaswani et al. (2017) - *Attention Is All You Need*:** Đặt nền móng cho cơ chế Multi-Head Scaled Dot-Product Self-Attention, loại bỏ hoàn toàn tính tuần tự của mạng hồi quy và cho phép tính toán song song triệt để trên GPU.
2. **Radford et al. (2019) - *Language Models are Unsupervised Multitask Learners* (GPT-2):** Đề xuất kiến trúc Decoder-only cải tiến với Pre-LayerNorm và Weight Tying giữa ma trận embedding đầu vào và classification head đầu ra; sử dụng Byte-Level Byte-Pair Encoding (BPE) để xử lý triệt để từ ngoài từ điển (OOV).
3. **Kaplan et al. (2020) - *Scaling Laws for Neural Language Models*:** Chứng minh hàm mất mát (loss) của Transformer tuân theo định luật lũy thừa (Power Law) theo số lượng tham số phi-embedding ($N$) và lượng dữ liệu ($D$), làm cơ sở để nhóm thiết kế thí nghiệm Scaling Law.

---

### 3. Dữ liệu và Tiền xử lý (Dataset & Tokenization)
- **Tập ngữ liệu chính:** Trích xuất từ Wikipedia tiếng Việt (dump sạch khoảng 60MB – 100MB văn bản thuần, tương đương ~25–35 triệu tokens).
- **Chuẩn hóa Unicode (NFC):** Văn bản tiếng Việt thường lẫn lộn giữa Unicode tổ hợp (NFD) và dựng sẵn (NFC). Nhóm chuẩn hóa toàn bộ về chuẩn NFC (`unicodedata.normalize('NFC')`) để bảo toàn tính nhất quán của các nguyên âm có dấu.
- **Phân chia dữ liệu:** Tỷ lệ phân tách là **90% Train / 5% Validation / 5% Test** theo cấp bài viết (tránh rò rỉ ngữ cảnh giữa các câu trong cùng một văn bản).
- **Tokenizer:** Huấn luyện bộ Byte-Level BPE với kích thước từ vựng $V = 16.384$ (chia hết cho 64 để tối ưu hóa nhân ma trận Tensor Cores trên GPU). Các token đặc biệt: `<|endoftext|>`, `<|pad|>`.
- **Định dạng nhị phân:** Dữ liệu sau khi mã hóa được lưu dưới dạng file `.bin` gồm các số nguyên 16-bit (`np.uint16`), nạp vào mô hình bằng `numpy.memmap` nhằm tối ưu hóa 0% RAM tĩnh.
- **(Mở rộng) Dữ liệu Instruction-tuning:** Chuẩn bị ~2.000–3.000 mẫu hỏi-đáp ngắn (ViAlpaca) theo cấu trúc: `### Câu hỏi: ... \n### Trả lời: ... <|endoftext|>`.

---

### 4. Phương pháp và Mô hình đề xuất (Methodology & Baselines)

#### 4.1. Hai mô hình Baseline đối chuẩn:
1. **Baseline 1 (N-gram LM):** Xây dựng mô hình 3-gram và 4-gram có làm mịn Add-$k$ (Laplace/Lidstone smoothing) và Backoff. **Điểm quan trọng:** N-gram sẽ được huấn luyện trực tiếp trên chuỗi ID của Tokenizer BPE để không gian từ vựng đồng nhất với GPT, đảm bảo phép đo Perplexity là hoàn toàn công bằng.
2. **Baseline 2 (LSTM LM):** Mạng LSTM đa tầng gồm `nn.Embedding(16384, 384)` $\to$ 2 tầng `nn.LSTM(hidden=512, dropout=0.2)` $\to$ `nn.Linear(512, 16384)`. Kích thước mô hình được thiết kế ở mức **~12 triệu tham số** nhằm đối trọng trực diện với mô hình GPT-Small cùng ngân sách tham số.

#### 4.2. Mô hình chính: Kiến trúc Mini-GPT (Transformer Decoder):
- Tự cài đặt bằng PyTorch từ đầu gồm các thành phần:
  - **Token & Learned Positional Embedding**: Context length cố định $T = 256$ tokens.
  - **Pre-LayerNorm Transformer Block**: Chuẩn hóa đầu vào của từng khối ($x + \text{Attn}(\text{LN}(x))$, $x + \text{MLP}(\text{LN}(x))$) giúp gradient ổn định ở mô hình sâu.
  - **FlashAttention**: Tận dụng `F.scaled_dot_product_attention` với cờ `is_causal=True`.
  - **Weight Tying**: Chia sẻ trọng số `lm_head.weight = wte.weight`.
- **Thiết lập huấn luyện:** Thuật toán tối ưu **AdamW** ($\beta_1=0.9, \beta_2=0.95$, weight decay = 0.1 đối với ma trận 2D). Lịch trình học Cosine Annealing decay có Warmup 500 bước. Huấn luyện bằng Mixed Precision (FP16) kết hợp Gradient Accumulation.

---

### 5. Phương pháp đánh giá (Evaluation Framework)
1. **Độ đo Perplexity (PPL) và Bits-Per-Byte (BPB):**
   $$\text{PPL} = \exp\left( -\frac{1}{M}\sum_{t=1}^M \ln P(x_t \mid x_{<t}) \right), \quad \text{BPB} = \frac{\text{Total NLL (nats)}}{\ln(2) \times \text{Total UTF-8 Bytes}}$$
   Đo lường trên cùng tập Test cố định để lập bảng so sánh 3 mô hình: N-gram, LSTM và GPT.
2. **Thí nghiệm Mở rộng 1 - Scaling Law:** Khảo sát 4 mốc tham số không tính embedding:
   - **Mini (~3.8M)**: 4 layers, 4 heads, $d=192$
   - **Small (~11.5M)**: 6 layers, 6 heads, $d=384$ (Mô hình chính đạt chuẩn đề tài)
   - **Medium (~22.8M)**: 6 layers, 8 heads, $d=512$
   - **Large (~31.2M)**: 9 layers, 8 heads, $d=512$
   Vẽ đồ thị log-log và fit phương trình hàm mũ $L(N) = a \cdot N^{-b} + c$.
3. **Thí nghiệm Mở rộng 2 - Khảo sát thuật toán Decoding:** So sánh văn bản sinh ra từ 5 prompt mở đầu khi thay đổi các tổ hợp tham số: Temperature $T \in \{0.5, 0.8, 1.2\}$, Top-$k \in \{10, 50\}$, Top-$p \in \{0.85, 0.95\}$; đo mức độ lặp từ (distinct-n) và chấm điểm mù (blind human evaluation).
4. **Thí nghiệm Mở rộng 3 - Instruction-Tuning:** So sánh khả năng trả lời chỉ dẫn trước và sau khi tinh chỉnh nhẹ trên tập hỏi-đáp.

---

### 6. Kế hoạch thực hiện và Phân công công việc (Timeline & Task Allocation)

| Tuần | Mốc công việc chính | Sản phẩm nộp / Bàn giao |
| :---: | :--- | :--- |
| **Tuần 1** | **Proposal**, thu thập ngữ liệu, chuẩn hóa NFC, train Tokenizer BPE, cấu trúc repo và chạy thử môi trường. | **Proposal (2 trang)**, `train.bin`, `tokenizer.json` |
| **Tuần 2** | Hoàn thành Baseline 1 (N-gram) và Baseline 2 (LSTM). Thiết lập pipeline `train.py`. | Điểm PPL của N-gram & LSTM |
| **Tuần 3** | Huấn luyện mô hình GPT chính (11.5M), tối ưu Mixed Precision và kiểm soát hội tụ. | **Milestone Report (4 trang)**, Checkpoint GPT |
| **Tuần 4** | Thực hiện thí nghiệm Scaling Law (train các mốc Mini, Medium, Large) và trích xuất loss curve. | Bảng và biểu đồ Scaling Law |
| **Tuần 5** | Thí nghiệm giải mã (Sampling Grid), chạy thử Instruction-tuning, phân tích lỗi văn bản tiếng Việt. | Bản nháp báo cáo đầy đủ |
| **Tuần 6** | Hoàn thiện Báo cáo ACL (8 trang), biên dịch mã nguồn sạch, chuẩn bị slide và demo sinh văn bản. | **Báo cáo cuối (8 trang)**, Source code, Slide, Demo |

**Bảng phân công trách nhiệm chi tiết:**
- **Thành viên 1 (Nguyễn Đức Vũ Bảo):** Phụ trách toàn bộ Data Pipeline (tải, lọc, chuẩn hóa Unicode NFC, phân chia train/val/test); Huấn luyện BPE Tokenizer 16k; Cài đặt và đánh giá Baseline 1 (N-gram LM); Chuẩn bị tập dữ liệu Instruction-tuning.
- **Thành viên 2:** Tự cài đặt kiến trúc GPT Decoder-only (Attention, MLP, LayerNorm, Weight Tying); Xây dựng script `train.py` (AdamW, Cosine schedule, AMP FP16, Gradient Accumulation); Huấn luyện mô hình GPT chính và chạy instruction-tuning.
- **Thành viên 3:** Cài đặt và huấn luyện Baseline 2 (LSTM LM); Xây dựng pipeline đánh giá Perplexity/BPB; Chạy các mô hình để vẽ biểu đồ Scaling Law; Viết module Sampling (Temperature, Top-k, Top-p) và thực hiện đánh giá định tính.
