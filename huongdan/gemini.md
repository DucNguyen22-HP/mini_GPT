

1. Cấu trúc GitHub Repository và Quy ước chung (Data Contract)
Trước khi ai làm việc của người nấy, cả nhóm tạo một repo chung theo cấu trúc sau để code của 3 người ghép vào là chạy được ngay:

Plaintext


viet-gpt-project/
├── configs/               # Chứa các file cấu hình (gpt_10M.yaml, gpt_30M.yaml, lstm.yaml)
├── data/
│   ├── raw/               # Văn bản thô (.txt)
│   ├── tokenizer.json     # File BPE tokenizer đã huấn luyện (TV1 xuất ra)
│   └── processed/         # train.bin, val.bin, test.bin (TV1 xuất ra)
├── models/
│   ├── ngram.py           # Code N-gram (TV1)
│   ├── lstm_lm.py         # Code LSTM (TV3)
│   └── gpt.py             # Code GPT Decoder (TV2)
├── dataset.py             # Class PyTorch Dataset dùng chung cho cả LSTM và GPT
├── train.py               # Script huấn luyện chính (TV2 viết, TV3 tái sử dụng)
└── evaluate.py            # Script tính Perplexity và sinh văn bản (TV3)


Quy ước định dạng dữ liệu (train.bin, val.bin, test.bin): Toàn bộ văn bản sau khi đi qua Tokenizer sẽ được nối thành 1 mảng 1 chiều duy nhất gồm các ID số nguyên 16-bit (np.uint16), lưu xuống ổ cứng bằng numpy.memmap. Giữa các bài viết/văn bản phải chèn token kết thúc <|endoftext|>.


Hàm DataLoader dùng chung (dataset.py): Cả LSTM và GPT đều học bài toán dự đoán từ tiếp theo (Next-token prediction) với chiều dài ngữ cảnh block_size (chọn block_size = 256 hoặc 512):



Đầu vào X: chuỗi token từ vị trí i đến i+block_size-1.


Nhãn Y: chuỗi token từ vị trí i+1 đến i+block_size.


2. Hướng dẫn kỹ thuật chi tiết cho từng thành viên

Thành viên 1: Dữ liệu, Tokenizer & Baseline N-gram

Xử lý dữ liệu tiếng Việt:



Dùng bộ Wikipedia tiếng Việt (bản dump sạch trên Hugging Face như wikimedia/wikipedia bản 20231101.vi) hoặc VNTC (tin tức), lấy khoảng 100MB – 200MB text thuần.


Bắt buộc chuẩn hóa bảng mã Unicode bằng unicodedata.normalize('NFC', text) (tránh lỗi cùng một chữ "òa" nhưng 2 kiểu gõ tổ hợp/dựng sẵn sinh ra 2 token khác nhau).


Huấn luyện BPE Tokenizer:



Dùng thư viện tokenizers (viết bằng Rust của Hugging Face, chạy mất khoảng 10 giây). Chọn ByteLevelBPETokenizer với vocab_size = 16384 (chia hết cho 64 để tối ưu nhân ma trận trên GPU), thêm các special tokens: ["<|endoftext|>", "<|pad|>"].


Lưu ý sống còn: Không dùng tách từ tiếng Việt (như Underthesea/PyVi) nếu đã dùng Subword BPE, hoặc nếu tách từ ghép bằng dấu _ (ví dụ học_sinh) thì phải thống nhất áp dụng bước đó ở cả lúc nhập Prompt suy luận. Cách hiện đại và ít lỗi nhất cho GPT là để BPE tự học trực tiếp trên văn bản thô.


Xây dựng N-gram và tính Perplexity công bằng:



Thay vì đếm N-gram trên từ (word-level), hãy đếm N-gram trực tiếp trên chuỗi Token ID mà BPE Tokenizer vừa sinh ra. Nếu khác bộ từ vựng V, điểm Perplexity giữa N-gram và GPT không thể đem so sánh trực tiếp với nhau.


Cài đặt mô hình 3-gram (Trigram) với kỹ thuật làm mịn Add-k (Laplace/Lidstone smoothing, với k=0.01) hoặc Kneser-Ney để tránh xác suất bằng 0 khi gặp cụm token lạ trên tập Test, gây ra lỗi PPL=.


Thành viên 2: Cài đặt kiến trúc GPT và Khảo sát Scaling

Cài đặt kiến trúc GPT (Decoder-only Transformer):



Tự viết các class trong models/gpt.py: CausalSelfAttention (dùng F.scaled_dot_product_attention(..., is_causal=True) của PyTorch 2.0+ để tự động kích hoạt FlashAttention giúp train nhanh gấp 3 lần), MLP (Linear  GELU  Linear), Block (Pre-LayerNorm: x = x + attn(ln1(x))) và GPT.


Weight Tying (Chia sẻ trọng số): Gán ma trận trọng số của tầng Token Embedding đầu vào trùng với tầng Linear đầu ra (lm_head.weight = wte.weight). Kỹ thuật chuẩn của GPT-2 này giúp tiết kiệm hàng triệu tham số ở tầng từ vựng và giúp mô hình hội tụ nhanh hơn.


Bảng tính sẵn cấu hình tham số (để làm thí nghiệm Scaling Law): Với vocab_size V=16384 và dùng Weight Tying, công thức tổng số tham số xấp xỉ là NVd+L(12d2). Nhóm thiết lập 4 mốc cấu hình sau:



GPT-Mini (~3.8M tham số - Mốc nhỏ): n_layer = 4, n_head = 4, d_model = 192.


GPT-Small (~11.5M tham số - Đạt chuẩn 10–30M): n_layer = 6, n_head = 6, d_model = 384.


GPT-Medium (~22.8M tham số - Đạt chuẩn 10–30M): n_layer = 6, n_head = 8, d_model = 512.


GPT-Large (~31.2M tham số - Mốc cực đại): n_layer = 9, n_head = 8, d_model = 512.


Chiến lược Huấn luyện (Training Loop):



Dùng torch.cuda.amp.autocast(dtype=torch.float16) để giảm một nửa VRAM và tăng tốc độ train trên GPU T4.


Dùng thuật toán tối ưu AdamW (1=0.9,2=0.95, weight_decay = 0.1), learning_rate = 6e-4 có Warmup trong 500 bước đầu và giảm dần theo hàm Cosine.


Dùng Gradient Accumulation (ví dụ batch_size = 32, grad_accum_steps = 4) để tạo batch hiệu dụng lớn (324256=32.768 tokens/step) mà không bị tràn bộ nhớ GPU.


Thành viên 3: Cài đặt LSTM, Sinh văn bản (Sampling) & Đánh giá

Cài đặt LSTM Baseline (models/lstm_lm.py):



Kiến trúc gồm: nn.Embedding(vocab_size, embed_dim)  nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)  nn.Linear(hidden_dim, vocab_size).


Chọn embed_dim = 384, hidden_dim = 512, num_layers = 2 (cũng có kích thước khoảng ~12M tham số để so sánh trực diện, công bằng với bản GPT-Small 11.5M tham số).


Đo lường Perplexity chuẩn xác:



Hàm mất mát khi huấn luyện là Cross-Entropy Loss trên từng token. Khi đánh giá trên tập test.bin, tính trung bình cộng loss trên toàn bộ các batch rồi lấy hàm mũ tự nhiên:





Perplexity=1Tt=1TLCE(yt,yt)
Cài đặt thuật toán Sinh văn bản (Decoding / Sampling):



Từ vector logits ở vị trí token cuối cùng do mô hình dự đoán, cài đặt tuần tự 3 bước biến đổi trong hàm generate():



Temperature (T): Chia logits = logits / T. Khi T0 (ví dụ T=0.1), mô hình chọn từ an toàn, dễ lặp từ; khi T>1.0, văn bản đa dạng nhưng dễ sai ngữ pháp.


Top-k Filtering: Giữ lại k giá trị logits cao nhất, gán toàn bộ các vị trí còn lại bằng - trước khi đưa qua hàm Softmax.


Top-p (Nucleus Sampling): Sắp xếp xác suất giảm dần, tính tổng tích lũy cumsum, loại bỏ các token ở đuôi ngay khi tổng xác suất vượt ngưỡng p (ví dụ p=0.9), chuẩn hóa lại phân phối rồi lấy mẫu bằng torch.multinomial.


Instruction-tuning nhẹ (Phần mở rộng ăn điểm tối đa):



Lấy khoảng 3.000 – 5.000 mẫu hỏi đáp tiếng Việt ngắn (từ tập dữ liệu mở như ViAlpaca trên Hugging Face), định dạng lại theo khuôn mẫu:
### Câu hỏi: {instruction}\n### Trả lời: {output}<|endoftext|>


Nạp checkpoint tốt nhất của mô hình GPT 22.8M tham số, giảm learning_rate xuống 5e-5 và fine-tune thêm 2–3 epoch (chỉ mất khoảng 10–15 phút trên GPU) để mô hình biết trả lời câu hỏi thay vì chỉ viết nối tiếp câu.


3. Checklist các bảng biểu cần có trong Báo cáo cuối kỳ
Khi chạy thí nghiệm, Thành viên 3 cần thu thập số liệu để vẽ đúng 4 bảng/biểu đồ cốt lõi mà giảng viên chấm điểm sẽ tìm kiếm:

Bảng 1 (So sánh Baseline): So sánh Số lượng tham số, Thời gian train/epoch, Validation Loss và Test Perplexity giữa 3 mô hình: 3-gram, LSTM (12M), và GPT-Small (11.5M).


Biểu đồ 2 (Khảo sát Scaling Law): Trục hoành là số tham số (3.8M  11.5M  22.8M  31.2M theo thang logarit), trục tung là Validation Loss sau cùng (và đường cong Loss theo số bước huấn luyện của cả 4 mô hình trên cùng một đồ thị).


Bảng 3 (Khảo sát Sampling): Giữ cố định 3 câu Prompt mở đầu (ví dụ: "Việt Nam là một quốc gia", "Ngày xửa ngày xưa", "Trong lĩnh vực trí tuệ nhân tạo"), xuất văn bản sinh ra ứng với các tổ hợp tham số (T{0.5,0.8,1.2}, k{10,50}, p{0.85,0.95}) và nhận xét hiện tượng lặp từ hoặc lan man.


Bảng 4 (Kết quả Instruction-Tuning): So sánh câu trả lời của mô hình GPT trước và sau khi fine-tune trên 3–5 câu hỏi thực tế.


Bạn muốn mình viết sẵn khung code mẫu (boilerplate code) cho phần nào trước: script tạo dữ liệu BPE uint16 của Thành viên 1, hay class kiến trúc GPT + Training Loop chuẩn của Thành viên 2?



1. Cấu trúc GitHub Repository và Quy ước chung (Data Contract)
Trước khi ai làm việc của người nấy, cả nhóm tạo một repo chung theo cấu trúc sau để code của 3 người ghép vào là chạy được ngay:

Plaintext


viet-gpt-project/
├── configs/               # Chứa các file cấu hình (gpt_10M.yaml, gpt_30M.yaml, lstm.yaml)
├── data/
│   ├── raw/               # Văn bản thô (.txt)
│   ├── tokenizer.json     # File BPE tokenizer đã huấn luyện (TV1 xuất ra)
│   └── processed/         # train.bin, val.bin, test.bin (TV1 xuất ra)
├── models/
│   ├── ngram.py           # Code N-gram (TV1)
│   ├── lstm_lm.py         # Code LSTM (TV3)
│   └── gpt.py             # Code GPT Decoder (TV2)
├── dataset.py             # Class PyTorch Dataset dùng chung cho cả LSTM và GPT
├── train.py               # Script huấn luyện chính (TV2 viết, TV3 tái sử dụng)
└── evaluate.py            # Script tính Perplexity và sinh văn bản (TV3)


Quy ước định dạng dữ liệu (train.bin, val.bin, test.bin): Toàn bộ văn bản sau khi đi qua Tokenizer sẽ được nối thành 1 mảng 1 chiều duy nhất gồm các ID số nguyên 16-bit (np.uint16), lưu xuống ổ cứng bằng numpy.memmap. Giữa các bài viết/văn bản phải chèn token kết thúc <|endoftext|>.


Hàm DataLoader dùng chung (dataset.py): Cả LSTM và GPT đều học bài toán dự đoán từ tiếp theo (Next-token prediction) với chiều dài ngữ cảnh block_size (chọn block_size = 256 hoặc 512):



Đầu vào X: chuỗi token từ vị trí i đến i+block_size-1.


Nhãn Y: chuỗi token từ vị trí i+1 đến i+block_size.


2. Hướng dẫn kỹ thuật chi tiết cho từng thành viên

Thành viên 1: Dữ liệu, Tokenizer & Baseline N-gram

Xử lý dữ liệu tiếng Việt:



Dùng bộ Wikipedia tiếng Việt (bản dump sạch trên Hugging Face như wikimedia/wikipedia bản 20231101.vi) hoặc VNTC (tin tức), lấy khoảng 100MB – 200MB text thuần.


Bắt buộc chuẩn hóa bảng mã Unicode bằng unicodedata.normalize('NFC', text) (tránh lỗi cùng một chữ "òa" nhưng 2 kiểu gõ tổ hợp/dựng sẵn sinh ra 2 token khác nhau).


Huấn luyện BPE Tokenizer:



Dùng thư viện tokenizers (viết bằng Rust của Hugging Face, chạy mất khoảng 10 giây). Chọn ByteLevelBPETokenizer với vocab_size = 16384 (chia hết cho 64 để tối ưu nhân ma trận trên GPU), thêm các special tokens: ["<|endoftext|>", "<|pad|>"].


Lưu ý sống còn: Không dùng tách từ tiếng Việt (như Underthesea/PyVi) nếu đã dùng Subword BPE, hoặc nếu tách từ ghép bằng dấu _ (ví dụ học_sinh) thì phải thống nhất áp dụng bước đó ở cả lúc nhập Prompt suy luận. Cách hiện đại và ít lỗi nhất cho GPT là để BPE tự học trực tiếp trên văn bản thô.


Xây dựng N-gram và tính Perplexity công bằng:



Thay vì đếm N-gram trên từ (word-level), hãy đếm N-gram trực tiếp trên chuỗi Token ID mà BPE Tokenizer vừa sinh ra. Nếu khác bộ từ vựng V, điểm Perplexity giữa N-gram và GPT không thể đem so sánh trực tiếp với nhau.


Cài đặt mô hình 3-gram (Trigram) với kỹ thuật làm mịn Add-k (Laplace/Lidstone smoothing, với k=0.01) hoặc Kneser-Ney để tránh xác suất bằng 0 khi gặp cụm token lạ trên tập Test, gây ra lỗi PPL=.


Thành viên 2: Cài đặt kiến trúc GPT và Khảo sát Scaling

Cài đặt kiến trúc GPT (Decoder-only Transformer):



Tự viết các class trong models/gpt.py: CausalSelfAttention (dùng F.scaled_dot_product_attention(..., is_causal=True) của PyTorch 2.0+ để tự động kích hoạt FlashAttention giúp train nhanh gấp 3 lần), MLP (Linear  GELU  Linear), Block (Pre-LayerNorm: x = x + attn(ln1(x))) và GPT.


Weight Tying (Chia sẻ trọng số): Gán ma trận trọng số của tầng Token Embedding đầu vào trùng với tầng Linear đầu ra (lm_head.weight = wte.weight). Kỹ thuật chuẩn của GPT-2 này giúp tiết kiệm hàng triệu tham số ở tầng từ vựng và giúp mô hình hội tụ nhanh hơn.


Bảng tính sẵn cấu hình tham số (để làm thí nghiệm Scaling Law): Với vocab_size V=16384 và dùng Weight Tying, công thức tổng số tham số xấp xỉ là NVd+L(12d2). Nhóm thiết lập 4 mốc cấu hình sau:



GPT-Mini (~3.8M tham số - Mốc nhỏ): n_layer = 4, n_head = 4, d_model = 192.


GPT-Small (~11.5M tham số - Đạt chuẩn 10–30M): n_layer = 6, n_head = 6, d_model = 384.


GPT-Medium (~22.8M tham số - Đạt chuẩn 10–30M): n_layer = 6, n_head = 8, d_model = 512.


GPT-Large (~31.2M tham số - Mốc cực đại): n_layer = 9, n_head = 8, d_model = 512.


Chiến lược Huấn luyện (Training Loop):



Dùng torch.cuda.amp.autocast(dtype=torch.float16) để giảm một nửa VRAM và tăng tốc độ train trên GPU T4.


Dùng thuật toán tối ưu AdamW (1=0.9,2=0.95, weight_decay = 0.1), learning_rate = 6e-4 có Warmup trong 500 bước đầu và giảm dần theo hàm Cosine.


Dùng Gradient Accumulation (ví dụ batch_size = 32, grad_accum_steps = 4) để tạo batch hiệu dụng lớn (324256=32.768 tokens/step) mà không bị tràn bộ nhớ GPU.


Thành viên 3: Cài đặt LSTM, Sinh văn bản (Sampling) & Đánh giá

Cài đặt LSTM Baseline (models/lstm_lm.py):



Kiến trúc gồm: nn.Embedding(vocab_size, embed_dim)  nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.2)  nn.Linear(hidden_dim, vocab_size).


Chọn embed_dim = 384, hidden_dim = 512, num_layers = 2 (cũng có kích thước khoảng ~12M tham số để so sánh trực diện, công bằng với bản GPT-Small 11.5M tham số).


Đo lường Perplexity chuẩn xác:



Hàm mất mát khi huấn luyện là Cross-Entropy Loss trên từng token. Khi đánh giá trên tập test.bin, tính trung bình cộng loss trên toàn bộ các batch rồi lấy hàm mũ tự nhiên:





Perplexity=1Tt=1TLCE(yt,yt)
Cài đặt thuật toán Sinh văn bản (Decoding / Sampling):



Từ vector logits ở vị trí token cuối cùng do mô hình dự đoán, cài đặt tuần tự 3 bước biến đổi trong hàm generate():



Temperature (T): Chia logits = logits / T. Khi T0 (ví dụ T=0.1), mô hình chọn từ an toàn, dễ lặp từ; khi T>1.0, văn bản đa dạng nhưng dễ sai ngữ pháp.


Top-k Filtering: Giữ lại k giá trị logits cao nhất, gán toàn bộ các vị trí còn lại bằng - trước khi đưa qua hàm Softmax.


Top-p (Nucleus Sampling): Sắp xếp xác suất giảm dần, tính tổng tích lũy cumsum, loại bỏ các token ở đuôi ngay khi tổng xác suất vượt ngưỡng p (ví dụ p=0.9), chuẩn hóa lại phân phối rồi lấy mẫu bằng torch.multinomial.


Instruction-tuning nhẹ (Phần mở rộng ăn điểm tối đa):



Lấy khoảng 3.000 – 5.000 mẫu hỏi đáp tiếng Việt ngắn (từ tập dữ liệu mở như ViAlpaca trên Hugging Face), định dạng lại theo khuôn mẫu:
### Câu hỏi: {instruction}\n### Trả lời: {output}<|endoftext|>


Nạp checkpoint tốt nhất của mô hình GPT 22.8M tham số, giảm learning_rate xuống 5e-5 và fine-tune thêm 2–3 epoch (chỉ mất khoảng 10–15 phút trên GPU) để mô hình biết trả lời câu hỏi thay vì chỉ viết nối tiếp câu.


3. Checklist các bảng biểu cần có trong Báo cáo cuối kỳ
Khi chạy thí nghiệm, Thành viên 3 cần thu thập số liệu để vẽ đúng 4 bảng/biểu đồ cốt lõi mà giảng viên chấm điểm sẽ tìm kiếm:

Bảng 1 (So sánh Baseline): So sánh Số lượng tham số, Thời gian train/epoch, Validation Loss và Test Perplexity giữa 3 mô hình: 3-gram, LSTM (12M), và GPT-Small (11.5M).


Biểu đồ 2 (Khảo sát Scaling Law): Trục hoành là số tham số (3.8M  11.5M  22.8M  31.2M theo thang logarit), trục tung là Validation Loss sau cùng (và đường cong Loss theo số bước huấn luyện của cả 4 mô hình trên cùng một đồ thị).


Bảng 3 (Khảo sát Sampling): Giữ cố định 3 câu Prompt mở đầu (ví dụ: "Việt Nam là một quốc gia", "Ngày xửa ngày xưa", "Trong lĩnh vực trí tuệ nhân tạo"), xuất văn bản sinh ra ứng với các tổ hợp tham số (T{0.5,0.8,1.2}, k{10,50}, p{0.85,0.95}) và nhận xét hiện tượng lặp từ hoặc lan man.


Bảng 4 (Kết quả Instruction-Tuning): So sánh câu trả lời của mô hình GPT trước và sau khi fine-tune trên 3–5 câu hỏi thực tế.


Bạn muốn mình viết sẵn khung code mẫu (boilerplate code) cho phần nào trước: script tạo dữ liệu BPE uint16 của Thành viên 1, hay class kiến trúc GPT + Training Loop chuẩn của Thành viên 2?

