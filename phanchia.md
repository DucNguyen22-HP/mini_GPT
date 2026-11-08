Bảng phân công dưới đây chia đều khối lượng công việc cho 3 thành viên bám sát theo đúng 3 mảng gợi ý trong đề bài, giúp cả nhóm có thể bắt tay vào code song song ngay lập tức mà không phải chờ đợi nhau:

Thành viên 1: Phụ trách Dữ liệu (Corpus), Tokenizer & Mô hình N-gram

Xây dựng bộ dữ liệu (Data Pipeline):



Thu thập và làm sạch tập văn bản tiếng Việt từ Wikipedia tiếng Việt, truyện cổ tích, thơ lục bát hoặc tập con tin tức (dung lượng khoảng 100MB – 150MB văn bản thô).


Chuẩn hóa bảng mã Unicode dựng sẵn (NFC), lọc bỏ ký tự rác và chia dữ liệu thành 3 tập train, val, test (tỷ lệ 90/5/5).


(Mở rộng) Chuẩn bị thêm một tập dữ liệu nhỏ gồm vài nghìn mẫu hỏi-đáp ngắn bằng tiếng Việt để phục vụ bước tinh chỉnh (instruction-tune) ở cuối dự án.


Huấn luyện và đóng gói Tokenizer:



Huấn luyện bộ Tokenizer tiếng Việt (sử dụng thuật toán BPE với kích thước từ vựng vocab_size = 16384) trên tập train.


Mã hóa toàn bộ tập dữ liệu sang dạng mảng số nguyên (uint16) và xuất ra 3 file nhị phân train.bin, val.bin, test.bin cùng file tokenizer.json để bàn giao cho Thành viên 2 và 3 dùng chung.


Cài đặt Baseline 1 (N-gram Language Model):



Xây dựng mô hình ngôn ngữ N-gram (3-gram hoặc 4-gram có làm mịn Laplace/Kneser-Ney) để làm baseline.


Đánh giá độ đo Perplexity của mô hình N-gram trên tập test và thử nghiệm sinh văn bản từ một vài câu mở đầu (prompt).


Thành viên 2: Phụ trách Kiến trúc GPT, Pipeline Huấn luyện & Instruction-tuning

Cài đặt kiến trúc GPT (Transformer Decoder):



Tự cài đặt từ đầu bằng PyTorch mô hình ngôn ngữ kiểu GPT cỡ nhỏ (khoảng 10–30 triệu tham số) dựa trên kiến trúc Transformer decoder (bao gồm: Token & Positional Embedding, Causal Self-Attention, MLP, LayerNorm).


Thiết kế file cấu hình linh hoạt cho phép thay đổi số tầng (n_layer), số đầu chú ý (n_head) và chiều ẩn (d_model) để tạo ra các phiên bản mô hình có kích thước khác nhau (ví dụ: 4M, 12M, 25M tham số).


Xây dựng Pipeline Huấn luyện & Huấn luyện mô hình GPT chính:



Viết class Dataset/DataLoader chuẩn nạp dữ liệu từ file .bin và viết script train.py (tích hợp AdamW, lịch giảm learning rate Cosine Annealing, và Mixed Precision fp16).


Trực tiếp huấn luyện mô hình GPT chính (10–30M tham số) trên văn bản tiếng Việt, theo dõi hàm mất mát (loss) và xuất trọng số (checkpoint.pt) tốt nhất cho Thành viên 3.


Thực hiện mở rộng (Instruction-tuning):



Sử dụng checkpoint GPT đã huấn luyện để tinh chỉnh nhẹ (instruction-tune) trên vài nghìn mẫu hướng dẫn tiếng Việt do Thành viên 1 chuẩn bị.


Thành viên 3: Phụ trách Baseline LSTM, Đánh giá, Sampling & Thí nghiệm Scaling

Cài đặt và huấn luyện Baseline 2 (LSTM Language Model):



Xây dựng mô hình ngôn ngữ LSTM bằng PyTorch (gồm các tầng Embedding  LSTM  Linear) để làm baseline thứ hai bên cạnh N-gram.


Sử dụng chung DataLoader và tập dữ liệu với mô hình GPT để huấn luyện LSTM và lưu lại checkpoint tốt nhất.


Thực hiện thí nghiệm Scaling (Khảo sát kích thước mô hình):



Sử dụng code kiến trúc GPT của Thành viên 2 để chạy huấn luyện ở nhiều mốc kích thước tham số khác nhau (ví dụ: 4M, 12M, 25M) và vẽ biểu đồ khảo sát sự thay đổi của hàm loss khi tăng kích thước mô hình.


Đánh giá Perplexity & Khảo sát thuật toán Sampling:



Viết module tính Perplexity chuẩn trên tập test và lập bảng so sánh tổng hợp giữa 3 mô hình: N-gram, LSTM và GPT.


Viết hàm sinh văn bản nối tiếp từ đoạn văn mở đầu (prompt) với đầu ra là phân phối xác suất token tiếp theo.


Cài đặt và lập bảng so sánh chất lượng văn bản tiếng Việt sinh ra khi thay đổi các tham số giải mã: temperature, top-k, và top-p.


Mốc phối hợp chung cho cả 3 người trong tuần đầu tiên:

Ngày 1–2: Thành viên 1 xuất nhanh một file dữ liệu nháp (khoảng 5MB) kèm bộ Tokenizer. Thành viên 2 dựng sẵn khung GitHub repository và DataLoader.


Ngày 3–7: Cả 3 người tự code module của mình và chạy thử trên tập dữ liệu 5MB để đảm bảo code ghép vào nhau không bị lệch chiều dữ liệu hay lệch bảng từ vựng trước khi cắm máy train toàn bộ tập dữ liệu lớn.


Trong 3 vị trí trên, bạn và 2 thành viên cùng nhóm đã chọn được ai đảm nhận vị trí số 1, số 2 và số 3 chưa?

