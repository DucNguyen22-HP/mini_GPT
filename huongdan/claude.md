
1. Khung công việc chung
Dữ liệu
Chọn một corpus chính, ví dụ Wikipedia tiếng Việt (dễ tải, sạch, vài trăm triệu token). Có thể thêm tập nhỏ như truyện cổ tích hoặc thơ lục bát để so sánh phong cách sinh văn bản.
Làm sạch: bỏ markup, chuẩn hóa Unicode (NFC, rất quan trọng với tiếng Việt), lọc câu quá ngắn hoặc trùng. Chia train/val/test theo tài liệu, không chia theo câu.
Tokenizer
Dùng BPE hoặc SentencePiece, vocab khoảng 8k–32k, train trên chính corpus.
Thử so sánh tokenizer mức âm tiết, BPE và byte-level. Tiếng Việt có nhiều dấu nên kết quả khá thú vị để đưa vào báo cáo.
Mô hình
Baseline 1: n-gram (3-gram, 4-gram, 5-gram) với Kneser-Ney smoothing (dùng KenLM hoặc tự cài).
Baseline 2: LSTM LM 2–3 lớp, kích thước tương đương GPT.
GPT nhỏ: decoder-only Transformer 10–30M tham số, ví dụ 6 lớp, d_model 384, 6 head, context 256–512. Cách nhanh nhất là tự cài theo kiểu nanoGPT.
Đánh giá
Perplexity trên tập test, kèm chất lượng văn bản sinh ra (đánh giá thủ công vài chục mẫu theo thang điểm: trôi chảy, đúng ngữ pháp, mạch lạc).
2. Lưu ý quan trọng khi so sánh
Perplexity chỉ so sánh công bằng được khi các mô hình dùng cùng tokenizer. Nếu khác tokenizer, hãy quy về bits-per-character hoặc bits-per-byte. Đây là điểm dễ bị trừ điểm nếu bỏ qua, nên nói rõ trong báo cáo.
3. Phần scaling (điểm ăn tiền của đề tài)
Train 4–5 kích thước mô hình (ví dụ 1M, 3M, 10M, 30M) với cùng dữ liệu và cùng recipe.
Vẽ val loss theo số tham số (log-log), fit power law L = a·N^(-b) + c.
Thêm thí nghiệm thứ hai: giữ model cố định, thay đổi lượng dữ liệu (10%, 30%, 100%).
Có thể đối chiếu với quy tắc Chinchilla (~20 token/tham số) để giải thích vì sao mô hình lớn bị overfit khi dữ liệu ít.
4. Các hướng mở rộng (chọn 1–2 tùy sức)
So sánh temperature, top-k, top-p trên cùng prompt, đo độ đa dạng (distinct-n) và độ lặp.
Instruction-tune nhẹ trên vài nghìn mẫu (có thể dịch hoặc dùng bộ instruction tiếng Việt có sẵn), so sánh trước và sau.
Ablation: positional encoding (learned vs RoPE), có hoặc không dropout, kích thước context.
So sánh tokenizer (âm tiết vs BPE vs byte) ảnh hưởng thế nào đến loss và chất lượng sinh.
5. Gợi ý phân công (đúng với gợi ý của đề)
Người
Việc chính
A
Thu thập và làm sạch corpus, tokenizer, n-gram baseline
B
Cài đặt và huấn luyện GPT, tối ưu training (mixed precision, LR schedule)
C
LSTM baseline, pipeline đánh giá, thí nghiệm scaling, vẽ biểu đồ

Cả ba cùng làm phần sampling và viết báo cáo. Nên thống nhất sớm một file config và một script đánh giá chung để số liệu so sánh được với nhau.
6. Tài nguyên tính toán
Model 10–30M tham số chạy được trên Kaggle (T4/P100, ~30 giờ GPU/tuần) hoặc Colab. Vì có 3 người, mỗi người có thể dùng một tài khoản Kaggle để chạy song song các cỡ model khác nhau. Nên lưu checkpoint thường xuyên vì session bị giới hạn thời gian.
7. Lộ trình gợi ý
Tuần 1: dữ liệu, tokenizer, n-gram baseline, khung code GPT chạy được trên một tập nhỏ.
Tuần 2: train LSTM và GPT chính, có số perplexity đầu tiên.
Tuần 3: scaling, sampling, mở rộng.
Tuần 4: viết báo cáo, làm slide, chuẩn bị demo sinh văn bản.
Nếu bạn cho mình biết thêm hạn nộp, tài nguyên GPU nhóm có, và mức kinh nghiệm của từng người với PyTorch, mình có thể chốt kế hoạch cụ thể hơn (chọn kích thước model, corpus, và chi tiết phân công). Mình cũng có thể viết sẵn khung code GPT hoặc script xử lý dữ liệu nếu bạn muốn.


CHI TIẾT

Mình đi sâu hơn vào từng phần, kèm số liệu cấu hình, công thức và code mẫu để bạn bắt tay vào làm được ngay.
1. Dữ liệu
Nguồn: Wikipedia tiếng Việt (dump viwiki hoặc bản có sẵn trên HuggingFace datasets). Với nhóm 3 người thì thế là đủ, nên đừng ôm thêm nhiều nguồn ngay từ đầu.
Pipeline làm sạch:
Trích text (WikiExtractor nếu dùng dump XML).
Chuẩn hóa Unicode về NFC (unicodedata.normalize("NFC", s)). Tiếng Việt hay bị lẫn dạng NFD, nếu không chuẩn hóa thì cùng một chữ thành hai chuỗi khác nhau và tokenizer sẽ bị nhiễu.
Bỏ bảng, danh sách tham khảo, đoạn quá ngắn (< 50 ký tự), đoạn trùng lặp.
Chia train/val/test theo bài viết, ví dụ hash tiêu đề rồi lấy 98% / 1% / 1%. Chia theo câu sẽ bị rò rỉ dữ liệu vì các câu cùng bài rất giống nhau.
Lưu thành file .bin (mảng uint16 token id, kiểu nanoGPT) để load nhanh, dùng np.memmap.
Tập phụ (tùy chọn): truyện cổ tích hoặc thơ lục bát chỉ nên dùng để đánh giá out-of-domain (đo perplexity của mô hình train trên Wikipedia khi gặp văn phong khác). Phần này làm báo cáo nhìn có chiều sâu.
2. Tokenizer
import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input="train.txt", model_prefix="vi_bpe",
    vocab_size=16000, model_type="bpe",
    character_coverage=1.0,        # giữ đủ ký tự có dấu
    normalization_rule_name="nfkc", # hoặc identity nếu đã NFC sẵn
    byte_fallback=True)
Vocab 16k là điểm cân bằng tốt cho model nhỏ. Vocab quá lớn làm embedding chiếm gần hết tham số. Ví dụ 32k × 384 = 12M, vượt cả phần Transformer.
Nếu làm thí nghiệm so sánh tokenizer, thử thêm: (a) vocab 8k, (b) mức âm tiết (tách theo khoảng trắng), (c) byte-level. Khi đó bắt buộc dùng BPB để so sánh (xem mục 5).
3. Cấu hình mô hình
Số tham số của Transformer decoder xấp xỉ 12 · n_layer · d_model² (chưa tính embedding). Mình đề xuất 5 cỡ cho thí nghiệm scaling:
Tên
n_layer
d_model
n_head
Non-embedding params
XS
2
128
4
~0.4M
S
4
192
6
~1.8M
M
6
256
8
~4.7M
L
6
384
6
~10.6M
XL
8
512
8
~25M

Cỡ L và XL chính là "10–30 triệu tham số" mà đề yêu cầu. Khi vẽ scaling, dùng số tham số không tính embedding làm trục x (cách của Kaplan et al.), nếu không các model nhỏ sẽ bị lệch vì embedding chiếm phần lớn.
Recipe huấn luyện (giữ cố định cho mọi cỡ, trừ LR):
AdamW, betas (0.9, 0.95), weight decay 0.1, grad clip 1.0
LR đỉnh: 1e-3 cho XS/S, 6e-4 cho M/L, 3e-4 cho XL. Warmup 500–1000 bước, rồi cosine decay xuống 10% đỉnh
Context 256 hoặc 512, batch khoảng 64k–256k token/bước (dùng gradient accumulation nếu thiếu VRAM)
Weight tying giữa embedding và output head
Dropout 0.0–0.1 (dữ liệu lớn so với model nhỏ thì không cần nhiều)
Trên T4/P100 không có bf16, dùng fp16 + GradScaler. Nếu loss ra NaN, hạ LR hoặc bật clip chặt hơn
Ngân sách token: quy tắc Chinchilla ~20 token/tham số, tức XL cần khoảng 500M token. Nếu không đủ thời gian GPU, cứ train mọi cỡ trên cùng số token cố định (ví dụ 200M) và nói rõ trong báo cáo. Quan trọng là so sánh công bằng, không nhất thiết phải compute-optimal. Hãy đo throughput thực tế (token/giây) trên một đoạn chạy thử rồi mới tính được thời gian.
4. Hai baseline
N-gram (KenLM):
Tokenize dữ liệu bằng đúng tokenizer BPE ở trên, ghi ra file mỗi dòng một câu/đoạn gồm các piece cách nhau bằng khoảng trắng.
lmplz -o 5 < train_tok.txt > lm.arpa, sau đó thử order 3, 4, 5 và có/không pruning.
Lưu ý: KenLM tính perplexity có tính </s> và <unk> theo cách riêng. Hãy tự tính lại NLL tổng trên tập test rồi chia theo cùng mẫu số với GPT (tổng số token hoặc tổng số byte) để chắc chắn khớp.
Nếu không muốn phụ thuộc thư viện, tự cài n-gram với Kneser-Ney cũng được, nhưng sẽ chậm hơn nhiều.
LSTM LM:
Embedding 256–512, 2–3 lớp LSTM hidden 512–1024, dropout 0.3, weight tying, BPTT 256.
Chỉnh sao cho số tham số xấp xỉ GPT cỡ L hoặc XL để so sánh công bằng ("cùng ngân sách tham số"). Có thể báo cáo thêm một LSTM nhỏ hơn.
Optimizer Adam, LR 1e-3, clip grad 0.25–1.0. LSTM thường chậm hơn GPT trên GPU vì tính tuần tự, nên chú ý thời gian.
5. Đánh giá (phần dễ sai nhất)
Perplexity theo token: PPL = exp(mean_NLL_per_token). Chỉ so sánh được giữa các mô hình dùng cùng tokenizer.
Bits-per-byte (so sánh mọi tokenizer):
import math
# total_nll_nats: tổng NLL (nat) trên toàn bộ tập test
# total_bytes: tổng số byte UTF-8 của văn bản test gốc
bpb = total_nll_nats / (math.log(2) * total_bytes)
Dùng byte thay vì ký tự vì tiếng Việt có ký tự nhiều byte, và byte là đơn vị trung lập giữa các tokenizer. Nên báo cáo cả PPL và BPB.
Đánh giá văn bản sinh (định lượng):
distinct-1/2/3 (tỉ lệ n-gram duy nhất, đo độ đa dạng)
Tỉ lệ lặp (phần trăm 4-gram bị lặp trong một mẫu)
Self-BLEU giữa các mẫu cùng prompt (cao nghĩa là các mẫu giống nhau)
Đánh giá thủ công: 20–30 prompt × vài cấu hình, mỗi người trong nhóm chấm riêng 3 tiêu chí (trôi chảy, đúng ngữ pháp, mạch lạc) theo thang 1–5, che tên mô hình khi chấm (blind). Tính độ đồng thuận giữa người chấm nếu muốn thêm điểm.
6. Thí nghiệm scaling
(a) Scaling theo kích thước mô hình: 5 cỡ ở mục 3, cùng dữ liệu và số token, ghi val loss cuối. Vẽ log-log rồi fit:
from scipy.optimize import curve_fit
import numpy as np

def power_law(N, a, b, c):
    return a * N ** (-b) + c   # c = loss không thể giảm thêm

popt, _ = curve_fit(power_law, N_nonemb, val_loss,
                    p0=[10, 0.3, 2.0], maxfev=20000)
Với chỉ 5 điểm, fit sẽ không chắc chắn. Hãy nói thẳng điều đó, và nên báo cáo khoảng tin cậy hoặc chạy thêm 2 seed cho các cỡ nhỏ để thấy độ nhiễu.
(b) Scaling theo dữ liệu: giữ model L cố định, train trên 10% / 30% / 100% dữ liệu. Vẽ val loss và train-val gap. Cho thấy khi dữ liệu ít thì model lớn bắt đầu overfit.
(c) (Nếu còn thời gian) Learning curve: vẽ val loss theo số bước cho từng cỡ trên cùng một đồ thị. Thường đủ để minh họa vì sao model lớn học hiệu quả hơn trên mỗi token.
7. Phần sampling
Sinh văn bản từ khoảng 20–30 prompt mở đầu (bạn tự soạn, đa dạng chủ đề: lịch sử, khoa học, đời sống, văn học). Bảng cấu hình gợi ý:
Temperature ∈ {0.5, 0.8, 1.0, 1.2} (greedy làm mốc)
Top-k ∈ {10, 50}
Top-p ∈ {0.9, 0.95}
Với mỗi cấu hình, sinh ~5 mẫu mỗi prompt (độ dài 100–200 token), tính các chỉ số ở mục 5. Kết quả điển hình: temperature thấp thì lặp, temperature cao thì mất mạch lạc, top-p thường cân bằng tốt nhất. Nhưng hãy để dữ liệu của nhóm tự nói.
8. Mở rộng instruction-tune (nếu chọn)
Tìm bộ Alpaca/instruction tiếng Việt trên HuggingFace (vài nghìn đến vài chục nghìn mẫu), lấy khoảng 2–5k mẫu sạch.
Định dạng mẫu: ### Câu hỏi: ... ### Trả lời: ..., chỉ tính loss trên phần trả lời (mask phần prompt).
Fine-tune LR nhỏ (1e-4 → 5e-5), 2–3 epoch.
Hãy kỳ vọng thực tế: model 10–30M tham số, pretrain trên Wikipedia, sẽ học được định dạng hỏi–đáp nhưng chất lượng nội dung sẽ yếu. Đó vẫn là một kết quả đáng báo cáo, miễn là bạn trình bày trung thực (so sánh trước/sau bằng chấm thủ công).
9. Các lỗi hay gặp
Quên NFC, hoặc tokenizer train cả trên val/test (rò rỉ).
So sánh PPL giữa các tokenizer khác nhau.
Tăng kích thước model nhưng quên đổi LR, làm model lớn trông tệ giả tạo.
Đếm cả embedding vào số tham số khi vẽ scaling.
Chỉ chạy 1 seed rồi kết luận chênh lệch nhỏ là có ý nghĩa.
Không lưu checkpoint thường xuyên, session Kaggle/Colab hết hạn là mất sạch.
Mỗi người dùng một cấu hình eval khác nhau, số liệu không ghép lại được. Hãy chốt một script eval chung từ tuần đầu.
10. Cấu trúc báo cáo gợi ý
Giới thiệu và mục tiêu
Dữ liệu và tiền xử lý (thống kê: số bài, số token, độ dài trung bình)
Tokenizer
Mô hình: n-gram, LSTM, GPT (bảng cấu hình, số tham số)
Thiết lập huấn luyện và đánh giá (nhấn mạnh PPL vs BPB)
Kết quả: so sánh 3 mô hình, scaling (kèm đồ thị), sampling
Phân tích lỗi: ví dụ văn bản sinh tốt và xấu, lỗi dấu/âm tiết đặc trưng tiếng Việt
Hạn chế và hướng phát triển
Phân công đóng góp từng thành viên
11. Timeline chi tiết hơn
Giai đoạn
Việc cần xong
Sản phẩm
Ngày 1–3
Tải và làm sạch dữ liệu, train tokenizer, chốt cấu trúc repo và script eval
train.bin, val.bin, test.bin
Ngày 4–7
GPT chạy được trên 10% dữ liệu, KenLM n-gram có PPL, LSTM chạy thử
Số PPL đầu tiên của cả 3 mô hình
Tuần 2
Train đầy đủ GPT cỡ L, LSTM, n-gram; bắt đầu chạy các cỡ nhỏ
Bảng so sánh chính
Tuần 3
Hoàn tất scaling (model và data), sampling, chấm thủ công
Đồ thị và bảng chỉ số
Tuần 4
Mở rộng (instruction-tune hoặc ablation), viết báo cáo, slide, demo
Báo cáo và demo

Để mình chốt kế hoạch sát hơn (chọn cỡ model, ngân sách token, chia việc cụ thể), cho mình biết nhóm bạn có hạn nộp là khi nào và dùng GPU gì (Kaggle, Colab hay máy cá nhân)?

