import os
import sys
import math
import argparse
import random
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

class NGramLM:
    """
    Mô hình ngôn ngữ N-gram (Bigram, Trigram, 4-gram, 5-gram) với Add-k Smoothing.
    Huấn luyện trực tiếp trên chuỗi Token ID (np.uint16) để so sánh 
    Perplexity công bằng với GPT và LSTM.
    """
    def __init__(self, n: int = 3, k: float = 0.01, vocab_size: int = 16384):
        self.n = n
        self.k = k  # Hệ số làm mịn Add-k (Laplace / Lidstone)
        self.vocab_size = vocab_size
        self.counts = defaultdict(Counter)
        self.context_totals = defaultdict(int)

    def train(self, tokens):
        """Huấn luyện N-gram bằng cách đếm tần suất xuất hiện."""
        n = self.n
        for i in range(len(tokens) - n + 1):
            context = tuple(tokens[i : i + n - 1])
            target = tokens[i + n - 1]
            self.counts[context][target] += 1
            self.context_totals[context] += 1

    def get_prob(self, context: tuple, token: int) -> float:
        """Tính xác suất P(token | context) với Add-k smoothing."""
        count = self.counts[context][token]
        total = self.context_totals[context]
        # P = (count + k) / (total + k * |V|)
        prob = (count + self.k) / (total + self.k * self.vocab_size)
        return prob

    def evaluate_perplexity(self, test_tokens, max_eval_tokens: int = 50000) -> float:
        """
        Tính Perplexity trên tập test:
        PPL = exp(- 1/M * sum(log P(w_i | w_{i-n+1}^{i-1})))
        """
        if len(test_tokens) > max_eval_tokens:
            test_tokens = test_tokens[:max_eval_tokens]

        n = self.n
        total_log_prob = 0.0
        m = 0

        for i in range(len(test_tokens) - n + 1):
            context = tuple(test_tokens[i : i + n - 1])
            target = test_tokens[i + n - 1]
            prob = self.get_prob(context, target)
            total_log_prob += math.log(prob)
            m += 1

        if m == 0:
            return float("inf")

        cross_entropy = - total_log_prob / m
        ppl = math.exp(cross_entropy)
        return ppl

    def generate(self, prompt_tokens: list, max_new_tokens: int = 30, temperature: float = 0.8) -> list:
        """Sinh chuỗi token tiếp theo từ prompt."""
        generated = list(prompt_tokens)
        n = self.n

        for _ in range(max_new_tokens):
            if len(generated) < n - 1:
                context = tuple([0] * (n - 1 - len(generated)) + generated)
            else:
                context = tuple(generated[-(n - 1):])

            counter = self.counts[context]
            if not counter:
                next_token = random.randint(0, self.vocab_size - 1)
            else:
                tokens = list(counter.keys())
                counts = np.array([counter[t] for t in tokens], dtype=np.float64)
                logits = np.log(counts + 1e-10) / max(temperature, 1e-5)
                probs = np.exp(logits - np.max(logits))
                probs /= probs.sum()
                next_token = int(np.random.choice(tokens, p=probs))

            generated.append(next_token)
        return generated

def load_tokens_from_bin(bin_path: str):
    return np.fromfile(bin_path, dtype=np.uint16)

def run_ablation_study(train_tokens, test_tokens, vocab_size: int = 16384):
    """Khảo sát sự thay đổi Perplexity theo bậc N và hệ số làm mịn k."""
    print("\n" + "=" * 60)
    print("      KHẢO SÁT PERPLEXITY CỦA N-GRAM THEO BẬC N VÀ HỆ SỐ k       ")
    print("=" * 60)
    print(f"{'Mô hình':<15} | {'Hệ số k':<10} | {'Test Perplexity':<15}")
    print("-" * 60)

    results = []
    for n in [2, 3, 4, 5]:
        for k in [0.001, 0.01, 0.1]:
            lm = NGramLM(n=n, k=k, vocab_size=vocab_size)
            lm.train(train_tokens)
            ppl = lm.evaluate_perplexity(test_tokens)
            name = f"{n}-gram"
            results.append((name, k, ppl))
            print(f"{name:<15} | {k:<10.3f} | {ppl:<15.2f}")

    print("=" * 60)
    return results

def main():
    parser = argparse.ArgumentParser(description="Baseline 1: N-gram Language Model (Thành viên 1)")
    parser.add_argument("--n", type=int, default=3, help="Bậc của N-gram (mặc định: 3 cho Trigram)")
    parser.add_argument("--k", type=float, default=0.01, help="Hệ số làm mịn Add-k")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Thư mục chứa train.bin & test.bin")
    parser.add_argument("--prompt", type=str, default="Việt Nam là một", help="Prompt tiếng Việt mở đầu")
    parser.add_argument("--ablation", action="store_true", help="Chạy khảo sát so sánh các bậc N-gram cho báo cáo")
    args = parser.parse_args()

    train_bin = Path(args.data_dir) / "train.bin"
    test_bin = Path(args.data_dir) / "test.bin"
    tok_json = Path("data/tokenizer.json")

    if not train_bin.exists() or not test_bin.exists():
        print(f"[!] Không tìm thấy dữ liệu tại {args.data_dir}. Hãy chạy: python prepare_data.py --demo trước!")
        return

    # 1. Nạp dữ liệu
    train_tokens = load_tokens_from_bin(str(train_bin))
    test_tokens = load_tokens_from_bin(str(test_bin))

    # Lấy vocab size từ tokenizer nếu có
    vocab_size = 16384
    tokenizer = None
    if tok_json.exists():
        try:
            from tokenizers import Tokenizer
            tokenizer = Tokenizer.from_file(str(tok_json))
            vocab_size = len(tokenizer.get_vocab())
        except Exception:
            pass

    # 2. Nếu có cờ ablation
    if args.ablation:
        run_ablation_study(train_tokens, test_tokens, vocab_size=vocab_size)
        return

    # 3. Huấn luyện mô hình đơn
    print(f"[*] Đang huấn luyện {args.n}-gram LM (k={args.k}) trên {len(train_tokens):,} tokens...")
    model = NGramLM(n=args.n, k=args.k, vocab_size=vocab_size)
    model.train(train_tokens)

    # 4. Đánh giá Perplexity
    print("[*] Đang đánh giá Perplexity trên tập test...")
    ppl = model.evaluate_perplexity(test_tokens)
    print(f"[+] Kết quả Baseline {args.n}-gram -> Test Perplexity: {ppl:.2f}")

    # 5. Thử nghiệm sinh văn bản
    if tokenizer is not None:
        try:
            prompt_ids = tokenizer.encode(args.prompt).ids
            generated_ids = model.generate(prompt_ids, max_new_tokens=30, temperature=0.7)
            generated_text = tokenizer.decode(generated_ids)
            print("\n[+] Thử nghiệm sinh văn bản:")
            print(f"    Prompt: '{args.prompt}'")
            print(f"    Kết quả {args.n}-gram sinh: '{generated_text}'")
        except Exception as e:
            print(f"[!] Lỗi decode: {e}")

if __name__ == "__main__":
    main()

