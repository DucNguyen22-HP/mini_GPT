import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer
from torch.nn import functional as F

from dataset import MemmapDataset
from models.gpt import GPT


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def learning_rate(step, settings):
    warmup = settings["warmup_steps"]
    peak = settings["learning_rate"]
    floor = settings["min_learning_rate"]
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup - 1) / max(1, settings["max_steps"] - warmup - 1)
    return floor + (peak - floor) * (1 + math.cos(math.pi * progress)) / 2


def validate_settings(settings):
    for key in ("batch_size", "accumulation_steps", "max_steps", "eval_every", "log_every"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    warmup = settings["warmup_steps"]
    if type(warmup) is not int or not 0 <= warmup < settings["max_steps"]:
        raise ValueError("warmup_steps must be an integer in [0, max_steps)")
    if not 0 < settings["min_learning_rate"] <= settings["learning_rate"]:
        raise ValueError("Expected 0 < min_learning_rate <= learning_rate")
    if settings["grad_clip"] <= 0 or settings["weight_decay"] < 0:
        raise ValueError("Expected grad_clip > 0 and weight_decay >= 0")
    if settings["precision"] not in ("fp16", "fp32"):
        raise ValueError("precision must be fp16 or fp32")


def validation_batches(dataset, batch_size):
    data = dataset.data
    block = dataset.block_size
    full, tail = divmod(len(data) - 1, block)
    for start in range(0, full, batch_size):
        offsets = np.arange(start, min(start + batch_size, full)) * block
        indices = offsets[:, None] + np.arange(block + 1)
        yield torch.from_numpy(data[indices].astype(np.int64))
    if tail:
        yield torch.from_numpy(data[full * block:].astype(np.int64)).unsqueeze(0)


@torch.no_grad()
def evaluate(model, dataset, batch_size, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in validation_batches(dataset, batch_size):
        batch = batch.to(device)
        targets = batch[:, 1:]
        logits = model(batch[:, :-1])
        total_loss += F.cross_entropy(
            logits.flatten(0, 1), targets.reshape(-1), reduction="sum"
        ).item()
        total_tokens += targets.numel()
    return total_loss / total_tokens, total_tokens


def save_checkpoint(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def sampler_state(device):
    algorithm, keys, position, has_gaussian, cached = np.random.get_state()
    return {
        "numpy": (algorithm, keys.tolist(), position, has_gaussian, cached),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_sampler(state, device):
    algorithm, keys, position, has_gaussian, cached = state["numpy"]
    np.random.set_state((algorithm, np.array(keys, dtype=np.uint32), position, has_gaussian, cached))
    torch.set_rng_state(state["torch"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)


def trim_metrics(path, last_step):
    if not path.exists():
        return
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            if row["step"] > last_step:
                break
            rows.append(line)
    temporary = path.with_suffix(".tmp")
    temporary.write_text("".join(rows), encoding="utf-8")
    temporary.replace(path)


def model_checkpoint(model, config, fingerprints, vocab_size, step, val_loss):
    return {
        "model_state": model.state_dict(),
        "config": config,
        "fingerprints": fingerprints,
        "vocab_size": vocab_size,
        "step": step,
        "val_loss": val_loss,
    }


def train(config, run_dir, device, resume=False, stop_after=None):
    settings = config["training"]
    validate_settings(settings)
    end_step = settings["max_steps"] if stop_after is None else stop_after
    if not 1 <= end_step <= settings["max_steps"]:
        raise ValueError("stop_after must be in [1, max_steps]")
    np.random.seed(settings["seed"])
    torch.manual_seed(settings["seed"])
    fingerprints = {key: file_hash(path) for key, path in config["data"].items()}
    tokenizer = Tokenizer.from_file(config["data"]["tokenizer"])
    vocab_size = tokenizer.get_vocab_size()
    block = config["model"]["block_size"]
    train_data = MemmapDataset(config["data"]["train"], block)
    val_data = MemmapDataset(config["data"]["validation"], block)
    if len(train_data) == 0 or val_data.total_tokens < 2:
        raise ValueError("Training needs more than block_size tokens and validation needs at least two")
    if max(train_data.data.max(), val_data.data.max()) >= vocab_size:
        raise ValueError("A token ID in the data is outside the tokenizer vocabulary")

    model = GPT(vocab_size=vocab_size, **config["model"]).to(device)
    decay, no_decay = [], []
    for param in model.parameters():
        (decay if param.ndim >= 2 else no_decay).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": settings["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=settings["learning_rate"],
        betas=(0.9, 0.95),
    )
    use_amp = device.type == "cuda" and settings["precision"] == "fp16"
    scaler = (
        torch.amp.GradScaler("cuda", enabled=use_amp)
        if hasattr(torch.amp, "GradScaler")
        else torch.cuda.amp.GradScaler(enabled=use_amp)
    )
    step, best_step, best_loss = 0, 0, math.inf
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    log_path = run_dir / "metrics.jsonl"
    if resume:
        state = torch.load(last_path, map_location="cpu", weights_only=True)
        if (
            state["config"] != config
            or state["fingerprints"] != fingerprints
            or state["device_type"] != device.type
        ):
            raise ValueError("Resume requires the same config, data and device type")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scaler.load_state_dict(state["scaler_state"])
        restore_sampler(state["rng"], device)
        step, best_step, best_loss = state["step"], state["best_step"], state["best_loss"]
        if step == best_step:
            save_checkpoint(model_checkpoint(model, config, fingerprints, vocab_size, step, best_loss), best_path)
        trim_metrics(log_path, step)
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    print(f"GPT: {sum(p.numel() for p in model.parameters()):,} parameters; "
          f"vocab={vocab_size}; device={device}; precision={'fp16' if use_amp else 'fp32'}", flush=True)
    batch_size = settings["batch_size"]
    microsteps = settings["accumulation_steps"]
    while step < end_step:
        next_step = step + 1
        lr = learning_rate(next_step, settings)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        optimizer.zero_grad(set_to_none=True)
        mean_loss = torch.zeros((), device=device)
        for _ in range(microsteps):
            x, y = train_data.get_batch(batch_size, str(device))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = F.cross_entropy(logits.flatten(0, 1), y.flatten()) / microsteps
            scaler.scale(loss).backward()
            mean_loss += loss.detach()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), settings["grad_clip"], error_if_nonfinite=not use_amp
        )
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < old_scale:
            if scaler.get_scale() < 1e-4:
                raise FloatingPointError("FP16 gradients remain nonfinite; inspect data and settings")
            print(f"FP16 overflow at step {next_step}; retrying with a lower scale", flush=True)
            continue

        step = next_step
        row = {"step": step, "tokens_seen": step * batch_size * microsteps * block, "lr": lr}
        validate = step % settings["eval_every"] == 0 or step == settings["max_steps"]
        improved = False
        if validate:
            val_loss, val_tokens = evaluate(model, val_data, batch_size, device)
            if not math.isfinite(val_loss):
                raise FloatingPointError("Validation loss is nonfinite")
            row.update(val_loss=val_loss, val_tokens=val_tokens)
            improved = val_loss < best_loss
            if improved:
                best_step, best_loss = step, val_loss
        if step % settings["log_every"] == 0 or validate or step == end_step:
            row.update(train_loss=mean_loss.item(), grad_norm=grad_norm.item())
            line = json.dumps(row, allow_nan=False)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(line + "\n")
            print(line, flush=True)
        if validate or step == end_step:
            snapshot = model_checkpoint(model, config, fingerprints, vocab_size, step, row.get("val_loss"))
            save_checkpoint({
                **snapshot,
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "rng": sampler_state(device),
                "device_type": device.type,
                "best_step": best_step,
                "best_loss": best_loss,
            }, last_path)
            if improved:
                save_checkpoint(snapshot, best_path)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/gpt_main.yaml"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/gpt_main"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        parser.error("--device must be auto, cpu or a CUDA device")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train(config, args.run_dir, device, args.resume, args.stop_after)


if __name__ == "__main__":
    main()
