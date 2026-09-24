import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from data import get_loaders
from model import TransformerTTS, make_pad_mask

# ---------------- config ----------------
CFG = {
    "n_mels": 80,
    "n_embd": 256,
    "n_head": 4,
    "n_enc_layers": 3,
    "n_dec_layers": 3,
    "dropout": 0.1,
    "batch_size": 16,
    "lr": 1e-3,  # peak LR after warmup
    "warmup": 4000,
    "max_steps": 200_000,
    "grad_clip": 1.0,
    "stop_pos_weight": 8.0,
    "ga_weight": 1.0,  # guided attention
    "ga_sigma": 0.2,
    "log_every": 100,
    "eval_every": 2000,
    "ckpt_every": 5000,
    "out_dir": "runs/tts",
}
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)


def masked_l1(pred, target, mel_lens):

    frame_mask = (~make_pad_mask(mel_lens, pred.size(1))).unsqueeze(-1).float()
    err = (pred - target).abs() * frame_mask
    return err.sum() / (frame_mask.sum() * pred.size(-1)).clamp(min=1)


def stop_loss(stop_logits, stop_target, mel_lens, pos_weight):
    pos_weight = torch.tensor(pos_weight, device=stop_logits.device)
    err = F.binary_cross_entropy_with_logits(
        stop_logits, stop_target, reduction="none", pos_weight=pos_weight
    )
    mask = (~make_pad_mask(mel_lens, stop_logits.size(1))).float()
    return (err * mask).sum() / mask.sum().clamp(min=1)


def guided_attention_loss(attn, text_lens, mel_lens, sigma):
    B, H, T, N = attn.shape
    dev = attn.device
    t = (
        torch.arange(T, device=dev)[None, :, None] / mel_lens[:, None, None]
    )  # (B, T, 1)
    n = (
        torch.arange(N, device=dev)[None, None, :] / text_lens[:, None, None]
    )  # (B, 1, N)
    W = 1 - torch.exp(-((n - t) ** 2) / (2 * sigma**2))  # (B, T, N)

    valid = (~make_pad_mask(mel_lens, T))[:, :, None] & (~make_pad_mask(text_lens, N))[
        :, None, :
    ]
    W = W * valid

    return (attn * W[:, None]).sum() / (valid.sum() * H).clamp(min=1)


def shift_right(mel):
    # teacher forcing: decoder sees a zero "go" frame then frames 0..T-2
    return F.pad(mel, (0, 0, 1, 0))[:, :-1]


def lr_at(step):
    # linear warmup, then inverse-sqrt decay (Noam-style)
    step = max(step, 1)
    return CFG["lr"] * min(step / CFG["warmup"], math.sqrt(CFG["warmup"] / step))


# ---------------- helpers ----------------
def save_attention_plot(attn, path):
    # attn: (T_mel, T_text) for one example, one head (or mean over heads)
    plt.figure(figsize=(6, 4))
    plt.imshow(attn.T.cpu(), aspect="auto", origin="lower", interpolation="none")
    plt.xlabel("decoder step (mel frame)")
    plt.ylabel("encoder step (character)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_ckpt(model, opt, step, path):
    torch.save(
        {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "step": step,
            "cfg": CFG,
        },
        path,
    )


@torch.no_grad()
def evaluate(model, val_dl, out_dir, step):
    model.eval()
    total, n = 0.0, 0
    for i, (text, text_lens, mel, mel_lens, stop) in enumerate(val_dl):
        text, text_lens, mel, mel_lens = (
            t.to(DEVICE) for t in (text, text_lens, mel, mel_lens)
        )
        mel_pre, mel_post, stop_logits, attns = model(text, text_lens, shift_right(mel))
        total += masked_l1(mel_post, mel, mel_lens).item()
        n += 1
        if (
            i == 0
        ):  # alignment plot for the first val example, last decoder layer, mean over heads
            a = attns[-1][0].mean(0)[: mel_lens[0], : text_lens[0]]
            save_attention_plot(a, out_dir / f"align_{step:07d}.png")
    model.train()
    return total / max(n, 1)


# ---------------- main loop ----------------
def main():
    out_dir = Path(CFG["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dl, val_dl, stoi, itos = get_loaders(batch_size=CFG["batch_size"])
    model = TransformerTTS(
        vocab_size=len(stoi) + 2,
        n_mels=CFG["n_mels"],
        n_embd=CFG["n_embd"],
        n_head=CFG["n_head"],
        n_enc_layers=CFG["n_enc_layers"],
        n_dec_layers=CFG["n_dec_layers"],
        dropout=CFG["dropout"],
    ).to(DEVICE)
    opt = torch.optim.Adam(
        model.parameters(), lr=CFG["lr"], betas=(0.9, 0.98), eps=1e-9
    )
    print(
        f"device={DEVICE}  params={sum(p.numel() for p in model.parameters())/1e6:.1f}M"
    )

    step = 0
    model.train()
    while step < CFG["max_steps"]:
        for text, text_lens, mel, mel_lens, stop in train_dl:
            text, text_lens, mel, mel_lens, stop = (
                t.to(DEVICE) for t in (text, text_lens, mel, mel_lens, stop)
            )
            for g in opt.param_groups:
                g["lr"] = lr_at(step)

            mel_pre, mel_post, stop_logits, attns = model(
                text, text_lens, shift_right(mel)
            )

            l_pre = masked_l1(mel_pre, mel, mel_lens)
            l_post = masked_l1(mel_post, mel, mel_lens)
            l_stop = stop_loss(stop_logits, stop, mel_lens, CFG["stop_pos_weight"])
            l_ga = guided_attention_loss(
                attns[-1], text_lens, mel_lens, CFG["ga_sigma"]
            )
            loss = l_pre + l_post + l_stop + CFG["ga_weight"] * l_ga

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            opt.step()
            step += 1

            if step % CFG["log_every"] == 0:
                print(
                    f"step {step:7d} | loss {loss.item():.4f} | pre {l_pre.item():.4f} "
                    f"post {l_post.item():.4f} stop {l_stop.item():.4f} ga {l_ga.item():.4f} "
                    f"| lr {lr_at(step):.2e}"
                )
            if step % CFG["eval_every"] == 0:
                val = evaluate(model, val_dl, out_dir, step)
                print(f"  val L1 (post) {val:.4f}  -> align_{step:07d}.png")
            if step % CFG["ckpt_every"] == 0:
                save_ckpt(model, opt, step, out_dir / "latest.pt")
            if step >= CFG["max_steps"]:
                break


if __name__ == "__main__":
    main()
