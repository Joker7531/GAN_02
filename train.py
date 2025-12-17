from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from eeg_dataset import EEGWindowDataset, make_window_index, split_indices
from losses import HybridLossConfig, HybridTimeFreqLoss, MRSTFTConfig
from models import WaveUNet1D, WaveUNetConfig


@torch.no_grad()
def save_best_preview(
    model: torch.nn.Module,
    sample_x: torch.Tensor,
    sample_y: torch.Tensor,
    device: str,
    out_path: Path,
) -> None:
    """Save a 3-sample visualization (raw input / clean target / model pred)."""
    try:
        import matplotlib  # type: ignore[import-not-found]

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for visualization. Install it via 'pip install matplotlib'.") from e

    model.eval()
    x = sample_x.to(device)
    y = sample_y.to(device)
    y_hat = model(x)

    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    yhat_np = y_hat.detach().cpu().numpy()

    n = min(3, x_np.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(12, 3.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    t = np.arange(x_np.shape[-1])
    for i in range(n):
        ax = axes[i]
        ax.plot(t, x_np[i, 0], label="raw(in)", linewidth=1.0)
        ax.plot(t, y_np[i, 0], label="clean(gt)", linewidth=1.0)
        ax.plot(t, yhat_np[i, 0], label="pred", linewidth=1.0)
        ax.set_title(f"Sample {i}")
        ax.grid(True, alpha=0.2)
        if i == 0:
            ax.legend(loc="upper right")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EEG Wave-U-Net denoising (L1 + Multi-Res STFT)")
    p.add_argument("--npz", type=str, default="/root/autodl-tmp/dataset_cz_v2.npz")
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--stride", type=int, default=1024)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num-workers", type=int, default=0)

    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--base-ch", type=int, default=32)

    p.add_argument("--l1", type=float, default=1.0, help="weight for time-domain L1")
    p.add_argument("--stft", type=float, default=1.0, help="weight for MR-STFT")
    p.add_argument("--fft-sizes", type=int, nargs="+", default=[512, 1024, 2048])

    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default="runs/eeg_wavunet")
    return p.parse_args()


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, loss_fn: torch.nn.Module, device: str) -> float:
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        total += float(loss.item()) * x.size(0)
        n += x.size(0)
    return total / max(1, n)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    obj = np.load(args.npz, allow_pickle=True)
    data = {k: obj[k] for k in obj.files}

    indices = make_window_index(data["section_id"], window_size=args.window, stride=args.stride)
    train_idx, test_idx = split_indices(indices, train_ratio=args.train_ratio, seed=args.seed)

    train_ds = EEGWindowDataset(
        npz_path=None,
        data=data,
        indices=train_idx,
        window_size=args.window,
        normalization="per_sample_zscore",
    )
    test_ds = EEGWindowDataset(
        npz_path=None,
        data=data,
        indices=test_idx,
        window_size=args.window,
        normalization="per_sample_zscore",
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(args.device.startswith("cuda")),
        drop_last=False,
    )

    model = WaveUNet1D(WaveUNetConfig(base_channels=args.base_ch, depth=args.depth)).to(args.device)

    loss_fn = HybridTimeFreqLoss(
        stft_cfg=MRSTFTConfig(fft_sizes=tuple(args.fft_sizes)),
        cfg=HybridLossConfig(l1_weight=args.l1, stft_weight=args.stft),
    ).to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.9))

    # Fixed 3 samples for visualization (deterministic under seed)
    rng = np.random.default_rng(args.seed)
    vis_n = min(3, len(test_ds))
    vis_ids = rng.choice(len(test_ds), size=vis_n, replace=False) if vis_n > 0 else []
    if vis_n > 0:
        xs, ys = [], []
        for i in vis_ids:
            x_i, y_i = test_ds[int(i)]
            xs.append(x_i)
            ys.append(y_i)
        vis_x = torch.stack(xs, dim=0)
        vis_y = torch.stack(ys, dim=0)
    else:
        vis_x = None
        vis_y = None

    best = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for x, y in pbar:
            x = x.to(args.device)
            y = y.to(args.device)

            y_hat = model(x)
            loss = loss_fn(y_hat, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pbar.set_postfix(loss=float(loss.item()))

        val = evaluate(model, test_loader, loss_fn, args.device)
        print(f"val_loss: {val:.6f}")

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "val_loss": val,
            "args": vars(args),
        }
        torch.save(ckpt, out_dir / "last.pt")
        if val < best:
            best = val
            torch.save(ckpt, out_dir / "best.pt")
            if vis_x is not None and vis_y is not None:
                save_best_preview(
                    model=model,
                    sample_x=vis_x,
                    sample_y=vis_y,
                    device=args.device,
                    out_path=out_dir / f"best_preview_epoch{epoch:03d}.png",
                )
                # Also keep a stable filename for the latest best
                save_best_preview(
                    model=model,
                    sample_x=vis_x,
                    sample_y=vis_y,
                    device=args.device,
                    out_path=out_dir / "best_preview.png",
                )


if __name__ == "__main__":
    main()
