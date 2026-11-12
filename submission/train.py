"""Train the chunked BC policy on an exported demo npz. Single GPU (RTX 4090).

uv run python -m submission.train --data datasets/move_plate.npz --out checkpoints/bc
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .model import ChunkPolicy, DiscreteChunkPolicy, DiscreteDiTPolicy, action_limits, arm_ref


def soft_ce(logits, target, bins, tau):
    """Cross-entropy with distance-smoothed targets over uniform bins (tau in normalised units)."""
    centers = (torch.arange(bins, device=target.device) + 0.5) * 2 / bins - 1
    w = torch.exp(-(target[..., None] - centers).abs() / tau)
    w = w / w.sum(-1, keepdim=True)
    return -(w * F.log_softmax(logits, dim=-1)).sum(-1).mean()


def random_shift(x, pad=6):
    """x: (B,cam,3,H,W) uint8->float; independent random translation per sample/camera."""
    b, c, ch, h, w = x.shape
    x = x.flatten(0, 1)
    xp = F.pad(x, (pad,) * 4, mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (b * c,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (b * c,), device=x.device)
    idx = torch.arange(h, device=x.device)
    rows = (oy[:, None] + idx)[:, :, None].expand(-1, h, w)
    cols = (ox[:, None] + torch.arange(w, device=x.device))[:, None, :].expand(-1, h, w)
    out = xp[torch.arange(b * c, device=x.device)[:, None, None], :, rows, cols]
    return out.permute(0, 3, 1, 2).reshape(b, c, ch, h, w)


def load_data(demo_npz, expert_dir, max_expert=0):
    """Human demos (one npz) plus expert episodes (ep_*.npz) -> flat arrays.

    Preallocates the big image array (no concatenate copy): peak RAM = final size.
    """
    sources = []  # (kind, handle, length, episode_id)
    if demo_npz:
        d = np.load(demo_npz)
        sources.append(("demo", d, len(d["action"]), None))
    files = sorted(Path(expert_dir).glob("*.npz")) if expert_dir else []
    if max_expert:
        files = files[:max_expert]
    for i, f in enumerate(files):
        e = np.load(f)
        sources.append(("expert", e, len(e["action"]), 100_000 + i))
    total = sum(s[2] for s in sources)
    out = {
        "rgb": np.empty((total, 3, 3, 84, 84), np.uint8),
        "state": np.empty((total, 50), np.float32),
        "action": np.empty((total, 20), np.float32),
        "episode": np.empty(total, np.int64),
    }
    at = 0
    for kind, h, n, ep in sources:
        for k in ("rgb", "state", "action"):
            out[k][at : at + n] = h[k]
        out["episode"][at : at + n] = h["episode"] if kind == "demo" else ep
        at += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="datasets/move_plate.npz", help="human demos ('' to skip)")
    ap.add_argument("--expert-dir", default="", help="dir of ep_*.npz from generate_data")
    ap.add_argument(
        "--head",
        choices=["continuous", "discrete", "dit-dr", "dit-df"],
        default="continuous",
        help="dit-dr / dit-df: paper-faithful DiT discrete heads (arXiv:2610.20429)",
    )
    ap.add_argument("--loss", choices=["l1", "mse"], default="l1", help="continuous head loss")
    ap.add_argument("--packed", default="", help="dir from pack_data.py (memory-mapped, shared)")
    ap.add_argument("--boost-slots", default="", help="e.g. 3,4,5: oversample these source slots")
    ap.add_argument("--boost", type=float, default=1.0, help="sampling weight for boosted slots")
    ap.add_argument(
        "--relative",
        action="store_true",
        help="relative arm-joint actions (target minus current pose), as in arXiv:2610.20429",
    )
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay (paper: 0.995)")
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=12)
    ap.add_argument("--bins", type=int, default=256)
    ap.add_argument(
        "--tau", type=float, default=0.0025, help="soft-label kernel (normalised units)"
    )
    ap.add_argument("--cond-drop", type=float, default=0.1, help="null-condition prob (for CFG)")
    ap.add_argument("--init", default="", help="warm-start from a checkpoint's weights")
    ap.add_argument("--max-expert", type=int, default=0, help="cap expert episodes (RAM)")
    ap.add_argument("--out", default="checkpoints/bc")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-episodes", type=int, default=6, help="held-out demos for val loss")
    ap.add_argument("--pretrained", action="store_true", help="ImageNet ResNet18 init")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.RandomState(args.seed)
    dev = "cuda"
    if args.packed:
        pk = Path(args.packed)
        rgb = np.load(pk / "rgb.npy", mmap_mode="r")  # shared page cache across trainings
        d = {k: np.load(pk / f"{k}.npy") for k in ("state", "action", "episode")}
    else:
        d = load_data(args.data, args.expert_dir, args.max_expert)
        rgb = torch.from_numpy(d["rgb"])  # uint8 in host RAM (~63 KB/step)
    state = torch.from_numpy(d["state"]).to(dev)
    action = torch.from_numpy(d["action"]).to(dev)
    episode = d["episode"]
    eps = np.unique(episode)
    val_eps = set(rng.choice(eps, args.val_episodes, replace=False)) if args.val_episodes else set()
    s_mean, s_std = state.mean(0), state.std(0).clamp_min(1e-3)
    lo_t, hi_t = (torch.tensor(v, device=dev, dtype=torch.float32) for v in action_limits())
    rel_range = torch.ones(20, device=dev)
    if args.relative:  # per-dim half-range of (target - current pose): 1.2 x p99.9, clipped
        sub = np.arange(0, len(d["state"]), 7)
        ref_np = arm_ref(torch.from_numpy(d["state"][sub]).to(dev), lo_t, hi_t).cpu().numpy()
        ep_np = d["episode"]
        last = {e: np.where(ep_np == e)[0][-1] for e in np.unique(ep_np)}
        ends_sub = np.array([last[e] for e in ep_np[sub]])
        dl = np.concatenate(
            [d["action"][np.minimum(sub + k, ends_sub)] - ref_np for k in (0, 5, 10, 19)]
        )
        q = np.percentile(np.abs(dl[:, 4:18]), 99.9, axis=0)
        rel_range[4:18] = torch.from_numpy(np.clip(1.2 * q, 0.02, 0.5)).to(dev).float()
        print("relative ranges (arm dims):", rel_range[4:18].cpu().numpy().round(3), flush=True)

    # Valid chunk starts; the tail of an episode repeats its last action.
    H = args.horizon
    ends = {e: np.where(episode == e)[0][-1] for e in eps}
    chunk_idx = np.arange(H)

    def make_idx(mask_eps):
        return np.array([i for i in range(len(episode)) if (episode[i] in mask_eps)])

    train_idx = make_idx(set(eps) - val_eps)
    probs = None
    if args.boost > 1 and args.boost_slots and args.packed:
        slots = np.load(Path(args.packed) / "slot.npy")[train_idx]
        hot = np.isin(slots, [int(x) for x in args.boost_slots.split(",")])
        probs = np.where(hot, args.boost, 1.0)
        probs = probs / probs.sum()
    val_idx = make_idx(val_eps) if val_eps else np.array([], dtype=int)
    end_of = torch.from_numpy(np.array([ends[e] for e in episode])).to(dev)

    def batch(ix):
        ix = np.sort(np.asarray(ix))  # sorted: cheap memmap reads; order is irrelevant
        ix_cpu = torch.as_tensor(ix)
        ixd = ix_cpu.to(dev)
        t = torch.minimum(ixd[:, None] + torch.as_tensor(chunk_idx, device=dev), end_of[ixd, None])
        imgs = rgb[ix] if isinstance(rgb, np.ndarray) else rgb[ix_cpu]
        imgs = torch.as_tensor(np.ascontiguousarray(imgs)) if isinstance(imgs, np.ndarray) else imgs
        ref = arm_ref(state[ixd], lo_t, hi_t) if args.relative else None
        return imgs.to(dev, non_blocking=True), (state[ixd] - s_mean) / s_std, action[t], ref

    def to_rel(a, ref):  # absolute -> relative, scaled into [-1, 1] for the quantiser
        if ref is None:
            return a
        return ((a - ref[:, None, :]) / rel_range).clamp(-1, 1)

    def to_abs(p, ref):
        return p if ref is None else p * rel_range + ref[:, None, :]

    if args.head in ("dit-dr", "dit-df"):
        model = DiscreteDiTPolicy(
            H,
            bins=args.bins,
            width=args.width,
            blocks=args.blocks,
            mode=args.head[-2:],
            pretrained=args.pretrained,
        ).to(dev)
    elif args.head == "discrete":
        model = DiscreteChunkPolicy(H, bins=args.bins, pretrained=args.pretrained).to(dev)
    else:
        model = ChunkPolicy(H, pretrained=args.pretrained).to(dev)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["model"])
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, args.lr, total_steps=args.steps, pct_start=0.05
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def save(tag):
        tmp = out / f"{tag}.pt.tmp"
        torch.save(
            {
                "model": ema.state_dict(),
                "horizon": H,
                "head": args.head,
                "bins": args.bins,
                "relative": args.relative,
                "rel_range": rel_range.cpu(),
                "width": args.width,
                "blocks": args.blocks,
                "state_mean": s_mean.cpu(),
                "state_std": s_std.cpu(),
                "args": vars(args),
            },
            tmp,
        )
        tmp.replace(out / f"{tag}.pt")  # atomic: a full disk can no longer leave a corrupt file

    def evaluate():
        if not len(val_idx):
            return float("nan")
        vi = val_idx[:: 8 if args.head == "dit-df" else 1]  # DF sampling is slow: subsample
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(vi), 128):
                r, s, a, ref = batch(vi[i : i + 128])
                pred = to_abs(ema.act(r.float() / 255, s), ref)
                tot += F.l1_loss(pred, a, reduction="sum").item()
        return tot / (len(vi) * H * 20)

    log = []
    for step in range(1, args.steps + 1):
        r, s, a, ref = batch(rng.choice(train_idx, args.batch, p=probs))
        a = to_rel(a, ref)
        r = random_shift(r.float()) / 255
        s = s + 0.05 * torch.randn_like(s)  # proprioception noise reduces copycat reliance
        drop = torch.rand(len(s), device=dev) < args.cond_drop
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if args.head in ("dit-dr", "dit-df"):
                n = len(s)
                z = ((a + 1) / 2 * args.bins).long().clamp(0, args.bins - 1).view(n, -1)
                noise = torch.randint(0, args.bins, z.shape, device=dev)
                if args.head == "dit-dr":
                    z_in, t = noise, torch.zeros(n, device=dev)
                else:  # discrete flow, mixture path alpha(t) = t^3, uniform source
                    t = torch.rand(n, device=dev)
                    keep = torch.rand(z.shape, device=dev) < (t**3)[:, None]
                    z_in = torch.where(keep, z, noise)
                pred = model(r, s, z_in, t, drop)
            elif args.head == "discrete":
                pred = model(r, s, drop)
            else:
                pred = model(r, s)
        if args.head in ("dit-dr", "dit-df", "discrete"):
            loss = soft_ce(pred.float(), a, args.bins, args.tau)
        elif args.loss == "mse":
            loss = F.mse_loss(pred.float(), a)
        else:
            loss = F.l1_loss(pred.float(), a)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        with torch.no_grad():
            decay = min(args.ema, (1 + step) / (10 + step))
            for pe, pm in zip(ema.parameters(), model.parameters(), strict=True):
                pe.mul_(decay).add_(pm.detach(), alpha=1 - decay)
            for be, bm in zip(ema.buffers(), model.buffers(), strict=True):
                be.copy_(bm)
        if step % 1000 == 0:
            rec = {"step": step, "train_loss": loss.item(), "val_l1": evaluate()}
            log.append(rec)
            print(json.dumps(rec), flush=True)
            save("last")
    save("last")
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "log": log}, indent=2))


if __name__ == "__main__":
    main()
