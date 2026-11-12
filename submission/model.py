"""Chunked behaviour-cloning policy: 3 RGB cameras + proprioception -> H x 20 actions."""

import torch
import torch.nn as nn
import torchvision

N_CAM, ACT_DIM, STATE_DIM = 3, 20, 50


class ChunkPolicy(nn.Module):
    def __init__(self, horizon=20, hidden=512, pretrained=False):
        super().__init__()
        self.horizon = horizon
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone  # shared across cameras
        self.cam_embed = nn.Parameter(torch.zeros(N_CAM, 512))
        self.state_mlp = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(N_CAM * 512 + 128, hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
            nn.Linear(hidden * 2, horizon * ACT_DIM),
        )
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def act(self, rgb, state, cfg=0.0):
        return self(rgb, state)

    def forward(self, rgb, state):
        """rgb: (B,3cam,3,H,W) float in [0,1]; state: (B,50) normalized."""
        b = rgb.shape[0]
        x = (rgb.flatten(0, 1) - self.img_mean) / self.img_std
        f = self.backbone(x).view(b, N_CAM, 512) + self.cam_embed
        z = torch.cat([f.flatten(1), self.state_mlp(state)], dim=1)
        return torch.tanh(self.head(z)).view(b, self.horizon, ACT_DIM)


class DiscreteChunkPolicy(nn.Module):
    """Discrete-regression head (Yi et al. 2026, arXiv:2610.20429): each (step, dim) of the chunk is
    a token over `bins` uniform bins on [-1, 1], trained with cross-entropy, decoded by argmax.
    Optional classifier-free guidance via a learned null condition (dropped with p=0.1 in training).
    Same image/state backbone as ChunkPolicy so the head is the only difference."""

    def __init__(self, horizon=20, hidden=512, bins=256, emb=64, pretrained=False):
        super().__init__()
        self.horizon, self.bins, self.emb = horizon, bins, emb
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.cam_embed = nn.Parameter(torch.zeros(N_CAM, 512))
        self.state_mlp = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.ReLU())
        cdim = N_CAM * 512 + 128
        self.null = nn.Parameter(torch.zeros(cdim))  # learned null condition
        self.trunk = nn.Sequential(
            nn.Linear(cdim, hidden * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden * 2, hidden * 2),
            nn.ReLU(),
        )
        self.to_tok = nn.Linear(hidden * 2, horizon * ACT_DIM * emb)
        self.tok_bias = nn.Parameter(torch.zeros(horizon * ACT_DIM, emb))
        self.cls = nn.Sequential(nn.LayerNorm(emb), nn.Linear(emb, bins))
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("centers", (torch.arange(bins) + 0.5) * 2 / bins - 1)

    def condition(self, rgb, state):
        b = rgb.shape[0]
        x = (rgb.flatten(0, 1) - self.img_mean) / self.img_std
        f = self.backbone(x).view(b, N_CAM, 512) + self.cam_embed
        return torch.cat([f.flatten(1), self.state_mlp(state)], dim=1)

    def logits_from(self, c):
        h = self.trunk(c)
        tok = self.to_tok(h).view(-1, self.horizon * ACT_DIM, self.emb) + self.tok_bias
        return self.cls(tok).view(-1, self.horizon, ACT_DIM, self.bins)

    def forward(self, rgb, state, drop=None):
        """Logits (B, H, 20, bins). `drop` (B,) bool replaces the condition by the null condition."""
        c = self.condition(rgb, state)
        if drop is not None:
            c = torch.where(drop[:, None], self.null.expand_as(c), c)
        return self.logits_from(c)

    @torch.no_grad()
    def act(self, rgb, state, cfg=0.0):
        c = self.condition(rgb, state)
        s = self.logits_from(c)
        if cfg > 0:
            s = (1 + cfg) * s - cfg * self.logits_from(self.null.expand_as(c))
        return self.centers[s.argmax(-1)]


class _DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning (the paper's DR/DF head uses DiT-style blocks)."""

    def __init__(self, width, heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.n1 = nn.LayerNorm(width, elementwise_affine=False)
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)
        self.n2 = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(width, mlp_ratio * width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_ratio * width, width),
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c):
        sh1, sc1, g1, sh2, sc2, g2 = self.ada(c).chunk(6, dim=-1)
        h = self.n1(x) * (1 + sc1[:, None]) + sh1[:, None]
        b, n, w = h.shape
        q, k, v = self.qkv(h).view(b, n, 3, self.heads, w // self.heads).permute(2, 0, 3, 1, 4)
        a = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x + g1[:, None] * self.proj(a.transpose(1, 2).reshape(b, n, w))
        h = self.n2(x) * (1 + sc2[:, None]) + sh2[:, None]
        return x + g2[:, None] * self.mlp(h)


class DiscreteDiTPolicy(nn.Module):
    """Paper-faithful discrete action head (Yi et al. 2026, arXiv:2610.20429).

    One token per (chunk step, action dim) over `bins` uniform bins on [-1, 1]. A DiT transformer
    over the L = H*20 tokens is conditioned (adaLN) on the observation embedding (+ flow time).
      mode "dr": discrete regression: input tokens are random (auxiliary random inputs), one pass, argmax.
      mode "df": discrete flow: input is the mixture-path corrupted chunk z_t (alpha(t)=t^3, uniform
                 source), trained on clean-token CE; inference runs `flow_steps` refinement steps.
    Classifier-free guidance via a learned null condition (dropped with p=0.1 during training).
    """

    def __init__(
        self, horizon=20, bins=512, width=256, blocks=12, heads=4, mode="dr", pretrained=False
    ):
        super().__init__()
        self.horizon, self.bins, self.width, self.mode = horizon, bins, width, mode
        self.L = horizon * ACT_DIM
        self.flow_steps = 20
        weights = torchvision.models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = torchvision.models.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.cam_embed = nn.Parameter(torch.zeros(N_CAM, 512))
        self.state_mlp = nn.Sequential(nn.Linear(STATE_DIM, 128), nn.ReLU())
        self.obs_mlp = nn.Sequential(
            nn.Linear(N_CAM * 512 + 128, 512), nn.GELU(), nn.Linear(512, width)
        )
        self.null = nn.Parameter(torch.zeros(width))
        self.tok = nn.Embedding(bins, width)
        self.pos = nn.Parameter(torch.randn(self.L, width) * 0.02)
        self.temb = nn.Sequential(nn.Linear(64, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList([_DiTBlock(width, heads) for _ in range(blocks)])
        self.out_norm = nn.LayerNorm(width)
        self.out = nn.Linear(width, bins)
        self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.register_buffer("centers", (torch.arange(bins) + 0.5) * 2 / bins - 1)

    def obs_embed(self, rgb, state):
        b = rgb.shape[0]
        x = (rgb.flatten(0, 1) - self.img_mean) / self.img_std
        f = self.backbone(x).view(b, N_CAM, 512) + self.cam_embed
        return self.obs_mlp(torch.cat([f.flatten(1), self.state_mlp(state)], dim=1))

    def _time_feat(self, t):
        freqs = torch.exp(torch.linspace(0, 6, 32, device=t.device))
        a = t[:, None] * 1000 * freqs[None] / 100
        return torch.cat([a.sin(), a.cos()], dim=-1)

    def head(self, c_obs, z_in, t):
        c = c_obs + self.temb(self._time_feat(t))
        x = self.tok(z_in) + self.pos
        for blk in self.blocks:
            x = blk(x, c)
        return self.out(self.out_norm(x)).view(-1, self.horizon, ACT_DIM, self.bins)

    def forward(self, rgb, state, z_in, t, drop=None):
        c = self.obs_embed(rgb, state)
        if drop is not None:
            c = torch.where(drop[:, None], self.null.expand_as(c), c)
        return self.head(c, z_in, t)

    @staticmethod
    def alpha(t):
        return t**3

    @torch.no_grad()
    def act(self, rgb, state, cfg=0.0):
        c = self.obs_embed(rgb, state)
        n = c.shape[0]
        dev = c.device
        z = torch.randint(0, self.bins, (n, self.L), device=dev)

        def logits(zz, t):
            lg = self.head(c, zz, t)
            if cfg > 0:
                lg = (1 + cfg) * lg - cfg * self.head(self.null.expand_as(c), zz, t)
            return lg

        if self.mode == "dr":
            return self.centers[logits(z, torch.zeros(n, device=dev)).argmax(-1)]
        steps = self.flow_steps
        for i in range(steps):
            t0, t1 = i / steps, (i + 1) / steps
            lg = logits(z, torch.full((n,), t0, device=dev)).view(n, self.L, self.bins)
            if i == steps - 1:
                prop = lg.argmax(-1)
                z = prop
            else:
                prop = torch.distributions.Categorical(logits=lg).sample()
                eta = (self.alpha(t1) - self.alpha(t0)) / (1 - self.alpha(t0))
                z = torch.where(torch.rand(n, self.L, device=dev) < eta, prop, z)
        return self.centers[z.view(n, self.horizon, ACT_DIM)]


def action_limits():
    """Raw action limits (low, high) from the evaluator's public action layout."""
    import json
    from pathlib import Path

    try:
        from importlib.resources import files

        text = files("advaitic_eval").joinpath("action_layout.json").read_text()
    except ModuleNotFoundError:  # training env without the evaluator package: use the repo copy
        text = (
            Path(__file__).resolve().parents[1] / "src/advaitic_eval/action_layout.json"
        ).read_text()
    lay = json.loads(text)
    return lay["action_low"], lay["action_high"]


def arm_ref(state_raw, low, high):
    """Normalised current arm-joint positions placed in action slots 4..17 (zeros elsewhere).

    `state_raw` is the un-normalised 50-d proprioception; left arm = state[0:7], right = state[9:16].
    Relative actions (as in arXiv:2610.20429) are measured from this pose at prediction time.
    """
    ref = torch.zeros(state_raw.shape[0], ACT_DIM, device=state_raw.device)
    raw = torch.cat([state_raw[:, 0:7], state_raw[:, 9:16]], dim=1)
    ref[:, 4:18] = 2 * (raw - low[4:18]) / (high[4:18] - low[4:18]) - 1
    return ref
