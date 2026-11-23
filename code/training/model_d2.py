"""
D2 architecture: v24 trunk (LSTM + DirectedGAT) + Mixture Density Network head.

The head emits the parameters of a K-component LogNormal mixture per node/day:

    p(y | x) = sum_k  pi_k * LogNormal(y; mu_k, sigma_k)

The mixture *components* are the implicit experts (baseflow / normal / flood
modes); the mixing weights pi are the implicit gate. There is no discrete switch
-- everything is one differentiable model trained by maximum likelihood (NLL).

Why LogNormal (not Gaussian):
    streamflow is non-negative, right-skewed and heavy-tailed. A flood component
    naturally takes a large mu and wide sigma -> fat right tail -> the mixture
    *expects* extremes, which is exactly the signal a single mean objective
    cannot encode (the -36% peak underprediction).

Outputs / deliverables:
    - point estimate : mixture mean (closed form) -> KGE, matches v24 metric
    - quantiles      : invert the mixture CDF (P10..P99) -> the D2 deliverable
    - regime weights : pi per node/day -> interpretable "expert" attribution

Staging (joint training, see train_multi_gpu_d2.py):
    A. warmup   : freeze trunk, train MDN head only (cheap, hours)
    B. pretrain : unfreeze trunk, NLL, temporal split on train basins
    C. finetune : unfreeze trunk, NLL + gauge dropout, PUB basin holdout
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_v15 import StreamflowGNN

# Quantiles we ultimately report (same set as v26 so downstream is unchanged)
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
QUANTILE_LABELS = ['p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99']

# normal CDF / inverse used for mixture CDF + PIT (no torch.distributions dependency)
_SQRT2 = math.sqrt(2.0)


def _normal_cdf(z: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(z / _SQRT2))


class MixtureDensityHead(nn.Module):
    """LogNormal mixture head, anchored on the v24 mean prediction.

    Per node the MLP emits 3K numbers from [h, log1p(anchor)]:
        delta_k       -> mu_k    = log(anchor) + delta_k      (location, log space)
        raw_sigma_k   -> sigma_k = softplus(raw) + sigma_floor (positive spread)
        pi_logits_k   -> pi      = softmax(logits)            (mixing weights)

    Anchoring mu on log(anchor) means at init (zero MLP weights) every component
    sits near the v24 mean, so the mixture inherits v24's skill on day 0 and only
    has to *learn spread / modes* -- the same warm-start trick that made v26 stable.

    Symmetry breaking: the K components get DISTINCT bias offsets (low/normal/high
    mode), otherwise identical components never differentiate (classic MDN failure).
    """

    def __init__(self, hidden_dim: int, n_components: int = 3,
                 mlp_hidden: int = 128, dropout: float = 0.1,
                 sigma_floor: float = 0.05):
        super().__init__()
        self.K = n_components
        self.sigma_floor = sigma_floor

        in_dim = hidden_dim + 1                              # + log1p(anchor)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 3 * n_components),
        )

        # Init: zero final weights (stable, input-independent at epoch 0) but
        # DISTINCT biases per component so the modes start separated.
        with torch.no_grad():
            final = self.mlp[-1]
            nn.init.zeros_(final.weight)
            bias = torch.zeros(3 * n_components)
            # delta (mu offset in log space): low / normal / high mode
            delta0 = torch.linspace(-0.3, 0.6, n_components)
            # raw_sigma: flood mode starts wider
            sig0 = torch.linspace(-1.2, -0.5, n_components)
            # pi logits: favour the "normal" middle component initially
            pi0 = torch.full((n_components,), 0.0)
            if n_components >= 3:
                pi0[n_components // 2] = 1.2                 # boost middle
                pi0[-1] = -1.0                               # rare flood
            bias[0:n_components] = delta0
            bias[n_components:2 * n_components] = sig0
            bias[2 * n_components:] = pi0
            final.bias.copy_(bias)

    def forward(self, h: torch.Tensor, anchor: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h:      [..., hidden_dim] node features
            anchor: [..., 1] v24 mean prediction (>= 0), the location anchor
        Returns:
            pi:    [..., K]  mixing weights (sum to 1)
            mu:    [..., K]  log-space component locations
            sigma: [..., K]  component spreads (> sigma_floor)
        """
        # fp32 throughout: log/exp/erf overflow fp16 at extreme flows.
        with torch.amp.autocast("cuda", enabled=False):
            h_f = h.float()
            a = torch.nan_to_num(anchor.float(), nan=0.0, posinf=1e4, neginf=0.0)
            a = a.clamp_min(0.0)
            log_anchor = torch.log1p(a)                      # [..., 1]
            x = torch.cat([h_f, log_anchor], dim=-1)
            raw = self.mlp(x)                                # [..., 3K]

            K = self.K
            delta = raw[..., 0:K]
            raw_sigma = raw[..., K:2 * K]
            pi_logits = raw[..., 2 * K:3 * K]

            # mu anchored in log space on log(anchor) (use log1p for anchor~0 safety)
            mu = torch.log1p(a) + delta                      # broadcast [...,1]+[...,K]
            sigma = F.softplus(raw_sigma) + self.sigma_floor
            pi = F.softmax(pi_logits, dim=-1)
            return pi, mu, sigma


class StreamflowGNNMDN(nn.Module):
    """v24 trunk + MDN head. forward returns (pred_mean_v24, (pi,mu,sigma), h_lstm)."""

    def __init__(self, base_model: StreamflowGNN, n_components: int = 3):
        super().__init__()
        self.base = base_model
        head_in_dim = base_model.config.gnn_hidden_dim * 2   # pre+post GNN concat
        self.mdn_head = MixtureDensityHead(head_in_dim, n_components=n_components)
        self.K = n_components
        self._trunk_frozen = False

    def freeze_trunk(self):
        """Stage A: freeze trunk, train head only (trunk kept in eval)."""
        for p in self.base.parameters():
            p.requires_grad = False
        for p in self.mdn_head.parameters():
            p.requires_grad = True
        self.base.eval()
        self._trunk_frozen = True

    def unfreeze_trunk(self):
        """Stages B/C: joint training -- trunk learns peak-relevant features."""
        for p in self.base.parameters():
            p.requires_grad = True
        for p in self.mdn_head.parameters():
            p.requires_grad = True
        self._trunk_frozen = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self._trunk_frozen:
            self.base.eval()                                 # keep trunk deterministic
        return self

    def lstm_step(self, x_t, h_lstm=None):
        return self.base.lstm_step(x_t, h_lstm)

    def forward(self, x_t, edge_index_up, edge_attr_up, edge_index_down,
                edge_attr_down, h_lstm=None, x_static=None):
        # LSTM
        lstm_out, h_lstm_new = self.base.lstm(x_t.unsqueeze(0), h_lstm)
        h_node = lstm_out.squeeze(0)

        # GNN trunk
        h_pre_gnn = self.base.lstm_to_gnn(h_node)
        if x_static is not None:
            h_pre_gnn = h_pre_gnn + self.base.static_proj(x_static)
        h_post_gnn = self.base.gnn(
            h_pre_gnn, edge_index_up, edge_attr_up, edge_index_down, edge_attr_down
        )
        h_combined = torch.cat([h_pre_gnn, h_post_gnn], dim=-1)

        # v24 mean head -> anchor for the MDN
        pred_mean = self.base.output_head(h_combined)        # [n_nodes, 1]
        anchor = pred_mean
        if self._trunk_frozen:
            anchor = anchor.detach()                         # no grad to frozen trunk
        pi, mu, sigma = self.mdn_head(h_combined, anchor)
        return pred_mean, (pi, mu, sigma), h_lstm_new


# ---------------------------------------------------------------------------
# Loss + inference helpers
# ---------------------------------------------------------------------------

def mixture_nll(obs: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor,
                sigma: torch.Tensor, obs_floor: float = 1e-2
                ) -> Tuple[torch.Tensor, dict]:
    """Negative log-likelihood of a LogNormal mixture.

    Args:
        obs:   [B, T]      observed flow (mm/day); NaN where missing
        pi/mu/sigma: [B, T, K]  mixture params
    Returns:
        loss (scalar), stats dict
    """
    with torch.amp.autocast("cuda", enabled=False):
        pi = pi.float(); mu = mu.float(); sigma = sigma.float()
        valid = torch.isfinite(obs) & torch.isfinite(pi).all(-1) \
            & torch.isfinite(mu).all(-1) & torch.isfinite(sigma).all(-1)
        if valid.sum() == 0:
            return pi.sum() * 0.0, {'nll': float('nan'), 'n': 0}

        # CRITICAL: replace invalid obs with a safe positive value BEFORE log().
        # If NaNs reach log(), they poison the autograd graph even at masked-out
        # positions (0 * NaN = NaN in backward), giving all-NaN gradients.
        obs_safe = torch.where(valid, obs, torch.ones_like(obs))
        y = obs_safe.clamp_min(obs_floor)                    # [B, T]
        ly = torch.log(y).unsqueeze(-1)                      # [B, T, 1]

        # log LogNormal pdf per component:
        #   -ln(y) - ln(sigma) - 0.5 ln(2pi) - (ln y - mu)^2 / (2 sigma^2)
        comp_logp = (
            -ly
            - torch.log(sigma)
            - 0.5 * math.log(2 * math.pi)
            - 0.5 * ((ly - mu) / sigma) ** 2
        )                                                    # [B, T, K]
        log_mix = torch.logsumexp(torch.log(pi + 1e-8) + comp_logp, dim=-1)  # [B, T]

        nll = -log_mix[valid].mean()
        return nll, {'nll': float(nll.detach().cpu()), 'n': int(valid.sum().item())}


def mixture_mean(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor
                 ) -> torch.Tensor:
    """Closed-form mean of a LogNormal mixture: sum_k pi_k exp(mu_k + sigma_k^2/2)."""
    comp_mean = torch.exp(mu + 0.5 * sigma ** 2)             # [..., K]
    return (pi * comp_mean).sum(-1)                          # [...]


def mixture_cdf(q: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor,
                sigma: torch.Tensor) -> torch.Tensor:
    """Mixture CDF at flow value(s) q (broadcast over components).

    q: [...]   pi/mu/sigma: [..., K]   -> returns [...] in [0,1]
    """
    qf = q.clamp_min(1e-6).unsqueeze(-1)                     # [..., 1]
    z = (torch.log(qf) - mu) / sigma                         # [..., K]
    return (pi * _normal_cdf(z)).sum(-1)                     # [...]


def mixture_pit(obs: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor,
                sigma: torch.Tensor) -> torch.Tensor:
    """PIT value u = F_pred(obs) -- used for calibration/coverage without inversion.

    coverage at level alpha == mean(u <= alpha). Perfect calibration -> u ~ Uniform.
    """
    return mixture_cdf(obs, pi, mu, sigma)


def mixture_crps(obs: torch.Tensor, pi: torch.Tensor, mu: torch.Tensor,
                 sigma: torch.Tensor, n_levels: int = 100, n_iter: int = 30
                 ) -> torch.Tensor:
    """CRPS of the LogNormal mixture via the quantile (pinball) decomposition:

        CRPS(F, y) = 2 * integral_0^1 rho_tau(y - q_tau) d_tau

    where rho_tau(u) = u * (tau - 1{u < 0}) is the pinball loss and q_tau is the
    mixture's tau-quantile. A LogNormal *mixture* has no closed-form CRPS, so we
    approximate the integral on a uniform grid of `n_levels` quantiles obtained
    from the exact CDF inverter (`mixture_quantiles`). Deterministic (no MC noise)
    and a proper score. Lower is better; units are flow units (same as obs).

    obs: [...]   pi/mu/sigma: [..., K]   ->   returns [...] CRPS per element.
    """
    with torch.amp.autocast("cuda", enabled=False):
        # midpoint grid avoids the degenerate tau=0,1 quantiles (q -> 0 / inf)
        taus = (torch.arange(n_levels, device=pi.device, dtype=torch.float32)
                + 0.5) / n_levels                                # [L]
        q = mixture_quantiles(pi, mu, sigma, levels=taus.tolist(),
                              n_iter=n_iter)                       # [..., L]
        u = obs.float().unsqueeze(-1) - q                          # [..., L]
        tau = taus.view(*([1] * (u.dim() - 1)), n_levels)          # broadcast
        pinball = u * (tau - (u < 0).float())                      # [..., L]
        return 2.0 * pinball.mean(-1)                              # [...]


def mixture_quantiles(pi: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
                      levels: List[float] = QUANTILES, n_iter: int = 40
                      ) -> torch.Tensor:
    """Invert the mixture CDF for each requested level via vectorized bisection.

    pi/mu/sigma: [..., K]   ->   returns [..., len(levels)] flow values.
    Bracket [lo, hi] in log-flow space; CDF is monotone so bisection is exact
    to ~2^-n_iter of the bracket.
    """
    with torch.amp.autocast("cuda", enabled=False):
        pi = pi.float(); mu = mu.float(); sigma = sigma.float()
        shape = pi.shape[:-1]
        # log-flow bracket from the component locations +/- a wide margin
        lo = (mu - 8.0 * sigma).min(-1).values                # [...]
        hi = (mu + 8.0 * sigma).max(-1).values                # [...]
        out = []
        for a in levels:
            target = torch.full(shape, float(a), device=pi.device)
            llo = lo.clone(); lhi = hi.clone()
            for _ in range(n_iter):
                mid = 0.5 * (llo + lhi)
                cdf = mixture_cdf(torch.exp(mid), pi, mu, sigma)
                go_up = cdf < target
                llo = torch.where(go_up, mid, llo)
                lhi = torch.where(go_up, lhi, mid)
            out.append(torch.exp(0.5 * (llo + lhi)))
        return torch.stack(out, dim=-1)                       # [..., n_levels]
