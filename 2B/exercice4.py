

import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Exercise 4 (Goodfellow et al., Ch. 8.2): destabilize training to observe "cliffs"/divergence.
# One deliberate destabilizing change used here: learning rate increased substantially.

seed = 123
N = 512
train_frac = 0.8
batch_size = 32
epochs = 25  # short runs on purpose


def set_seed(s):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def make_data(s=seed, n=N):
    """Small synthetic regression with noise (stable for baseline)."""
    set_seed(s)
    x = torch.randn(n, 2)
    # mild nonlinearity + noise
    y = (0.7 * x[:, :1] - 0.4 * x[:, 1:2] + 0.3 * torch.sin(2.0 * x[:, :1])
         + 0.1 * torch.randn(n, 1))

    idx = torch.randperm(n)
    n_tr = int(train_frac * n)
    tr_idx, va_idx = idx[:n_tr], idx[n_tr:]
    return x[tr_idx], y[tr_idx], x[va_idx], y[va_idx]


def make_model(s=seed, depth=2, width=64, use_norm=True, act="relu"):
    """MLP for regression. Keep this fixed for baseline and cliff except ONE change (lr)."""
    set_seed(s)

    if act == "relu":
        activation = nn.ReLU()
    elif act == "tanh":
        activation = nn.Tanh()
    elif act == "sigmoid":
        activation = nn.Sigmoid()
    else:
        raise ValueError(f"Unknown act: {act}")

    layers = []
    in_dim = 2
    for _ in range(depth):
        lin = nn.Linear(in_dim, width)
        layers.append(lin)
        if use_norm:
            layers.append(nn.LayerNorm(width))
        layers.append(activation)
        in_dim = width

    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def is_finite(x):
    return torch.isfinite(x).all().item()


def global_grad_norm(model):
    """g_t = ||∇_θ L_t||_2 aggregated over all trainable parameters."""
    s = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        s += float(p.grad.detach().pow(2).sum())
    return math.sqrt(s)


def per_layer_grad_norms(model):
    """One scalar grad norm per Linear layer (weights+bias together)."""
    out = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            s = 0.0
            if m.weight.grad is not None:
                s += float(m.weight.grad.detach().pow(2).sum())
            if (m.bias is not None) and (m.bias.grad is not None):
                s += float(m.bias.grad.detach().pow(2).sum())
            out[name] = math.sqrt(s)
    return out


def train_logged(model, lr, x_tr, y_tr, x_va, y_va, perms):
    """Record iteration-level loss and gradient norms. Stop early on NaN/Inf."""
    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=lr)

    it_loss = []
    it_gnorm = []
    it_layer_gnorm = []  # list of dicts

    ep_tr_loss, ep_va_loss = [], []

    it = 0
    for p in perms:
        model.train()
        x_s, y_s = x_tr[p], y_tr[p]

        for i in range(0, len(x_s), batch_size):
            xb, yb = x_s[i:i + batch_size], y_s[i:i + batch_size]
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()

            # record iteration-level signals (more informative than epoch-level when things explode)
            it_loss.append(float(loss.item()))
            it_gnorm.append(global_grad_norm(model))
            it_layer_gnorm.append(per_layer_grad_norms(model))

            opt.step()
            it += 1

            # stop early if we hit numerical instability
            if (not math.isfinite(it_loss[-1])) or (not math.isfinite(it_gnorm[-1])):
                break

        model.eval()
        with torch.no_grad():
            tr = loss_fn(model(x_tr), y_tr)
            va = loss_fn(model(x_va), y_va)
            ep_tr_loss.append(float(tr.item()))
            ep_va_loss.append(float(va.item()))

        if (len(it_loss) > 0) and ((not math.isfinite(it_loss[-1])) or (not math.isfinite(it_gnorm[-1]))):
            break

    return {
        "it_loss": it_loss,
        "it_gnorm": it_gnorm,
        "it_layer_gnorm": it_layer_gnorm,
        "ep_tr_loss": ep_tr_loss,
        "ep_va_loss": ep_va_loss,
    }


def plot_overlays(signals_by_name, key, title, ylabel):
    plt.figure()
    for name, sig in signals_by_name.items():
        plt.plot(sig[key], label=name)
    plt.xlabel("Iteration" if key.startswith("it_") else "Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_per_layer_overlays(signals_by_name):
    # Collect Linear layer names from the first run
    first = next(iter(signals_by_name.values()))
    if not first["it_layer_gnorm"]:
        return
    layer_names = list(first["it_layer_gnorm"][0].keys())

    for ln in layer_names:
        plt.figure()
        for name, sig in signals_by_name.items():
            series = [d.get(ln, float("nan")) for d in sig["it_layer_gnorm"]]
            plt.plot(series, label=name)
        plt.xlabel("Iteration")
        plt.ylabel("Per-layer grad norm (L2)")
        plt.title(f"Per-layer grad norm vs iteration (overlay): {ln}")
        plt.legend()
        plt.tight_layout()
        plt.show()


# (a) Stable baseline: same dataset/split, same model, MSE, SGD, batch size, seed.
x_tr, y_tr, x_va, y_va = make_data(seed)

g = torch.Generator().manual_seed(seed)
perms = [torch.randperm(len(x_tr), generator=g) for _ in range(epochs)]

baseline_model = make_model(seed, depth=2, width=64, use_norm=True, act="relu")

# keep baseline stable
lr_baseline = 1e-2
sig_baseline = train_logged(baseline_model, lr_baseline, x_tr, y_tr, x_va, y_va, perms)

print("Baseline final epoch train/val:", sig_baseline["ep_tr_loss"][-1], sig_baseline["ep_va_loss"][-1])


# (b) Deliberately create a cliff: EXACTLY ONE change -> learning rate increased substantially.
cliff_model = make_model(seed, depth=2, width=64, use_norm=True, act="relu")

lr_cliff = lr_baseline * 10.0
sig_cliff = train_logged(cliff_model, lr_cliff, x_tr, y_tr, x_va, y_va, perms)

print("Cliff final epoch train/val:", sig_cliff["ep_tr_loss"][-1], sig_cliff["ep_va_loss"][-1])


# (c) Record learning signals at iteration level (already recorded in train_logged).
# Why iteration-level is more informative than epoch-level for instability:
# within one epoch, a single minibatch update can push parameters into a region with exploding activations/gradients.
# Epoch averages can hide the exact update where things go from finite to divergent.


# Overlay plots for direct comparison
signals = {
    f"Baseline (lr={lr_baseline:g})": sig_baseline,
    f"Cliff (lr={lr_cliff:g})": sig_cliff,
}

# (c) plots requested
plot_overlays(signals, "it_loss", "Mini-batch loss L_t vs iteration (overlay)", "Mini-batch loss")
plot_overlays(signals, "it_gnorm", "Global grad norm ||∇θ L_t||2 vs iteration (overlay)", "Global grad norm")
plot_per_layer_overlays(signals)

# epoch-level losses too (useful, but can be late/blurred for diagnosing instability)
plot_overlays(signals, "ep_tr_loss", "Train loss vs epoch (overlay)", "Train loss")
plot_overlays(signals, "ep_va_loss", "Validation loss vs epoch (overlay)", "Validation loss")


# (d) Reason about the cliff (brief, conceptual, no numeric obsession):
# A "cliff" corresponds to a sudden transition where one update crosses into a region of the loss surface
# with extreme curvature or poor conditioning, making the next steps overshoot and amplify errors.
# It is typically sudden at the iteration level (one or a few updates), not gradual.
# Often the earliest warning shows up in per-layer or global gradient norms (spiking) before loss becomes NaN,
# because exploding gradients can occur while the loss is still finite for that minibatch.
# Instability often originates in deeper/later layers first (closer to the loss) where gradients are larger,
# then propagates backward; but once activations explode, it can rapidly contaminate the whole network.


# (e) Mechanistic explanation (Session 2A language):
# Increasing learning rate worsens the effective conditioning by making the update step too large relative to curvature.
# Forward signal propagation: large parameter updates can increase pre-activation magnitudes, pushing activations into
# regimes with large outputs (or saturation for tanh/sigmoid), making the network output highly sensitive.
# Backward gradient flow: large activations + large Jacobians cause gradients to explode; with a too-large step,
# the optimizer overshoots minima and bounces into steeper regions, matching the "cliff" intuition in Fig. 8.2/8.3.