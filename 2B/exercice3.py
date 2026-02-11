import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

"""
Exercise 2
Initialization and normalization diagnostics.

This file is organized by parts (a) to (d), each announced in the terminal.
All runs reuse exactly the same:
dataset and train validation split
model architecture family (same builder, same depth and activation per comparison)
loss function (MSE)
optimizer and batch size
random seed
minibatch order (precomputed permutations)
"""

# =============================================================
# STRUCTURE OF THIS FILE RELATIVE TO THE ASSIGNMENT
# (a) Baseline configuration          -> see section in main() labeled (a)
# (b) Naive initialization            -> init_naive_gaussian + section (b)
# (c) Variance-preserving init        -> init_variance_aware + section (c)
# (d) Adding normalization (BatchNorm)-> use_batchnorm=True + section (d)
# =============================================================

# =============================
# (a) BASELINE CONFIGURATION
# This block fixes:
# - random seed
# - dataset size
# - batch size
# - optimizer and learning rate
# - loss function (MSE)
# These remain constant across (a)–(d) to ensure fair comparison.
# =============================
seed = 123
N = 64
batch_size = 16
epochs = 120

lr = 1e-2
optimizer_name = "sgd_momentum"   # keep fixed for all comparisons
momentum = 0.9

loss_fn = nn.MSELoss()


# This function ensures SAME dataset and SAME train/validation split
# for all experiments (requirement of part (a)).
def make_data(seed_value: int):
    torch.manual_seed(seed_value)
    x = torch.randn(N, 1)
    y = 3 * x - 1 + 0.1 * torch.randn(N, 1)

    n_train = int(0.8 * N)
    idx = torch.randperm(N)
    tr, va = idx[:n_train], idx[n_train:]
    return x[tr], y[tr], x[va], y[va]


# This precomputes identical minibatch permutations for every epoch.
# This guarantees SAME minibatch order across all experiments,
# satisfying the "same training conditions" constraint of part (a).
def make_epoch_perms(n_train: int, epochs_value: int, seed_value: int):
    g = torch.Generator().manual_seed(seed_value)
    return [torch.randperm(n_train, generator=g) for _ in range(epochs_value)]


# Model architecture builder.
# The architecture family is fixed and reused across (a)–(d).
# Depth and activation are varied deliberately in (b) and (c)
# to observe sensitivity to initialization and normalization.
def get_activation(name: str):
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Unknown activation: {name}")


def make_mlp(hidden_layers, activation: str, use_batchnorm: bool):
    layers = []
    in_dim = 1

    for width in hidden_layers:
        layers.append(nn.Linear(in_dim, width))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(width))
        layers.append(get_activation(activation))
        in_dim = width

    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


# =============================
# (b) NAIVE INITIALIZATION
# All Linear weights are sampled from N(0, sigma^2),
# independent of layer width or activation function.
# This intentionally ignores variance preservation principles.
# Used in section (b) and compared in (c) and (d).
# =============================
def init_default(model: nn.Module):
    # PyTorch default initialization, leave as is
    return


def init_naive_gaussian(model: nn.Module, sigma: float):
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=sigma)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# =============================
# (c) VARIANCE-PRESERVING INITIALIZATION
# - He initialization for ReLU
# - Xavier initialization for tanh/sigmoid
# Designed to preserve activation and gradient variance across depth.
# Compared directly to naive initialization in section (c).
# =============================
def init_variance_aware(model: nn.Module, activation: str):
    act = activation.lower()
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if act == "relu":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            else:
                nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# -----------------------------
# Sanity checks
# -----------------------------
def finite_tensor(t: torch.Tensor) -> bool:
    return torch.isfinite(t).all().item()


# Part (a) requirement:
# "Verify once more that the model passes the test run before training."
# This performs a forward pass, computes loss, runs backward,
# and checks that all gradients are finite.
def test_run_forward_backward(model: nn.Module, x_tr: torch.Tensor, y_tr: torch.Tensor):
    model.train()
    xb, yb = x_tr[:batch_size], y_tr[:batch_size]
    pred = model(xb)
    loss = loss_fn(pred, yb)

    if pred.shape != yb.shape:
        raise RuntimeError(f"Shape mismatch, pred {pred.shape} vs y {yb.shape}")
    if not finite_tensor(loss):
        raise RuntimeError("Loss is not finite in test run")

    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    loss.backward()

    for name, p in model.named_parameters():
        if p.grad is None:
            raise RuntimeError(f"Missing grad for {name}")
        if not finite_tensor(p.grad):
            raise RuntimeError(f"Non finite grad for {name}")

    print("Sanity check ok: forward, loss, backward, grads are finite")


def print_init_stats(model: nn.Module, tag: str):
    ws = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            ws.append(m.weight.detach().flatten())
    if not ws:
        return
    w = torch.cat(ws)
    print(f"{tag} weight stats: mean={w.mean().item():.4f} std={w.std(unbiased=False).item():.4f} maxabs={w.abs().max().item():.4f}")


# -----------------------------
# Training with logging
# -----------------------------

# Used in (b), (c), and (d).
# Records:
# - iteration-level training loss
# - epoch-level train and validation loss
# - gradient norms per parameter
# This satisfies the requirement to record loss curves and gradient norms.
def grad_norm(t):
    if t is None:
        return float("nan")
    return t.detach().norm(2).item()


def make_optimizer(model: nn.Module):
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    if optimizer_name == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    raise ValueError(f"Unknown optimizer_name: {optimizer_name}")


def train_logged(model: nn.Module, x_tr, y_tr, x_va, y_va, perms):
    opt = make_optimizer(model)

    it_tr_loss = []
    ep_tr_loss, ep_va_loss = [], []

    param_names = [n for n, _ in model.named_parameters()]
    grad_norms = {n: [] for n in param_names}

    for p in perms:
        model.train()
        x_s, y_s = x_tr[p], y_tr[p]

        for i in range(0, len(x_s), batch_size):
            xb, yb = x_s[i:i + batch_size], y_s[i:i + batch_size]
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()

            for n, param in model.named_parameters():
                grad_norms[n].append(grad_norm(param.grad))

            opt.step()
            it_tr_loss.append(loss.item())

        model.eval()
        with torch.no_grad():
            ep_tr_loss.append(loss_fn(model(x_tr), y_tr).item())
            ep_va_loss.append(loss_fn(model(x_va), y_va).item())

    return {
        "it_tr_loss": it_tr_loss,
        "ep_tr_loss": ep_tr_loss,
        "ep_va_loss": ep_va_loss,
        "grad_norms": grad_norms,
    }


# -----------------------------
# Plot helpers
# -----------------------------
def _moving_avg(xs, w=10):
    if w is None or w <= 1 or len(xs) < w:
        return xs
    out = []
    s = 0.0
    for i, v in enumerate(xs):
        s += float(v)
        if i >= w:
            s -= float(xs[i - w])
        if i >= w - 1:
            out.append(s / w)
    return out


def overlay_training_loss_iter(logs_by_name, title, smooth_window=10, logy=True):
    plt.figure(figsize=(9, 4.5))
    for name, logs in logs_by_name.items():
        y = logs["it_tr_loss"]
        y_s = _moving_avg(y, w=smooth_window)
        x0 = (smooth_window - 1) if (smooth_window and smooth_window > 1 and len(y) >= smooth_window) else 0
        plt.plot(range(x0, x0 + len(y_s)), y_s, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Training loss")
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def overlay_train_val_epoch(logs_by_name, title, logy=True):
    plt.figure(figsize=(9, 4.5))
    for name, logs in logs_by_name.items():
        (line_tr,) = plt.plot(logs["ep_tr_loss"], label=f"{name} train")
        plt.plot(logs["ep_va_loss"], linestyle="--", color=line_tr.get_color(), label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best", ncols=2)
    plt.tight_layout()
    plt.show()


def _global_grad_norm_per_iter(grad_norms: dict):
    # Approximate global L2 norm per iteration via sqrt(sum_i ||g_i||^2)
    keys = list(grad_norms.keys())
    if not keys:
        return []
    T = len(grad_norms[keys[0]])
    out = [0.0] * T
    for k in keys:
        g = grad_norms[k]
        for t in range(T):
            v = float(g[t])
            out[t] += v * v
    for t in range(T):
        out[t] = math.sqrt(out[t])
    return out


def overlay_global_grad_norm(logs_by_name, title, smooth_window=10, logy=True):
    plt.figure(figsize=(9, 4.5))
    for name, logs in logs_by_name.items():
        y = _global_grad_norm_per_iter(logs["grad_norms"])
        y_s = _moving_avg(y, w=smooth_window)
        x0 = (smooth_window - 1) if (smooth_window and smooth_window > 1 and len(y) >= smooth_window) else 0
        plt.plot(range(x0, x0 + len(y_s)), y_s, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Global grad norm (approx L2)")
    plt.title(title)
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


# -----------------------------
# Experiment runner
# -----------------------------

# Utility wrapper used in (a)–(d).
# Ensures SAME seed, SAME dataset, SAME optimizer, SAME batch size.
# Only initialization type and BatchNorm flag change.
def run_one(tag, hidden_layers, activation, use_batchnorm, init_kind, init_param, x_tr, y_tr, x_va, y_va, perms):
    torch.manual_seed(seed)

    model = make_mlp(hidden_layers=hidden_layers, activation=activation, use_batchnorm=use_batchnorm)

    if init_kind == "default":
        init_default(model)
    elif init_kind == "naive":
        init_naive_gaussian(model, sigma=float(init_param))
    elif init_kind == "variance_aware":
        init_variance_aware(model, activation=activation)
    else:
        raise ValueError(f"Unknown init_kind: {init_kind}")

    print_init_stats(model, tag)
    test_run_forward_backward(model, x_tr, y_tr)

    logs = train_logged(model, x_tr, y_tr, x_va, y_va, perms)
    print(f"{tag} final train={logs['ep_tr_loss'][-1]:.6f} val={logs['ep_va_loss'][-1]:.6f}")
    return logs


def main():
    x_tr, y_tr, x_va, y_va = make_data(seed)
    perms = make_epoch_perms(len(x_tr), epochs, seed)

    # =============================================================
    # (a) BASELINE CONFIGURATION
    # Reference model:
    # - default PyTorch initialization
    # - no BatchNorm
    # - fixed architecture
    # Serves as comparison reference for (b)–(d).
    # =============================================================
    print("\n(a) Baseline configuration")
    print("Same data split, same perms, same optimizer, same batch size, same seed")
    baseline_depth = "shallow"
    baseline_activation = "relu"
    hidden_layers = depths[baseline_depth]

    logs_baseline = run_one(
        tag=f"baseline default init, {baseline_depth}, {baseline_activation}",
        hidden_layers=hidden_layers,
        activation=baseline_activation,
        use_batchnorm=False,
        init_kind="default",
        init_param=None,
        x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va, perms=perms
    )

    # =============================================================
    # (b) NAIVE INITIALIZATION EXPERIMENTS
    # - Two sigma values: small (0.01) and large (1.0)
    # - Depth varied: shallow vs deep
    # - Activation varied: ReLU vs tanh
    # Purpose: observe instability, vanishing/exploding gradients,
    # and depth sensitivity under poor initialization.
    # =============================================================
    print("\n(b) Naive initialization")
    sigmas = [0.01, 1.0]

    for depth_name, hidden_layers in depths.items():
        for act in activations:
            logs_by_name = {}
            for s in sigmas:
                tag = f"naive sigma={s}, {depth_name}, {act}"
                logs_by_name[f"sigma={s}"] = run_one(
                    tag=tag,
                    hidden_layers=hidden_layers,
                    activation=act,
                    use_batchnorm=False,
                    init_kind="naive",
                    init_param=s,
                    x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va, perms=perms
                )

            overlay_training_loss_iter(logs_by_name, f"(b) train loss vs iter (smoothed), {depth_name}, {act}")
            overlay_train_val_epoch(logs_by_name, f"(b) train and val loss vs epoch, {depth_name}, {act}")
            overlay_global_grad_norm(logs_by_name, f"(b) global grad norm vs iter (smoothed), {depth_name}, {act}")

    # =============================================================
    # (c) VARIANCE-AWARE INITIALIZATION EXPERIMENTS
    # Direct comparison:
    # - naive sigma=0.01
    # - variance-aware (He/Xavier)
    # Same training conditions as (b).
    # Purpose: compare convergence speed, stability,
    # and gradient magnitude behavior.
    # =============================================================
    print("\n(c) Variance preserving initialization")
    for depth_name, hidden_layers in depths.items():
        for act in activations:
            logs_by_name = {}

            logs_by_name["naive sigma=0.01"] = run_one(
                tag=f"naive sigma=0.01, {depth_name}, {act}",
                hidden_layers=hidden_layers,
                activation=act,
                use_batchnorm=False,
                init_kind="naive",
                init_param=0.01,
                x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va, perms=perms
            )

            logs_by_name["variance aware"] = run_one(
                tag=f"variance aware, {depth_name}, {act}",
                hidden_layers=hidden_layers,
                activation=act,
                use_batchnorm=False,
                init_kind="variance_aware",
                init_param=None,
                x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va, perms=perms
            )

            overlay_training_loss_iter(logs_by_name, f"(c) train loss vs iter (smoothed), {depth_name}, {act}")
            overlay_train_val_epoch(logs_by_name, f"(c) train and val loss vs epoch, {depth_name}, {act}")
            overlay_global_grad_norm(logs_by_name, f"(c) global grad norm vs iter (smoothed), {depth_name}, {act}")

    # =============================================================
    # (d) ADDING BATCH NORMALIZATION
    # Compare:
    # - BN + naive
    # - BN + variance-aware
    # - no BN + naive
    # Purpose:
    # - Evaluate convergence speed
    # - Compare gradient magnitudes
    # - Test whether BN reduces sensitivity to initialization.
    # =============================================================
    print("\n(d) Adding normalization")
    for depth_name, hidden_layers in depths.items():
        for act in activations:
            logs_by_name = {}

            logs_by_name["BN + naive sigma=0.01"] = run_one(
                tag=f"BN naive sigma=0.01, {depth_name}, {act}",
                hidden_layers=hidden_layers,
                activation=act,
                use_batchnorm=True,
                init_kind="naive",
                init_param=0.01,
                x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va, perms=perms
            )

            logs_by_name["BN + variance aware"] = run_one(
                tag=f"BN variance aware, {depth_name}, {act}",
                hidden_layers=hidden_layers,
                activation=act,
                use_batchnorm=True,
                init_kind="variance_aware",
                init_param=None,
                x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va, perms=perms
            )

            logs_by_name["no BN + naive sigma=0.01"] = run_one(
                tag=f"no BN naive sigma=0.01, {depth_name}, {act}",
                hidden_layers=hidden_layers,
                activation=act,
                use_batchnorm=False,
                init_kind="naive",
                init_param=0.01,
                x_tr=x_tr, y_tr=y_tr, x_va=x_va, y_va=y_va, perms=perms
            )

            overlay_training_loss_iter(logs_by_name, f"(d) train loss vs iter (smoothed), {depth_name}, {act}")
            overlay_train_val_epoch(logs_by_name, f"(d) train and val loss vs epoch, {depth_name}, {act}")
            overlay_global_grad_norm(logs_by_name, f"(d) global grad norm vs iter (smoothed), {depth_name}, {act}")

    print("\nDone")


depths = {
    "shallow": [16],
    "deep": [16, 16, 16],
}
activations = ["relu", "tanh"]

if __name__ == "__main__":
    main()
