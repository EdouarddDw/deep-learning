import torch
import torch.nn as nn
import matplotlib.pyplot as plt

seed = 123
N = 64
batch_size = 16
epochs = 100

def make_data(seed):
    torch.manual_seed(seed)
    x = torch.randn(N, 1)
    y = 3 * x - 1 + 0.1 * torch.randn(N, 1)

    n_train = int(0.8 * N)
    idx = torch.randperm(N)                      # Chapter 5 style split  [oai_citation:0‡Deep-Learning-with-PyTorch.pdf](sediment://file_00000000120c7246b76a5e4505becd15)
    tr, va = idx[:n_train], idx[n_train:]
    return x[tr], y[tr], x[va], y[va]

def make_model(seed):
    torch.manual_seed(seed)                      # ensures identical init
    return nn.Sequential(
        nn.Linear(1, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )

loss_fn = nn.MSELoss()

# --- training with logging for plots ---

def grad_norm(t):
    if t is None:
        return float("nan")
    return t.detach().norm(2).item()


def train_with_opt_logged(opt_name, lr, x_tr, y_tr, x_va, y_va, perms, seed_for_init,
                          batch_size=16, momentum=0.9):
    """Train the same model with a chosen optimizer, logging per-iteration loss and grad norms."""
    model = make_model(seed_for_init)

    if opt_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt_name == "sgd_momentum":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    elif opt_name == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown opt_name: {opt_name}")

    it_tr_loss = []
    ep_tr_loss, ep_va_loss = [], []

    # per-parameter grad norms (acts as per-layer for this small Sequential)
    param_names = [n for n, _ in model.named_parameters()]
    grad_norms = {n: [] for n in param_names}

    for p in perms:  # one perm = one epoch
        model.train()
        x_s, y_s = x_tr[p], y_tr[p]

        for i in range(0, len(x_s), batch_size):
            xb, yb = x_s[i:i + batch_size], y_s[i:i + batch_size]
            pred = model(xb)
            loss = loss_fn(pred, yb)

            opt.zero_grad()
            loss.backward()

            # log grad norms before the parameter update
            for n, param in model.named_parameters():
                grad_norms[n].append(grad_norm(param.grad))

            opt.step()

            it_tr_loss.append(loss.item())

        model.eval()
        with torch.no_grad():
            ep_tr_loss.append(loss_fn(model(x_tr), y_tr).item())
            ep_va_loss.append(loss_fn(model(x_va), y_va).item())

    return {
        "model": model,
        "it_tr_loss": it_tr_loss,
        "ep_tr_loss": ep_tr_loss,
        "ep_va_loss": ep_va_loss,
        "grad_norms": grad_norms,
    }


def plot_optimizer_runs(name, logs):
    # 1) training loss vs iterations
    plt.figure()
    plt.plot(logs["it_tr_loss"])
    plt.xlabel("Iteration (minibatch update)")
    plt.ylabel("Training loss")
    plt.title(f"{name}: training loss vs iterations")
    plt.tight_layout()
    plt.show()

    # 2) train + val loss vs epochs
    plt.figure()
    plt.plot(logs["ep_tr_loss"], label="train")
    plt.plot(logs["ep_va_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name}: train and val loss vs epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3) gradient norms per layer/parameter vs iterations
    plt.figure()
    for param_name, series in logs["grad_norms"].items():
        plt.plot(series, label=param_name)
    plt.xlabel("Iteration (minibatch update)")
    plt.ylabel("Gradient norm (L2)")
    plt.title(f"{name}: gradient norms per parameter vs iterations")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_overlays(logs_by_name):
    """Overlay plots for direct comparison across optimizers."""

    # A) training loss vs iterations (all optimizers)
    plt.figure()
    for name, logs in logs_by_name.items():
        plt.plot(logs["it_tr_loss"], label=name)
    plt.xlabel("Iteration (minibatch update)")
    plt.ylabel("Training loss")
    plt.title("Training loss vs iterations (overlay)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # B) train loss vs epochs (all optimizers)
    plt.figure()
    for name, logs in logs_by_name.items():
        plt.plot(logs["ep_tr_loss"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Train loss")
    plt.title("Train loss vs epochs (overlay)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # C) val loss vs epochs (all optimizers)
    plt.figure()
    for name, logs in logs_by_name.items():
        plt.plot(logs["ep_va_loss"], label=name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation loss")
    plt.title("Validation loss vs epochs (overlay)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # D) gradient norms per parameter vs iterations (one plot per parameter)
    # assumes all runs have the same parameter names
    param_names = list(next(iter(logs_by_name.values()))["grad_norms"].keys())
    for p in param_names:
        plt.figure()
        for name, logs in logs_by_name.items():
            plt.plot(logs["grad_norms"][p], label=name)
        plt.xlabel("Iteration (minibatch update)")
        plt.ylabel("Gradient norm (L2)")
        plt.title(f"Grad norm vs iterations (overlay): {p}")
        plt.legend()
        plt.tight_layout()
        plt.show()

def train_with_opt(opt_name, lr, x_tr, y_tr, x_va, y_va, perms, seed_for_init, momentum=0.9):
    logs = train_with_opt_logged(
        opt_name=opt_name,
        lr=lr,
        x_tr=x_tr,
        y_tr=y_tr,
        x_va=x_va,
        y_va=y_va,
        perms=perms,
        seed_for_init=seed_for_init,
        batch_size=batch_size,
        momentum=momentum,
    )
    return logs["ep_tr_loss"], logs["ep_va_loss"], logs

x_tr, y_tr, x_va, y_va = make_data(seed)

g = torch.Generator().manual_seed(seed)
perms = [torch.randperm(len(x_tr), generator=g) for _ in range(epochs)]

lr = 1e-2
seed_for_init = seed

tr_sgd, va_sgd, logs_sgd = train_with_opt("sgd", lr, x_tr, y_tr, x_va, y_va, perms, seed_for_init)
tr_mom, va_mom, logs_mom = train_with_opt("sgd_momentum", lr, x_tr, y_tr, x_va, y_va, perms, seed_for_init, momentum=0.9)
tr_adam, va_adam, logs_adam = train_with_opt("adam", lr, x_tr, y_tr, x_va, y_va, perms, seed_for_init)

print("final sgd         train, val:", tr_sgd[-1], va_sgd[-1])
print("final sgd momentum train, val:", tr_mom[-1], va_mom[-1])
print("final adam        train, val:", tr_adam[-1], va_adam[-1])

logs_by_name = {
    "SGD": logs_sgd,
    "SGD + Momentum": logs_mom,
    "Adam": logs_adam,
}

plot_overlays(logs_by_name)

# Optional: keep individual per-optimizer plots too
# plot_optimizer_runs("SGD", logs_sgd)
# plot_optimizer_runs("SGD + Momentum", logs_mom)
# plot_optimizer_runs("Adam", logs_adam)
