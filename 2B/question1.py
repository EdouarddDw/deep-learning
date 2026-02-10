import torch
import torch.nn as nn

#sub question a
N = 64

x = torch.randn(N, 1)

#noise epsion
epsilon = 0.1 * torch.randn(N, 1)

y = 3 * x - 1 + epsilon

train_size = int(0.8 * N)
perm = torch.randperm(N)

train_idx = perm[:train_size]
val_idx = perm[train_size:]

x_train, y_train = x[train_idx], y[train_idx]
x_val, y_val = x[val_idx], y[val_idx]

#sub question b

model = nn.Sequential(
    nn.Linear(1, 16),
    nn.ReLU(),
    nn.Linear(16,1)
        )
print(model)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_params}")

# sub question C
#test run

test_x = torch.ones(x_train.shape)
model_output = model(test_x)
#the write shape should be the same shape as y train as that is what we are trying to predict
print("does the test run have the same shape as y_train",model_output.shape == y_train.shape)

model.train()
model.zero_grad()

#froward
pred = model(test_x)

#backwards
loss = nn.MSELoss()(pred, y_train)

loss.backward()

for name, p in model.named_parameters():
    if p.requires_grad:
        print(name, "grad is None?", p.grad is None)
        assert p.grad is not None, f"{name} has None grad"

# sub question d


batch_size = 16
lr = 1e-2
epochs = 100

loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)

iter_logs = []   # iteration level: batch_loss, grad_norm, epoch, iter_in_epoch
epoch_logs = []  # epoch level: avg_train_loss, val_loss

N_train = x_train.shape[0]

grad_stats = {}
global_iter = 0

for epoch in range(epochs):
    model.train()

    perm = torch.randperm(N_train)
    x_shuf = x_train[perm]
    y_shuf = y_train[perm]

    sum_train_loss = 0.0
    it_in_epoch = 0

    for i in range(0, N_train, batch_size):
        xb = x_shuf[i:i + batch_size]
        yb = y_shuf[i:i + batch_size]

        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        # record gradient stats per parameter tensor
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue

            if name not in grad_stats:
                grad_stats[name] = {"l2": [], "maxabs": []}

            if p.grad is None:
                grad_stats[name]["l2"].append(float("nan"))
                grad_stats[name]["maxabs"].append(float("nan"))
            else:
                g = p.grad.detach()
                grad_stats[name]["l2"].append(g.norm(2).item())
                grad_stats[name]["maxabs"].append(g.abs().max().item())

        global_iter += 1

        grad_sq_sum = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_sq_sum += p.grad.detach().pow(2).sum().item()
        grad_norm = grad_sq_sum ** 0.5

        optimizer.step()

        batch_loss = loss.item()
        sum_train_loss += batch_loss * xb.shape[0]

        iter_logs.append({
            "epoch": epoch,
            "iter_in_epoch": it_in_epoch,
            "batch_loss": batch_loss,
            "grad_norm": grad_norm,
            "batch_size": xb.shape[0],
        })
        it_in_epoch += 1

    avg_train_loss = sum_train_loss / N_train

    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(x_val), y_val).item()

    epoch_logs.append({
        "epoch": epoch,
        "avg_train_loss": avg_train_loss,
        "val_loss": val_loss,
    })

    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1:3d}  avg_train_loss={avg_train_loss:.4f}  val_loss={val_loss:.4f}")



# Plot the training loss as a function of iterations.


import matplotlib.pyplot as plt

# extract iteration-level training loss
train_losses = [log["batch_loss"] for log in iter_logs]
iterations = range(len(train_losses))

plt.figure()
plt.plot(iterations, train_losses)
plt.xlabel("Iteration")
plt.ylabel("Mini-batch training loss")
plt.title("Training loss vs iterations")
plt.show()


epoch_ids = [log["epoch"] for log in epoch_logs]
train_loss = [log["avg_train_loss"] for log in epoch_logs]
val_loss = [log["val_loss"] for log in epoch_logs]

plt.figure()
plt.plot(epoch_ids, train_loss, label="Training loss")
plt.plot(epoch_ids, val_loss, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and validation loss vs epochs")
plt.legend()
plt.show()

# Question E: plot gradient norm per *layer* over training (one figure)
# Better than per-parameter spaghetti: aggregate weight+bias per layer, use log scale, and smooth.

import math

# number of recorded iterations (assumes all params recorded every iteration)
T = len(next(iter(grad_stats.values()))["l2"]) if len(grad_stats) > 0 else 0

# Aggregate per layer: layer_id -> list of L2 norms over iterations
# For a Sequential model, parameter names look like: "0.weight", "0.bias", "2.weight", ...
layer_sq = {}  # layer_id -> list of accumulated squared norms
for name, stats in grad_stats.items():
    layer_id = name.split(".")[0]  # "0", "2", ...
    if layer_id not in layer_sq:
        layer_sq[layer_id] = [0.0] * T

    # stats["l2"] is already ||grad||_2 for that tensor at each iteration
    for t, g_l2 in enumerate(stats["l2"]):
        if g_l2 == g_l2:  # not NaN
            layer_sq[layer_id][t] += g_l2 * g_l2

layer_l2 = {layer_id: [math.sqrt(v) for v in sq_list] for layer_id, sq_list in layer_sq.items()}

def moving_average(x, window=25):
    if window <= 1:
        return x
    out = []
    s = 0.0
    for i, v in enumerate(x):
        s += v
        if i >= window:
            s -= x[i - window]
        out.append(s / min(i + 1, window))
    return out

plt.figure()
for layer_id in sorted(layer_l2.keys(), key=lambda z: int(z) if z.isdigit() else z):
    raw = layer_l2[layer_id]
    smooth = moving_average(raw, window=25)

    # raw (faint) + smoothed (solid)
    plt.plot(raw, alpha=0.25, label=f"layer {layer_id} raw")
    plt.plot(smooth, label=f"layer {layer_id} smoothed")

plt.yscale("log")
plt.xlabel("Iteration")
plt.ylabel("Layer gradient L2 norm")
plt.title("Gradient norm per layer over training (log scale, smoothed)")
plt.legend()
plt.show()

# sub question f



lr_bad = lr * 10
model_bad = nn.Sequential(nn.Linear(1, 16), nn.ReLU(), nn.Linear(16, 1))
opt_bad = torch.optim.SGD(model_bad.parameters(), lr=lr_bad)
loss_fn = nn.MSELoss()

iter_bad, epoch_bad, grad_bad = [], [], {}
N = x_train.shape[0]

for e in range(epochs):
    model_bad.train()
    perm = torch.randperm(N)
    xs, ys = x_train[perm], y_train[perm]
    sum_loss = 0.0

    for i in range(0, N, batch_size):
        xb, yb = xs[i:i+batch_size], ys[i:i+batch_size]
        pred = model_bad(xb)
        loss = loss_fn(pred, yb)

        opt_bad.zero_grad()
        loss.backward()

        for name, p in model_bad.named_parameters():
            if name not in grad_bad:
                grad_bad[name] = []
            g = p.grad.detach()
            grad_bad[name].append(g.norm(2).item())

        opt_bad.step()

        iter_bad.append(loss.item())
        sum_loss += loss.item() * xb.shape[0]

    model_bad.eval()
    with torch.no_grad():
        vloss = loss_fn(model_bad(x_val), y_val).item()

    epoch_bad.append((sum_loss / N, vloss))

# compare losses vs epoch
train_base = [d["avg_train_loss"] for d in epoch_logs]
val_base   = [d["val_loss"] for d in epoch_logs]
train_bad  = [t for t, v in epoch_bad]
val_bad    = [v for t, v in epoch_bad]

plt.figure()
plt.plot(train_base, label="train baseline")
plt.plot(val_base, label="val baseline")
plt.plot(train_bad, label=f"train lr*10 ({lr_bad:g})")
plt.plot(val_bad, label=f"val lr*10 ({lr_bad:g})")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.show()

# compare layer grad-norm curves vs iteration (aggregate weight+bias)
def layer_curve(gs):
    T = len(next(iter(gs.values())))
    layer_sq = {}
    for name, series in gs.items():
        lid = name.split(".")[0]
        layer_sq.setdefault(lid, [0.0]*T)
        for t, v in enumerate(series):
            layer_sq[lid][t] += v*v
    return {lid: [math.sqrt(v) for v in seq] for lid, seq in layer_sq.items()}

layer_base = layer_curve({k: v["l2"] for k, v in grad_stats.items()})
layer_bad  = layer_curve(grad_bad)

plt.figure()
for lid in sorted(layer_base.keys(), key=int):
    plt.plot(layer_base[lid], label=f"baseline layer {lid}")
    plt.plot(layer_bad[lid], label=f"lr*10 layer {lid}")
plt.yscale("log")
plt.xlabel("Iteration"); plt.ylabel("Layer grad L2"); plt.legend(); plt.show()

