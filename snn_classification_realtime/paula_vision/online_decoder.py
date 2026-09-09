"""Online, bio-plausible external decoder for a frozen PAULA reservoir. NOT closed-form, NOT
gradient/backprop: a single-layer reward-modulated competitive rule (the reward_hebb three-factor
form, w* ~ nm*pre/rh_decay) applied ONE TRIAL AT A TIME in a streaming pass. Local update
Dw = eta*(pre_activity) gated by reward (correct/incorrect) -> finds a discriminative boundary
(so it solves the MULTI-MODAL XOR classes that a pure Hebbian class-mean prototype cannot).
Reports streaming (online) accuracy as it learns + held-out test accuracy. This is the 'deliverable
must LEARN ONLINE' arm: frozen reservoir = fixed substrate, decoder learns online with no gradient."""
import argparse, numpy as np

def online_reward_competitive(Xtr, ytr, Xte, yte, K, eta=0.02, rh_decay=0.1, epochs=8, seed=0):
    rng = np.random.RandomState(seed)
    D = Xtr.shape[1]
    W = np.zeros((K, D), np.float32)          # class weight vectors, start blank
    b = np.zeros(K, np.float32)
    # standardize features online-safe: use train mean/std (fixed, data-level norm, not label-fit)
    mu = Xtr.mean(0); sd = Xtr.std(0) + 1e-6
    Xtr = (Xtr - mu) / sd; Xte = (Xte - mu) / sd
    online_correct = 0; online_seen = 0; curve = []
    order = np.arange(len(Xtr))
    for ep in range(epochs):
        rng.shuffle(order)
        ep_correct = 0
        for i in order:
            x = Xtr[i]; y = ytr[i]
            scores = W @ x + b
            pred = int(np.argmax(scores))
            if ep == 0:  # first epoch = the true ONLINE (never-seen) accuracy
                online_correct += (pred == y); online_seen += 1
            ep_correct += (pred == y)
            # reward-modulated competitive update (three-factor: pre=x, post=winner, reward=correct)
            # potentiate TRUE class toward x (reward gate ON), decay term = -rh_decay*W (bounded fp)
            W[y] += eta * (x - rh_decay * W[y]); b[y] += eta * (1.0 - rh_decay * b[y])
            if pred != y:                      # anti-reward the wrongly-fired winner (stress)
                W[pred] -= eta * x;            b[pred] -= eta
        # held-out test each epoch
        te_pred = np.argmax(Xte @ W.T + b, 1)
        te_acc = float((te_pred == yte).mean())
        curve.append(te_acc)
    online_acc = online_correct / max(online_seen, 1)
    return online_acc, curve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True); ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--eta", type=float, default=0.02); ap.add_argument("--rh_decay", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=8); ap.add_argument("--tag", default="dec")
    a = ap.parse_args()
    d = np.load(a.feats); X = d["X"].astype(np.float32); y = d["y"].astype(int)
    n2 = len(X) // 2
    Xtr, ytr, Xte, yte = X[:n2], y[:n2], X[n2:], y[n2:]
    chance = 1.0 / a.K
    # closed-form reference on same split
    from snn_classification_realtime.paula_vision.separability_probe import ridge_acc
    cf = ridge_acc(Xtr, ytr, Xte, yte)
    online_acc, curve = online_reward_competitive(Xtr, ytr, Xte, yte, a.K, a.eta, a.rh_decay, a.epochs)
    print(f"[{a.tag}] chance={chance:.3f} closed-form-ridge={cf:.3f}", flush=True)
    print(f"[{a.tag}] ONLINE first-pass (never-seen) acc={online_acc:.3f}", flush=True)
    print(f"[{a.tag}] online-decoder test-acc by epoch: "+" ".join(f"{c:.3f}" for c in curve), flush=True)
    print(f"[{a.tag}] FINAL online-decoder held-out test={curve[-1]:.3f} (chance {chance:.3f})", flush=True)
