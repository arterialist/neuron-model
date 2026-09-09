"""Readout heads + external (numpy) substrate-plasticity for the mini-brain sweep.

Two ideas the MNIST-vs-reservoir reconciliation motivates:

1. READOUT. The working conv-PAULA pipeline reaches ~95% on MNIST with a *trained
   nonlinear SNN classifier*, not a single online linear layer. TorchMLPHead is a
   small trained nonlinear readout standing in for that classifier -- it isolates
   whether the linear decoder (not PAULA) was the separability ceiling.

2. EXTERNAL PLASTICITY. Feature-flagged reward-gated rules applied OUTSIDE neuron.py
   to the reservoir's synaptic weights (u_i.info), and a dedicated reward-gated
   association layer. These test whether reward-gated change ACCUMULATES structure
   when it is a proper rule / a layer designed for it -- complementing the native
   `reward_hebb` rule now gated inside neuron.py.

Nothing here changes default behavior: the sweep opts in via MiniBrainConfig flags.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  MLP readout (trained offline on buffered reps) -- the "proper classifier" arm
# --------------------------------------------------------------------------- #
class TorchMLPHead:
    """1-hidden-layer MLP trained on buffered (rep, label) pairs. Full-batch Adam,
    early-stopped on a held-out split. Stands in for the pipeline's SNN classifier."""

    def __init__(self, dim, n_classes=10, hidden=128, seed=0, device="cpu", dropout=0.0):
        """device='cpu' (default, pins 1 thread for the parallel sweep) or 'mps'/'cuda'
        for a big standalone train. 'auto' picks mps>cuda>cpu. dropout=0.0 (default) keeps
        the legacy no-dropout net; >0 inserts Dropout after each ReLU (regularizes big trains)."""
        import torch
        if device == "auto":
            device = ("mps" if torch.backends.mps.is_available()
                      else "cuda" if torch.cuda.is_available() else "cpu")
        if device == "cpu":
            torch.set_num_threads(1)      # one process per core in the sweep
        self.device = device
        self.torch = torch
        g = torch.Generator().manual_seed(seed)
        if dropout > 0.0:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dim, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden, n_classes))
        else:
            self.net = torch.nn.Sequential(
                torch.nn.Linear(dim, hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                torch.nn.Linear(hidden, n_classes))
        for m in self.net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_uniform_(m.weight, a=0.1, generator=g)
                torch.nn.init.zeros_(m.bias)
        self.net.to(device)
        self.n = n_classes

    def fit(self, X, y, epochs=120, lr=3e-3, wd=1e-3, val=0.2, batch=None, stream=False):
        """batch=None -> full-batch (legacy, unchanged). batch=int -> minibatch SGD
        (needed to train well on >~5k samples); patience is counted in epochs either way.
        stream=True keeps X/y on CPU and moves only each minibatch to self.device -- use
        for big trains (e.g. 100k x 9216) that would OOM MPS if resident."""
        t = self.torch
        host = "cpu" if stream else self.device
        X = t.tensor(np.asarray(X), dtype=t.float32).to(host)
        y = t.tensor(np.asarray(y), dtype=t.long).to(host)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        self.mu, self.sd = mu.to(self.device), sd.to(self.device)
        Xn = (X - mu) / sd
        n = len(y); ntr = int(n * (1 - val))
        idx = t.randperm(n)
        tr, te = idx[:ntr], idx[ntr:]
        Xte_d = Xn[te].to(self.device); yte_d = y[te].to(self.device)
        opt = t.optim.Adam(self.net.parameters(), lr=lr, weight_decay=wd)
        lossf = t.nn.CrossEntropyLoss()
        best, best_state, patience = 0.0, None, 0
        for ep in range(epochs):
            self.net.train()
            if batch is None:
                opt.zero_grad()
                loss = lossf(self.net(Xn[tr].to(self.device)), y[tr].to(self.device))
                loss.backward(); opt.step()
            else:
                perm = tr[t.randperm(len(tr))]
                for bs in range(0, len(perm), batch):
                    bi = perm[bs:bs + batch]
                    opt.zero_grad()
                    loss = lossf(self.net(Xn[bi].to(self.device)), y[bi].to(self.device))
                    loss.backward(); opt.step()
            if ep % 4 == 0:
                self.net.eval()
                with t.no_grad():
                    acc = (self.net(Xte_d).argmax(1) == yte_d).float().mean().item()
                if acc > best:
                    best, best_state, patience = acc, [p.detach().clone() for p in self.net.parameters()], 0
                else:
                    patience += 1
                    if patience > 8:
                        break
        if best_state is not None:
            with t.no_grad():
                for p, b in zip(self.net.parameters(), best_state):
                    p.copy_(b)
        return best

    def score(self, X, y):
        t = self.torch
        Xn = (t.tensor(np.asarray(X), dtype=t.float32).to(self.device) - self.mu) / self.sd
        with t.no_grad():
            pred = self.net(Xn).argmax(1).cpu().numpy()
        return float((pred == np.asarray(y)).mean())


# --------------------------------------------------------------------------- #
#  External reward-gated association layer (numpy three-factor) -- Proposal 3
# --------------------------------------------------------------------------- #
class AssociationLayer:
    """A fixed random nonlinear projection (rep -> tanh(W_in rep)) whose OUTPUT
    weights to the decoder are shaped by a three-factor reward-gated Hebbian rule.

    This is the 'dedicated association layer' proposal: reward-gated change on a
    layer built for it, external to neuron.py. The projection is fixed (reservoir-
    style); plasticity lives on an eligibility-trace * reward product, so structure
    accumulates only where activity CO-OCCURS with reward."""

    def __init__(self, dim, out_dim=256, eta=0.01, decay=0.02, trace_tau=0.6, seed=0):
        rng = np.random.RandomState(seed)
        self.W_in = rng.randn(out_dim, dim) / np.sqrt(dim)
        self.b_in = rng.uniform(-0.1, 0.1, out_dim)
        self.A = np.zeros(out_dim)          # associative gain per unit (plastic)
        self.trace = np.zeros(out_dim)      # eligibility trace of unit activity
        self.eta, self.decay, self.tau = eta, decay, trace_tau
        self.out_dim = out_dim

    def project(self, rep):
        h = np.tanh(self.W_in @ rep + self.b_in)
        self.trace = self.tau * self.trace + (1 - self.tau) * h
        return (1.0 + self.A) * h           # associative gain modulates the feature

    def reinforce(self, reward_signal):
        """reward_signal in [-1,1] (e.g. m1 - m0). Grow gain on units that were
        active (eligibility) when reward arrived; linear decay bounds it."""
        self.A += self.eta * (reward_signal * self.trace - self.decay * self.A)
        np.clip(self.A, -2.0, 4.0, out=self.A)


# --------------------------------------------------------------------------- #
#  External synaptic plasticity on the reservoir weights (u_i.info) -- Proposal 3
# --------------------------------------------------------------------------- #
class ExternalReservoirPlasticity:
    """Applies a chosen plasticity rule to the reservoir's synaptic weights each
    fixation, OUTSIDE neuron.py (native eta_post stays 0). Uses the connection graph
    (pre neuron -> post synapse) and per-tick spikes to compute pre/post correlation.

    Rules (all bounded, none multiplicative-in-w-only so none rail to a ceiling):
      oja    : Δw = eta*post*(pre - post*w)              (normalised Hebbian)
      bcm    : Δw = eta*pre*post*(post - theta), theta=<post^2> sliding threshold
      rmhebb : Δw = eta*(m1-m0)*pre*post - eta*decay*w   (reward-modulated Hebbian)
    """

    def __init__(self, brain, rule="oja", eta=0.002, decay=0.1, seed=0):
        self.brain = brain
        self.rule = rule
        self.eta, self.decay = eta, decay
        net = brain.sub.sim.network
        ids = list(net.neurons.keys())
        self.pos = {nid: i for i, nid in enumerate(ids)}
        lay = brain.sub.layer_of_pos
        # target synapses in the reservoir (layer>=1): (post_id, syn_id, pre_pos, post_pos)
        self.syn = []
        for (src, _term, tgt, syn) in net.connections:
            if int(net.neurons[tgt].metadata.get("layer", 0)) >= 1 and src in self.pos:
                self.syn.append((tgt, syn, self.pos[src], self.pos[tgt]))
        self.theta = np.full(brain.sub.num_neurons, 0.05)   # BCM sliding threshold

    def apply(self, states, nm=(0.0, 0.0)):
        """states = list of PopulationState over the fixation; nm = (m0,m1)."""
        O = np.array([s.O for s in states])              # ticks x neurons
        pre_r = O.mean(0)                                # mean rate per unit this fixation
        # BCM threshold update (slow): theta -> <post^2>
        self.theta = 0.98 * self.theta + 0.02 * (pre_r ** 2)
        reward = float(nm[1] - nm[0])
        neurons = self.brain.sub.sim.network.neurons
        for (tgt, syn, pp, qp) in self.syn:
            w = neurons[tgt].postsynaptic_points[syn].u_i.info
            pre, post = pre_r[pp], pre_r[qp]
            if self.rule == "oja":
                dw = self.eta * post * (pre - post * w)
            elif self.rule == "bcm":
                dw = self.eta * pre * post * (post - self.theta[qp])
            else:  # rmhebb
                dw = self.eta * (reward * pre * post - self.decay * (w - 1.0))
            neurons[tgt].postsynaptic_points[syn].u_i.info = float(np.clip(w + dw, -5.0, 10.0))
