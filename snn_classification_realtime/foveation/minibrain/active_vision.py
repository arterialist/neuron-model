"""Active vision: a learned multi-glimpse gaze over a FROZEN conv substrate.

Single-glimpse classification caps near the conv ceiling (~0.22-0.32). The way past
it is temporal: take a SEQUENCE of saccades, let the frozen reservoir integrate the
glimpses (its real job -- temporal memory a single snapshot can't hold), and learn
WHERE to look next from reward. Nothing about the movement is hardcoded: a
linear-Gaussian policy maps the current substrate state + gaze proprioception to a
saccade, trained by REINFORCE on the teacher's reward.

Learnable: the gaze policy + the decoder. The substrate stays frozen (measured:
plasticity there only saturates/erodes). This is the "learn where to look" thesis.
"""

from __future__ import annotations

import numpy as np


class GazePolicy:
    """Linear-Gaussian saccade policy trained by REINFORCE with a reward baseline.
    Input = [substrate readout | gaze proprioception (cy,cx normalised)].

    Mean is BOUNDED via tanh (mean = max_step*tanh(W x + b)) so the action clip is
    never the binding constraint -- an unbounded linear mean + clipped action makes
    REINFORCE run away (|W| -> 1e8). Input is standardised online and the gradient
    is norm-clipped; both keep the policy-gradient stable."""

    def __init__(self, dim, lr=0.01, sigma=0.4, max_step=8.0, bl_tau=0.02, seed=0):
        self.W = np.zeros((2, dim)); self.b = np.zeros(2)
        self.lr = lr; self.sigma = sigma; self.max_step = max_step   # sigma is a
        self.astd = sigma * max_step                                 # frac of max_step
        self.baseline = 0.0; self.bl_tau = bl_tau
        self.rng = np.random.RandomState(seed)
        self.mu = np.zeros(dim); self.var = np.ones(dim); self.n = 0

    def _norm(self, x):
        self.n += 1
        self.mu += (x - self.mu) / self.n
        self.var += ((x - self.mu) ** 2 - self.var) / self.n
        return (x - self.mu) / (np.sqrt(self.var) + 1e-6)

    def act(self, x, explore=True):
        xn = self._norm(x)
        pre = self.W @ xn + self.b
        mean = self.max_step * np.tanh(pre)              # bounded saccade (dy,dx) px
        noise = self.astd * self.rng.randn(2) if explore else 0.0
        action = np.clip(mean + noise, -self.max_step, self.max_step)
        return action, (xn, pre, mean)

    def update(self, traj, reward):
        """traj = [((xn,pre,mean), action), ...] for one image; REINFORCE w/ baseline."""
        adv = float(np.clip(reward - self.baseline, -1.0, 1.0))
        self.baseline += self.bl_tau * (reward - self.baseline)
        for (xn, pre, mean), action in traj:
            self._step((xn, pre, mean), action, adv)

    def update_dense(self, traj, gamma=0.9):
        """traj = [((xn,pre,mean), action, r_step), ...] with DENSE per-step rewards
        (e.g. teacher distance-to-object shaping). REINFORCE on returns-to-go with a
        running baseline -- gives a learning signal at every saccade, not just the end."""
        G = 0.0; returns = []
        for _, _, r in reversed(traj):
            G = r + gamma * G; returns.append(G)
        returns.reverse()
        for (aux, action, _), Gt in zip(traj, returns):
            adv = float(np.clip(Gt - self.baseline, -2.0, 2.0))
            self.baseline += self.bl_tau * (Gt - self.baseline)
            self._step(aux, action, adv)

    def _step(self, aux, action, adv):
        xn, pre, mean = aux
        dmean_dpre = self.max_step * (1.0 - np.tanh(pre) ** 2)          # tanh derivative
        g = ((action - mean) / (self.astd ** 2)) * dmean_dpre           # d logN/d pre
        dW = np.outer(g, xn)
        nrm = np.linalg.norm(dW)
        if nrm > 1.0:
            dW /= nrm                                                    # grad clip
        self.W += self.lr * adv * dW
        self.b += self.lr * adv * g


class MLPGazePolicy:
    """Nonlinear (1-hidden-layer tanh) Gaussian saccade policy, REINFORCE-trained.
    Same interface as GazePolicy. The hidden layer lets the policy compose the
    'where' gist into a saccade that a linear map cannot -- the ceiling on foveation
    precision the linear policy hit. Bounded tanh mean + online input standardisation
    + grad-norm clip, exactly as the linear policy, so it stays stable."""

    def __init__(self, dim, hidden=32, lr=0.01, sigma=0.4, max_step=8.0, bl_tau=0.02, seed=0):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(hidden, dim) / np.sqrt(dim); self.b1 = np.zeros(hidden)
        self.W2 = rng.randn(2, hidden) / np.sqrt(hidden); self.b2 = np.zeros(2)
        self.lr = lr; self.sigma = sigma; self.max_step = max_step
        self.astd = sigma * max_step
        self.baseline = 0.0; self.bl_tau = bl_tau
        self.rng = rng
        self.mu = np.zeros(dim); self.var = np.ones(dim); self.n = 0

    def _norm(self, x):
        self.n += 1
        self.mu += (x - self.mu) / self.n
        self.var += ((x - self.mu) ** 2 - self.var) / self.n
        return (x - self.mu) / (np.sqrt(self.var) + 1e-6)

    def act(self, x, explore=True):
        xn = self._norm(x)
        h = np.tanh(self.W1 @ xn + self.b1)
        pre = self.W2 @ h + self.b2
        mean = self.max_step * np.tanh(pre)
        noise = self.astd * self.rng.randn(2) if explore else 0.0
        action = np.clip(mean + noise, -self.max_step, self.max_step)
        return action, (xn, h, pre, mean)

    def _step(self, aux, action, adv):
        xn, h, pre, mean = aux
        dmean_dpre = self.max_step * (1.0 - np.tanh(pre) ** 2)
        g = ((action - mean) / (self.astd ** 2)) * dmean_dpre        # d logN / d pre
        dW2 = np.outer(g, h); db2 = g
        dh = (self.W2.T @ g) * (1.0 - h ** 2)                        # backprop through tanh
        dW1 = np.outer(dh, xn); db1 = dh
        for arr in (dW1, dW2):
            nrm = np.linalg.norm(arr)
            if nrm > 1.0:
                arr /= nrm
        self.W2 += self.lr * adv * dW2; self.b2 += self.lr * adv * db2
        self.W1 += self.lr * adv * dW1; self.b1 += self.lr * adv * db1

    def update(self, traj, reward):
        adv = float(np.clip(reward - self.baseline, -1.0, 1.0))
        self.baseline += self.bl_tau * (reward - self.baseline)
        for aux, action in traj:
            self._step(aux, action, adv)

    def update_dense(self, traj, gamma=0.9):
        G = 0.0; returns = []
        for _, _, r in reversed(traj):
            G = r + gamma * G; returns.append(G)
        returns.reverse()
        for (aux, action, _), Gt in zip(traj, returns):
            adv = float(np.clip(Gt - self.baseline, -2.0, 2.0))
            self.baseline += self.bl_tau * (Gt - self.baseline)
            self._step(aux, action, adv)

    @property
    def W(self):   # for the |W_g| diagnostic used by the experiments
        return self.W1


class RecurrentGazePolicy(MLPGazePolicy):
    """Memory-augmented policy: the hidden activation from the previous saccade is fed
    back as extra input, so the policy carries state ACROSS saccades within one image
    (where it has been, what it saw) -- a leaky recurrence. Gradient is 1-step
    truncated (h_prev treated as a fixed input each step), which keeps it stable
    without fragile full BPTT. reset_state() is called per image."""

    def __init__(self, dim, hidden=32, **kw):
        super().__init__(dim + hidden, hidden=hidden, **kw)
        self.h_dim = hidden
        self.core_dim = dim
        self._h_prev = np.zeros(hidden)

    def reset_state(self):
        self._h_prev = np.zeros(self.h_dim)

    def act(self, x, explore=True):
        xin = np.concatenate([x, self._h_prev])
        action, aux = super().act(xin, explore=explore)
        self._h_prev = aux[1].copy()          # carry hidden state forward
        return action, aux


def present_active(brain, image, label, policy, *, n_saccades=5, ticks_per_sacc=12,
                   start="center", start_pos=None, learn_gaze=True, train_decoder=True,
                   teach=False, explore=True, record=None):
    """One image, a sequence of saccades over the FROZEN substrate.

    Substrate is NOT reset (continuous/time-based); gaze re-centres per image (a new
    object). Reward = decoder correct on the integrated final state. Returns
    (pred, reward, x_final).
    """
    H = brain.H
    if start_pos is not None:
        brain.retina.set_center(*start_pos)               # oracle / fixed fixation
    elif start == "center":
        brain.retina.center()
    else:
        brain.retina.set_center(brain._gaze_rng.uniform(0, H), brain._gaze_rng.uniform(0, H))
    traj, all_states = [], []
    x_sacc = None
    for s in range(n_saccades):
        sig = brain.sub.patch_to_signals(brain._encode(image))
        states = [brain.sub.step(sig + brain._tonic) for _ in range(ticks_per_sacc)]
        all_states.extend(states)
        x_sacc = brain._rep(states)                      # readout after this glimpse
        prop = np.array([brain.retina.cy / H - 0.5, brain.retina.cx / H - 0.5])
        pin = np.concatenate([x_sacc, prop])
        action, aux = policy.act(pin, explore=explore)
        traj.append((aux, action))
        if record is not None:
            record.append(dict(saccade=s, cy=brain.retina.cy, cx=brain.retina.cx,
                               fovea_box=brain.retina.fovea_box(),
                               periph_box=brain.retina.periph_box(),
                               O=np.array([st.O for st in states]),
                               action=action.copy()))
        brain.retina.move(*action)                       # execute saccade
    # integrated read = mean over the WHOLE sequence (frozen reservoir memory)
    x_final = brain._rep(all_states)
    pred = brain.decoder.predict(x_final)
    reward = float(pred == label)
    if train_decoder:
        brain.decoder.update(x_final, label)
    if learn_gaze:
        policy.update(traj, reward)
    return pred, reward, x_final


def _periph_gist(brain, image):
    """Retinotopic peripheral view (blurred whole-canvas gist) at current gaze --
    the 'where' signal: it encodes the object's location relative to the fovea, which
    the scrambled reservoir readout does not. Feeds the saccade policy (dorsal path)."""
    r = brain.retina.render(image)                     # (2C, grid, grid)
    C = r.shape[0] // 2
    return r[C:].flatten().numpy()                     # periphery channels


def periph_dim(brain):
    return brain.C * brain.cfg.grid * brain.cfg.grid


def present_flm(brain, image, label, policy, obj_center, *, max_saccades=6,
                ticks_per_sacc=8, memorize_ticks=30, lock_std=None, canvas=72,
                where_signal="periph", memorize_fovea_only=True,
                learn_gaze=True, train_decoder=True, record=None):
    """Find -> Lock -> Memorize (the user's controller).

    FIND: saccade to search; the teacher gives a DENSE potential-shaped reward for
    reducing the fovea's distance to the object (it knows obj_center) -- this breaks
    the sparse-reward trap that pins a blind policy at chance.
    LOCK: when the readout's F_avg (the FE proxy) settles (std over recent ticks <
    lock_std), the substrate has locked onto a stable percept -> stop searching.
    MEMORIZE: hold the gaze, let the frozen reservoir build a stable trace, and read
    the decoder off THAT settled state (not the transient search glimpses).

    Returns (pred, reward, locked_at_saccade|None, final_distance_px).
    """
    H = brain.H
    traj = []
    locked_at = None
    if hasattr(policy, "reset_state"):
        policy.reset_state()                               # recurrent policy: clear per-image memory
    if obj_center is not None and max_saccades == 0:       # ORACLE: fixate object, skip search
        brain.retina.set_center(*obj_center)
        sig = brain.sub.patch_to_signals(brain._encode(image, fovea_only=memorize_fovea_only))
        mem = [brain.sub.step(sig + brain._tonic) for _ in range(memorize_ticks)]
        x_mem = brain._rep(mem); pred = brain.decoder.predict(x_mem)
        if train_decoder:
            brain.decoder.update(x_mem, label)
        return pred, float(pred == label), 0, 0.0
    brain.retina.center()

    def dist():
        return float(np.hypot(brain.retina.cy - obj_center[0],
                              brain.retina.cx - obj_center[1]))
    prev = dist()
    for s in range(max_saccades):
        where = _periph_gist(brain, image) if where_signal == "periph" else None
        sig = brain.sub.patch_to_signals(brain._encode(image))
        states = [brain.sub.step(sig + brain._tonic) for _ in range(ticks_per_sacc)]
        x = brain._rep(states)
        prop = np.array([brain.retina.cy / H - 0.5, brain.retina.cx / H - 0.5])
        feat = where if where is not None else x       # dorsal 'where' vs cortical readout
        action, aux = policy.act(np.concatenate([feat, prop]))
        now = dist()
        r_step = (prev - now) / canvas                 # dense find reward (teacher)
        traj.append((aux, action, r_step)); prev = now
        if record is not None:
            record.append(dict(phase="find", saccade=s, cy=brain.retina.cy,
                               cx=brain.retina.cx, fovea_box=brain.retina.fovea_box(),
                               periph_box=brain.retina.periph_box(),
                               O=np.array([st.O for st in states]), dist=now))
        # FE-proxy lock: readout F_avg settled?
        fe = np.array([float(st.F_avg[brain._readout_mask].mean()) for st in states])
        if lock_std is not None and s >= 1 and fe[-min(4, len(fe)):].std() < lock_std:
            locked_at = s; break
        brain.retina.move(*action)                     # execute saccade (else: hold)
    # MEMORIZE: hold gaze, integrate a stable trace for the classifier. Read the
    # FOVEA alone (periphery clutter suppressed) so the object drives the readout.
    sig = brain.sub.patch_to_signals(brain._encode(image, fovea_only=memorize_fovea_only))
    mem = [brain.sub.step(sig + brain._tonic) for _ in range(memorize_ticks)]
    if record is not None:
        record.append(dict(phase="memorize", cy=brain.retina.cy, cx=brain.retina.cx,
                           fovea_box=brain.retina.fovea_box(),
                           periph_box=brain.retina.periph_box(),
                           O=np.array([st.O for st in mem]), dist=dist()))
    x_mem = brain._rep(mem)
    pred = brain.decoder.predict(x_mem)
    reward = float(pred == label)
    if train_decoder:
        brain.decoder.update(x_mem, label)
    if learn_gaze and traj:
        aux, a, r = traj[-1]                            # terminal classification bonus
        traj[-1] = (aux, a, r + reward)
        policy.update_dense(traj)
    return pred, reward, locked_at, dist()
