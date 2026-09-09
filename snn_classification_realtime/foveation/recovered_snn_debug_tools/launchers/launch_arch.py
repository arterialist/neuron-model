#!/usr/bin/env python
"""Fan the architecture sweep across 12 cores (1 arm/process, OMP=1), then aggregate.
Each arm = (tag, extra CLI args). See exp_arch.py for the experiment definitions."""
import os, sys, subprocess, time, itertools

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
PY = os.path.join(ROOT, ".venv/bin/python")
OUT = os.path.join(ROOT, "foveation_results/minibrain/arch")
os.makedirs(OUT, exist_ok=True)
N_PAR = 6          # 6-way so each arm runs at full single-core speed and finishes
                   # under the ~70-min background cap (12-way contended -> none finished)
SEED = 0

SEP = f"--exp sep --dwell 500 --warmup 500 --probe-dwell 200 --images 350".split()
PLAST = f"--exp plast --dwell 500 --warmup 500 --probe-dwell 120 --images 400 --probe-every 100 --probe-n 120".split()
GAZE = ("--exp gaze --dwell 500 --warmup 500 --images 300 --probe-every 75 "
        "--ticks-per-sacc 8 --memorize-ticks 250 --fovea 44 --canvas 72 --distractors 4 --grid 16").split()
RETINO = "--arch reservoir_retino --conv-bank rich".split()

ARMS = []
def add(tag, base, extra, animate=False):
    a = list(base) + extra.split() + ["--tag", tag]
    if animate: a += ["--animate", "--anim-images", "12", "--anim-ticks", "50"]
    ARMS.append((tag, a))

# --- Exp1: encoder x readout separability ---
add("sep_rand_gray",   SEP, "--arch reservoir_random --conv-bank rich --dataset cifar10_grayscale")
add("sep_retino_gray", SEP, "--arch reservoir_retino --conv-bank rich --dataset cifar10_grayscale")
add("sep_conv_gray",   SEP, "--arch conv --conv-bank rich --dataset cifar10_grayscale")
add("sep_rand_color",  SEP, "--arch reservoir_random --conv-bank rich --dataset cifar10")
add("sep_retino_color",SEP, "--arch reservoir_retino --conv-bank rich --dataset cifar10", animate=True)
add("sep_conv_color",  SEP, "--arch conv --conv-bank rich --dataset cifar10")
add("sep_retino_mlp_gray",  SEP, "--arch reservoir_retino --conv-bank rich --readout-head mlp --dataset cifar10_grayscale")
add("sep_retino_mlp_color", SEP, "--arch reservoir_retino --conv-bank rich --readout-head mlp --dataset cifar10")
add("sep_retino_def_gray",  SEP, "--arch reservoir_retino --conv-bank default --dataset cifar10_grayscale")
add("sep_conv_mlp_color",   SEP, "--arch conv --conv-bank rich --readout-head mlp --dataset cifar10")

# --- Exp2: substrate-plasticity trajectory (retinotopic, rich bank) ---
add("plast_frozen_gray",       PLAST, "--arch reservoir_retino --conv-bank rich --teach 0 --dataset cifar10_grayscale", animate=True)
add("plast_rhebb_teach_gray",  PLAST, "--arch reservoir_retino --conv-bank rich --unfreeze --plasticity-mode reward_hebb --nm-kappa 2 --eta-post-res 0.01 --teach 1 --dataset cifar10_grayscale", animate=True)
add("plast_rhebb_noteach_gray",PLAST, "--arch reservoir_retino --conv-bank rich --unfreeze --plasticity-mode reward_hebb --nm-kappa 0 --eta-post-res 0.01 --teach 0 --dataset cifar10_grayscale")
add("plast_oja_gray",          PLAST, "--arch reservoir_retino --conv-bank rich --ext-plasticity oja --teach 0 --dataset cifar10_grayscale")
add("plast_bcm_gray",          PLAST, "--arch reservoir_retino --conv-bank rich --ext-plasticity bcm --teach 0 --dataset cifar10_grayscale")
add("plast_rmhebb_gray",       PLAST, "--arch reservoir_retino --conv-bank rich --ext-plasticity rmhebb --teach 1 --dataset cifar10_grayscale")
add("plast_assoc_gray",        PLAST, "--arch reservoir_retino --conv-bank rich --assoc --teach 1 --dataset cifar10_grayscale")
add("plast_frozen_color",      PLAST, "--arch reservoir_retino --conv-bank rich --teach 0 --dataset cifar10")
add("plast_rhebb_teach_color", PLAST, "--arch reservoir_retino --conv-bank rich --unfreeze --plasticity-mode reward_hebb --nm-kappa 2 --eta-post-res 0.01 --teach 1 --dataset cifar10")
add("plast_assoc_color",       PLAST, "--arch reservoir_retino --conv-bank rich --assoc --teach 1 --dataset cifar10")

# --- Exp3: gaze policy (cluttered, grayscale) ---
add("gaze_oracle",    GAZE, "--gaze-mode oracle")
add("gaze_random",    GAZE, "--gaze-mode random")
add("gaze_linear",    GAZE, "--policy linear --gaze-mode flmlock")
add("gaze_mlp",       GAZE, "--policy mlp --gaze-mode flmlock")
add("gaze_recurrent", GAZE, "--policy recurrent --gaze-mode flmlock")


import json
def _done(tag):
    """Arm is done iff its JSON exists AND is marked complete (resumability)."""
    p = os.path.join(OUT, f"{tag}_seed{SEED}.json")
    if not os.path.exists(p):
        return False
    try:
        return bool(json.load(open(p)).get("complete", False))
    except Exception:
        return False


def main():
    env = dict(os.environ, PYTHONPATH=ROOT, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
    pending = [(t, a) for (t, a) in ARMS if not _done(t)]
    skipped = len(ARMS) - len(pending)
    print(f"[launch] {len(ARMS)} arms ({skipped} already complete, {len(pending)} to run), "
          f"{N_PAR}-way, out={OUT}", flush=True)
    t0 = time.time(); running = {}; queue = list(pending); done = 0
    while queue or running:
        while queue and len(running) < N_PAR:
            tag, a = queue.pop(0)
            cmd = [PY, "-m", "snn_classification_realtime.foveation.minibrain.exp_arch",
                   *a, "--seed", str(SEED), "--out", OUT]
            lf = open(os.path.join(OUT, f"{tag}_run.out"), "w")
            running[tag] = (subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=lf, stderr=subprocess.STDOUT), lf, time.time())
            print(f"[start {time.time()-t0:5.0f}s] {tag}  ({len(running)} running, {len(queue)} queued)", flush=True)
        time.sleep(3)
        for tag in list(running):
            proc, lf, ts = running[tag]
            if proc.poll() is not None:
                lf.close(); done += 1
                print(f"[done  {time.time()-t0:5.0f}s] {tag}  rc={proc.returncode}  ({time.time()-ts:.0f}s)  [{done}/{len(pending)}]", flush=True)
                del running[tag]
    print(f"[launch] all arms done in {time.time()-t0:.0f}s; aggregating", flush=True)
    subprocess.run([PY, "-m", "snn_classification_realtime.foveation.minibrain.exp_arch",
                    "--exp", "sep", "--aggregate", "--out", OUT], cwd=ROOT, env=env)
    open(os.path.join(OUT, "SWEEP_DONE"), "w").write(f"done {time.time()-t0:.0f}s\n")
    print("[launch] DONE", flush=True)


if __name__ == "__main__":
    main()
