"""Multi-class nonlinear temporal task: 4 bits at 4 separated times (8 channels, 2/bit); class =
2*(b1 XOR b2) + (b3 XOR b4) -> 4 classes, each a PAIR of XORs. Linear baselines can read the 4
bits but NOT their XORs -> ~chance (0.25). A substrate that computes the XORs nonlinearly ->
multi-class. Confirms the verified XOR capability scales to a dataset-like multi-class problem."""
import argparse, json, time
import numpy as np, multiprocessing as mp
from neuron.nn_core import NNCore
from neuron.network_config import NetworkConfig
from neuron.neuron import Neuron
from snn_classification_realtime.paula_vision.separability_probe import ridge_acc, lda_acc

_G = {}
CH = {0:(100,200),1:(300,400),2:(500,600),3:(700,120)}  # (bit-> (chan for 0, chan for 1)); 8 chans

def make(n, t1hi, seed):
    rng=np.random.RandomState(seed); tr=[]
    for i in range(n):
        b=[rng.randint(2) for _ in range(4)]
        ts=sorted(rng.choice(range(5,t1hi),4,replace=False))
        ev=[(int(ts[k]), CH[k][b[k]]) for k in range(4)]
        lab=2*(b[0]^b[1])+(b[2]^b[3])
        tr.append((ev,lab))
    rng.shuffle(tr); return tr

def _init(net_path, chan_path, T, K):
    net=NetworkConfig.load_network_config(net_path,neuron_class=Neuron)
    core=NNCore(); core.neural_net=net; core.set_log_level("CRITICAL")
    for nu in net.network.neurons.values(): nu.params.eta_post=0.0; nu.params.eta_retro=0.0
    chan={int(k):v for k,v in json.load(open(chan_path)).items()}
    _G.update(net=net,core=core,chan=chan,T=T,K=K,neurons=list(net.network.neurons.values()))

def _feat(trial):
    ev,lab=trial; net=_G["net"]; core=_G["core"]; chan=_G["chan"]; T=_G["T"]; K=_G["K"]; neurons=_G["neurons"]; N=len(neurons)
    net.reset_simulation(); core.state.current_tick=0; net.current_tick=0
    bt={}
    for (t,c) in ev: bt.setdefault(t,[]).append(c)
    win=np.zeros((K,N),np.float32); cnt=np.zeros(K)
    for t in range(T):
        if t in bt:
            for c in bt[t]:
                for (nid,sid,w) in chan.get(c,[]): net.set_external_input(nid,sid,8.0)
        core.do_tick()
        k=min(K-1,t*K//T); win[k]+=np.array([nu.S for nu in neurons],np.float32); cnt[k]+=1
    win/=np.maximum(cnt[:,None],1); return np.concatenate(win),lab

def baseline(trials,T,K):
    def feat(decay):
        X=[];y=[]
        for ev,lab in trials:
            bt={}
            for (t,c) in ev: bt.setdefault(t,[]).append(c)
            tr=np.zeros(701); win=np.zeros((K,701)); cnt=np.zeros(K)
            for t in range(T):
                tr*=decay
                if t in bt:
                    for c in bt[t]: tr[c]+=1
                k=min(K-1,t*K//T); win[k]+=tr; cnt[k]+=1
            win/=np.maximum(cnt[:,None],1); X.append(win.ravel()); y.append(lab)
        X=np.stack(X);y=np.array(y);n2=len(X)//2
        return ridge_acc(X[:n2],y[:n2],X[n2:],y[n2:])
    return {"leaky.95":feat(0.95),"leaky.99":feat(0.99)}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--net",required=True); ap.add_argument("--chan",required=True)
    ap.add_argument("--T",type=int,default=120); ap.add_argument("--K",type=int,default=6)
    ap.add_argument("--n",type=int,default=800); ap.add_argument("--t1hi",type=int,default=90)
    ap.add_argument("--workers",type=int,default=12); ap.add_argument("--tag",default="mc")
    ap.add_argument("--dump",default="")
    a=ap.parse_args()
    trials=make(a.n,a.t1hi,seed=333)
    print(f"[{a.tag}] 4-class temporal XOR-pairs (chance 0.25). n={a.n}",flush=True)
    b=baseline(trials,a.T,a.K)
    print(f"[{a.tag}] LINEAR BASELINES (should ~=0.25): "+" ".join(f"{k}={v:.3f}" for k,v in b.items()),flush=True)
    with mp.Pool(a.workers,initializer=_init,initargs=(a.net,a.chan,a.T,a.K)) as pool:
        out=pool.map(_feat,trials)
    X=np.stack([o[0] for o in out]); y=np.array([o[1] for o in out]); n2=len(X)//2
    if a.dump:
        np.savez(a.dump, X=X, y=y); print(f"[{a.tag}] dumped feats {X.shape} -> {a.dump}",flush=True)
    acc_r=ridge_acc(X[:n2],y[:n2],X[n2:],y[n2:]); acc_l=lda_acc(X[:n2],y[:n2],X[n2:],y[n2:])
    print(f"[{a.tag}] PAULA reservoir 4-class: ridge={acc_r:.3f} lda={acc_l:.3f} (chance 0.25)",flush=True)

if __name__=="__main__":
    main()
