"""PAULA spiking LOGIC layer — the CPU foundation. Bit convention: a line HELD (driven every tick
over the eval window) = logic 1; silent = 0. A gate neuron SPIKES (>=THR times in the window) = 1.
A tonic 'VDD' powers inverting gates. Truth-table verified. Gates compose into memory/arith/control."""
import numpy as np
from paula_agent import ckit as k
W=24           # eval window ticks
THR=3          # >=THR output spikes in window => logic 1
VDD=3.0        # tonic power level
HI=4.0         # driven-line amplitude

def _run(path, hi_lines, T=None, extra_drives=None, mod=None, probe=None):
    """hi_lines: list of (neuron_id, synapse_id) to hold HIGH over the window."""
    T=T or (W+8)
    d={}
    for t in range(4,4+W):
        for (nid,sid) in hi_lines: d.setdefault(t,[]).append((nid,sid,HI))
    if extra_drives:
        for t,lst in extra_drives.items(): d.setdefault(t,[]).extend(lst)
    o=k.simulate(path,T,drives=d,mod=mod,probe_ids=probe)
    return o
def _bit(train): return int(sum(train)>=THR)

# ---- primitive gates (single neuron = threshold logic) ----
def g_and():
    ne=[k.neuron(1,r=6.0,lam=3,c=1)]; sy=[k.syn(1,0,1.0,1),k.syn(1,1,1.0,1),k.term(1)]
    return k.build(ne,sy,[],[k.ext(1,0),k.ext(1,1)])
def g_or():
    ne=[k.neuron(1,r=0.8,lam=3,c=1)]; sy=[k.syn(1,0,1.0,1),k.syn(1,1,1.0,1),k.term(1)]
    return k.build(ne,sy,[],[k.ext(1,0),k.ext(1,1)])
def g_not():
    # tonic VDD keeps it firing; input inhibits -> NOT
    ne=[k.neuron(1,r=0.8,lam=3,c=1)]; sy=[k.syn(1,0,1.0,1),k.syn(1,1,-6.0,1),k.term(1)]
    return k.build(ne,sy,[],[k.ext(1,0),k.ext(1,1)])  # syn0=VDD, syn1=input
def g_buffer():
    ne=[k.neuron(1,r=0.8,lam=3,c=1)]; sy=[k.syn(1,0,1.5,1),k.term(1)]
    return k.build(ne,sy,[],[k.ext(1,0)])
def g_nand():
    # NAND = OR(NOT a, NOT b). n1=NOT a, n2=NOT b, n3=OR.
    ne=[k.neuron(1,r=0.8,lam=3,c=1),k.neuron(2,r=0.8,lam=3,c=1),k.neuron(3,r=0.8,lam=3,c=1)]
    sy=[k.syn(1,0,1.0,1),k.syn(1,1,-6.0,1),k.term(1),     # NOT a: VDD, a
        k.syn(2,0,1.0,1),k.syn(2,1,-6.0,1),k.term(2),     # NOT b: VDD, b
        k.syn(3,0,1.5,1),k.syn(3,1,1.5,1),k.term(3)]      # OR(n1,n2)
    conns=[k.conn(1,3,0),k.conn(2,3,1)]
    return k.build(ne,sy,conns,[k.ext(1,0),k.ext(1,1),k.ext(2,0),k.ext(2,1)])
def g_nor():
    ne=[k.neuron(1,r=0.8,lam=3,c=1),k.neuron(2,r=0.8,lam=3,c=1)]
    sy=[k.syn(1,0,1.0,1),k.syn(1,1,1.0,1),k.term(1),
        k.syn(2,0,1.5,1),k.syn(2,1,-15.0,1),k.term(2)]
    return k.build(ne,sy,[k.conn(1,2,1)],[k.ext(1,0),k.ext(1,1),k.ext(2,0)])
def g_xor():
    # XOR = (A OR B) AND NOT(A AND B): n1=OR, n2=AND, n3 = n1 AND NOT n2
    ne=[k.neuron(1,r=0.8,lam=3,c=1),k.neuron(2,r=6.0,lam=3,c=1),k.neuron(3,r=0.9,lam=3,c=1)]
    sy=[k.syn(1,0,1.0,1),k.syn(1,1,1.0,1),k.term(1),          # OR
        k.syn(2,0,1.0,1),k.syn(2,1,1.0,1),k.term(2),          # AND
        k.syn(3,0,2.5,1),k.syn(3,1,-15.0,1),k.term(3)]        # 3 = OR AND NOT AND
    conns=[k.conn(1,3,0),k.conn(2,3,1)]
    return k.build(ne,sy,conns,[k.ext(1,0),k.ext(1,1),k.ext(2,0),k.ext(2,1)])
def g_xnor():
    # XNOR = OR(AND(a,b), NOR(a,b)). n1=AND, n2=OR, n3=NOT(OR)=NOR, n4=OR(AND,NOR)
    ne=[k.neuron(1,r=6.0,lam=3,c=1),k.neuron(2,r=0.8,lam=3,c=1),k.neuron(3,r=0.8,lam=3,c=1),k.neuron(4,r=0.6,lam=5,c=1)]
    sy=[k.syn(1,0,1.0,1),k.syn(1,1,1.0,1),k.term(1),      # AND(a,b)
        k.syn(2,0,1.0,1),k.syn(2,1,1.0,1),k.term(2),      # OR(a,b)
        k.syn(3,0,0.8,1),k.syn(3,1,-16.0,1),k.term(3),    # NOR = VDD, strongly inhibited by OR
        k.syn(4,0,3.5,1),k.syn(4,1,3.5,1),k.term(4)]      # OR(AND,NOR)
    conns=[k.conn(2,3,1),k.conn(1,4,0),k.conn(3,4,1)]
    return k.build(ne,sy,conns,[k.ext(1,0),k.ext(1,1),k.ext(2,0),k.ext(2,1),k.ext(3,0)])
def g_majority():
    # majority of 3 inputs: fires if >=2 high
    ne=[k.neuron(1,r=6.0,lam=3,c=1)]; sy=[k.syn(1,0,1.0,1),k.syn(1,1,1.0,1),k.syn(1,2,1.0,1),k.term(1)]
    return k.build(ne,sy,[],[k.ext(1,0),k.ext(1,1),k.ext(1,2)])
def g_imply():
    # A -> B  ==  (NOT A) OR B
    ne=[k.neuron(1,r=0.8,lam=3,c=1),k.neuron(2,r=0.8,lam=3,c=1)]
    sy=[k.syn(1,0,1.5,1),k.syn(1,1,-15.0,1),k.term(1),         # NOT A
        k.syn(2,0,1.0,1),k.syn(2,1,1.0,1),k.term(2)]          # OR(notA,B)
    conns=[k.conn(1,2,0)]
    return k.build(ne,sy,conns,[k.ext(1,0),k.ext(1,1),k.ext(2,1)])  # 1.syn0=VDD,1.syn1=A ; 2.syn1=B

# ---- truth-table verification ----
def tt2(gate_fn, lines_for, expected, vdd_lines=None):
    """lines_for(a,b)-> list of (nid,sid) to hold HIGH; expected: dict (a,b)->bit; vdd_lines held always."""
    ok=True; rows=[]
    for a in (0,1):
        for b in (0,1):
            p=gate_fn(); hi=list(vdd_lines or [])
            hi+=lines_for(a,b)
            o=_run(p,hi,probe=[max(n for (n,_) in ([(1,0)]+hi))])  # probe highest neuron id (output)
            # output neuron = the last neuron; find it
            outid=max(k.load(p)[0].network.neurons.keys())
            o=_run(p,hi,probe=[outid]); got=_bit(o[outid])
            rows.append(f"{a}{b}->{got}"); ok = ok and got==expected[(a,b)]
    return ok,rows

def verify_all():
    res=[]
    AB=lambda a,b:[(1,0)]*a+[(1,1)]*b  # helper not used; explicit below
    # AND
    res.append(("AND", *tt2(g_and, lambda a,b:([(1,0)] if a else [])+([(1,1)] if b else []),
                            {(0,0):0,(0,1):0,(1,0):0,(1,1):1})))
    res.append(("OR", *tt2(g_or, lambda a,b:([(1,0)] if a else [])+([(1,1)] if b else []),
                           {(0,0):0,(0,1):1,(1,0):1,(1,1):1})))
    # NOT: input on 1.syn1, VDD on 1.syn0 always
    res.append(("NOT", *tt2(lambda:g_not(), lambda a,b:([(1,1)] if a else []),
                            {(0,0):1,(0,1):1,(1,0):0,(1,1):0}, vdd_lines=[(1,0)])))
    res.append(("NAND", *tt2(g_nand, lambda a,b:([(1,1)] if a else [])+([(2,1)] if b else []),
                             {(0,0):1,(0,1):1,(1,0):1,(1,1):0}, vdd_lines=[(1,0),(2,0)])))
    res.append(("NOR", *tt2(g_nor, lambda a,b:([(1,0)] if a else [])+([(1,1)] if b else []),
                            {(0,0):1,(0,1):0,(1,0):0,(1,1):0}, vdd_lines=[(2,0)])))
    res.append(("XOR", *tt2(g_xor, lambda a,b:([(1,0),(2,0)] if a else [])+([(1,1),(2,1)] if b else []),
                            {(0,0):0,(0,1):1,(1,0):1,(1,1):0})))
    res.append(("XNOR", *tt2(g_xnor, lambda a,b:([(1,0),(2,0)] if a else [])+([(1,1),(2,1)] if b else []),
                             {(0,0):1,(0,1):0,(1,0):0,(1,1):1}, vdd_lines=[(3,0)])))
    res.append(("IMPLY", *tt2(g_imply, lambda a,b:([(1,1)] if a else [])+([(2,1)] if b else []),
                              {(0,0):1,(0,1):1,(1,0):0,(1,1):1}, vdd_lines=[(1,0)])))
    return res
if __name__=="__main__":
    import sys; sys.path.insert(0,".")
    ok=0; tot=0
    for name,passed,rows in verify_all():
        tot+=1; ok+=bool(passed); print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {' '.join(rows)}",flush=True)
    print(f"LOGIC GATES: {ok}/{tot} verified",flush=True)
