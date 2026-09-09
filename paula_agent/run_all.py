"""Master regression runner: verify the entire PAULA agent circuit library + the three agents."""
import sys; sys.path.insert(0,".")
def run_batch(mod, listname):
    m=__import__(f"paula_agent.{mod}",fromlist=[listname]); items=getattr(m,listname)
    ok=0
    for name,vf in items:
        try: ok+=bool(vf()[0])
        except Exception: pass
    return ok,len(items)
total_ok=total=0
for mod,ln in [("circuits_a","BATCH_A"),("circuits_b","BATCH_B"),("circuits_c","BATCH_C"),
               ("arch_d","BATCH_D"),("arch_e","BATCH_E"),("arch_f","BATCH_F"),
               ("circuits_g","CIRCUITS"),("circuits_h","CIRCUITS")]:
    ok,n=run_batch(mod,ln); total_ok+=ok; total+=n
    print(f"  {mod:12s}: {ok}/{n}",flush=True)
# logic
from paula_agent import logic
lok=sum(1 for _,p,_ in logic.verify_all() if p); ltot=len(logic.verify_all())
total_ok+=lok; total+=ltot; print(f"  logic       : {lok}/{ltot}",flush=True)
# agents
from paula_agent import agent_sim, agent_learn, agent_full, agent_2d
import numpy as np
res=[agent_sim.episode(s) for s in range(5)]
ate=np.mean([r['eaten'] for r in res]); esc=sum(r['escapes'] for r in res); caught=sum(r['caught'] for r in res)
noesc=sum(agent_sim.episode(s,escape_enabled=False)['caught'] for s in range(5))
a1 = ate>1 and esc>0 and caught==0 and noesc>caught
print(f"  agent_sim   : {'PASS' if a1 else 'FAIL'} (forage {ate:.1f}, escapes {esc}, caught {caught} vs {noesc} no-escape)",flush=True)
h,wA,wB=agent_learn.run(seed=0); a2 = wA>wB*3
print(f"  agent_learn : {'PASS' if a2 else 'FAIL'} (cueA-w {wA:.1f} >> cueB-w {wB:.1f})",flush=True)
log,fwA,fwB,en=agent_full.life(seed=0); da_e=sum(d for _,_,d,_ in log[:len(log)//3]); da_l=sum(d for _,_,d,_ in log[2*len(log)//3:])
a3 = fwA>fwB*3 and da_l<=da_e and en>0
print(f"  agent_full  : {'PASS' if a3 else 'FAIL'} (danger-appr {da_e}->{da_l}, survived {en>0})",flush=True)
nav=agent_2d.build_nav_brain(); nr=[agent_2d.run_agent(nav,s)[0] for s in range(8)]; a4=sum(1 for r in nr if r>=0)>=6
print(f"  agent_2d    : {'PASS' if a4 else 'FAIL'} (reached food {sum(1 for r in nr if r>=0)}/8 in 2D)",flush=True)
from paula_agent import agent_complex
Lc=[agent_complex.life(seed=s,learning=True) for s in range(4)]; Cc=[agent_complex.life(seed=s,learning=False) for s in range(4)]
lifeL=np.mean([r['lifespan'] for r in Lc]); lifeC=np.mean([r['lifespan'] for r in Cc])
toxL=np.mean([r['toxic_eaten'] for r in Lc]); toxC=np.mean([r['toxic_eaten'] for r in Cc])
a5 = lifeL>lifeC and toxL<toxC
print(f"  agent_complex: {'PASS' if a5 else 'FAIL'} (lifespan learn {lifeL:.0f}>{lifeC:.0f} ctrl, toxic {toxL:.1f}<{toxC:.1f})",flush=True)
agents_ok=int(a1)+int(a2)+int(a3)+int(a4)+int(a5)
print(f"\nLIBRARY: {total_ok}/{total} circuits verified | AGENTS: {agents_ok}/5 | TOTAL: {total_ok+agents_ok}/{total+5}",flush=True)
