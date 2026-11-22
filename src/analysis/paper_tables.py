"""Regenerates every functional-validity table of the paper (v5 main-corpus re-run and
v6 GramBeddings/BiLSTM re-runs) from the per-URL result files, and the syntactic-versus-functional
comparison (Table S40). Writes one Markdown file per table to --outdir.

Usage (from the repository root):
    python src/analysis/paper_tables.py --main results_v5/tables --main_syntactic results/tables \
        --gram results_v6/gram --bilstm results_v6/bilstm --outdir results_v6/paper_tables

Tables written: Table3, Table4, Table5, Table7, Table8 (main text) and S1, S2, S7, S8, S9, S9b, S11,
S12, S14, S15, S19, S20, S20b, S22, S26-equivalent counts, S28, S34, S35, S39, S40.
Definitions follow §4.7 of the paper: reference baseline = best unguided method at 10 edits on the
seed's 10,000-URL sample (main corpus) or on the 500-URL sample (replications); exact McNemar tests
with Benjamini-Hochberg correction (families as stated in each function).
"""
import argparse, json, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from urllib.parse import urlsplit
from scipy.stats import binomtest, wilcoxon, spearmanr

SEEDS=[0,1,2,3,4]; UNG=['random_insertion','random_substitution','domain_path_tld_transform','random_deletion','adjacent_swap','segment_manipulation']
RK=['occlusion','ig','saliency','gradxinput','random']; MODES=['one_shot','adaptive']; BUD=[1,2,3,5,8,10]
def succ(df,b=10): return df['targeted_edits'].apply(lambda e: pd.notna(e) and e<=b).to_numpy()
def bh(p):
    p=np.asarray(p,float); out=np.full_like(p,np.nan); ok=~np.isnan(p); n=ok.sum()
    if n==0: return out
    idx=np.where(ok)[0]; o=idx[np.argsort(p[idx])]; m=1.0
    for k in range(n-1,-1,-1):
        m=min(m,p[o[k]]*n/(k+1)); out[o[k]]=m
    return out
def mcn(a,b):
    n=a+b
    return 1.0 if n==0 else binomtest(min(a,b),n,0.5).pvalue
def rb(x,y):
    d=np.asarray(x,float)-np.asarray(y,float); d=d[d!=0]
    if len(d)==0: return np.nan
    r=pd.Series(np.abs(d)).rank().to_numpy(); return (r[d>0].sum()-r[d<0].sum())/r.sum()
def auc(asrs,b=BUD):
    b=np.array(b,float); return float(np.trapezoid(asrs,b)/(b[-1]-b[0]))
class R:
    def __init__(s,root): s.root=Path(root); s.c={}
    def _r(s,p):
        if p not in s.c: s.c[p]=pd.read_csv(s.root/p)
        return s.c[p]
    def ung(s,seed): return s._r(f'seed{seed}/unguided_attack_results.csv')
    def ungm(s,seed,m): d=s.ung(seed); return d[d.method==m].set_index('url')
    def var(s,seed,rk,md,prot=False): return s._r(f'seed{seed}/variants{"_protected" if prot else ""}/{rk}_{md}.csv').set_index('url')
    def hf(s,seed,md,prot=False): return s._r(f'seed{seed}/hotflip{"_protected" if prot else ""}/hotflip_{md}.csv').set_index('url')
    def ungp(s,seed,m): return s._r(f'seed{seed}/unguided_protected/{m}.csv').set_index('url')
    def ig_init(s,seed,md):
        d=s._r(f'seed{seed}/guided_attack_results.csv'); return d[d['mode']==md].set_index('url')
    def faith(s,seed,prot=False): return s._r(f'seed{seed}/faithfulness_by_ranker{"_protected" if prot else ""}.csv')
    def urls500(s,seed): return s.var(seed,'ig','one_shot').index
    def ref(s,seed):
        d=s.ung(seed); a={m:succ(d[d.method==m]).mean() for m in UNG}; return max(a,key=a.get)
    def pref(s,seed):
        u=s.urls500(seed); a={m:succ(s.ungp(seed,m).loc[u]).mean() for m in UNG}; return max(a,key=a.get)
def ms(x,d=4): x=np.asarray(x,float); return f"{x.mean():.{d}f}±{x.std(ddof=1):.{d}f}"

# ---- from t3.py
def table3(root,guided='var'):
    r=R(root); rows={}
    per={}
    for s in SEEDS:
        ref=r.ref(s); u=r.urls500(s); refd=r.ungm(s,ref).loc[u]; rs=succ(refd)
        names=MODES+UNG; ps=[]; info={}
        for n in names:
            if n in MODES:
                d=(r.ig_init(s,n) if guided=='init' else r.var(s,'ig',n)).loc[u]
            else: d=r.ungm(s,n)
            rr=refd if n in MODES else r.ungm(s,ref).loc[d.index]
            rs_=succ(rr); ms_=succ(d)
            if n==ref: ps.append(np.nan); info[n]=dict(ref=True); continue
            a=int((ms_&~rs_).sum()); b=int((~ms_&rs_).sum()); ps.append(mcn(a,b))
            j=ms_&rs_; es=rb(d['targeted_edits'][j],rr['targeted_edits'][j]) if j.sum()>=2 else np.nan
            info[n]=dict(ref=False,a=a,b=b,es=es,above=ms_.mean()>rs_.mean())
        q=bh(ps)
        for n,qq in zip(names,q): info[n]['q']=qq
        per[s]=info
    out=[]
    for n in MODES+UNG:
        asr=[];au=[]
        for s in SEEDS:
            if n in MODES:
                d=(r.ig_init(s,n) if guided=='init' else r.var(s,'ig',n)).loc[r.urls500(s)]
            else: d=r.ungm(s,n)
            asr.append(succ(d).mean()); au.append(auc([succ(d,b).mean() for b in BUD]))
        es=[per[s][n]['es'] for s in SEEDS if not per[s][n]['ref']]
        nonref=[s for s in SEEDS if not per[s][n]['ref']]
        sig=[s for s in nonref if per[s][n]['q']<0.05]
        beat=[s for s in sig if per[s][n]['a']>per[s][n]['b']]; under=[s for s in sig if per[s][n]['a']<per[s][n]['b']]
        out.append((n,ms(asr),ms(au),f"{np.nanmean(es):+.4f}±{np.nanstd(es,ddof=1):.4f}",f"{len(sig)}/{len(nonref)}",",".join(map(str,beat)) or '---',",".join(map(str,under)) or '---', [round(x,3) for x in asr],[round(x,3) for x in es]))
    return out,per

# ---- from t3b.py
from scipy.stats import wilcoxon
def run(root, guided='var'):
    r=R(root); per=[]
    for s in SEEDS:
        ref=r.ref(s); u=r.urls500(s); rows=[]
        for n in MODES+UNG:
            d=(r.var(s,'ig',n) if guided=='var' else r.ig_init(s,n)).loc[u] if n in MODES else r.ungm(s,n)
            rr=r.ungm(s,ref).loc[d.index]; ms_=succ(d); rs=succ(rr)
            row=dict(method=n,seed=s,asr=ms_.mean() if n in MODES else succ(r.ungm(s,n)).mean(),ref=(n==ref))
            if n!=ref:
                a=int((ms_&~rs).sum()); b=int((~ms_&rs).sum()); row.update(a=a,b=b,pm=mcn(a,b))
                j=ms_&rs; x=d['targeted_edits'][j].to_numpy(); y=rr['targeted_edits'][j].to_numpy()
                try: pw=wilcoxon(x,y,zero_method='wilcox').pvalue
                except ValueError: pw=np.nan
                row.update(pw=pw,es=rb(x,y))
            rows.append(row)
        D=pd.DataFrame(rows); D['qm']=bh(D.get('pm',np.nan)); D['qw']=bh(D.get('pw',np.nan)); per.append(D)
    return pd.concat(per)
def f(p):
    if pd.isna(p): return '—'
    return f"{p:.4f}" if p>=1e-4 else f"{p:.2e}"

# ---- from rc.py
# Ranking control (Table 4/S8), V1 (Table 5/S9/S9b), HotFlip (Table 8/S15) from per-URL data
NM={'occlusion':'Occlusion','ig':'Integrated Gradients','saliency':'Saliency','gradxinput':'Gradient × input','random':'Random ranking'}
def v0(r):
    rows=[]
    for s in SEEDS:
        ref=r.ref(s); u=r.urls500(s); rh=succ(r.ungm(s,ref).loc[u])
        sr=[]
        for md in MODES:
            base=succ(r.var(s,'ig',md).loc[u])
            for rk in RK:
                h=succ(r.var(s,rk,md).loc[u]); d=dict(seed=s,mode=md,ranker=rk,asr=h.mean(),ref_asr=rh.mean(),ref=ref)
                a=int((h&~rh).sum()); b=int((~h&rh).sum()); d.update(vr_only=a,ref_only=b,p_ref=mcn(a,b))
                if rk!='ig':
                    a2=int((base&~h).sum()); b2=int((~base&h).sum()); d.update(ig_only=a2,var_only=b2,p_ig=mcn(a2,b2))
                else: d.update(p_ig=np.nan)
                sr.append(d)
        D=pd.DataFrame(sr); D['q_ref']=bh(D.p_ref); D['q_ig']=bh(D.p_ig); rows.append(D)
    return pd.concat(rows)
def v1(r):
    rows=[]
    for s in SEEDS:
        u=r.urls500(s); pr=r.pref(s); rh=succ(r.ungp(s,pr).loc[u])
        for md in MODES:
            H={rk:succ(r.var(s,rk,md,True).loc[u]) for rk in RK}; sr=[]
            for rk in RK:
                h=H[rk]; d=dict(seed=s,mode=md,ranker=rk,asr=h.mean(),pref_asr=rh.mean(),pref=pr)
                a=int((h&~rh).sum()); b=int((~h&rh).sum()); d.update(vr_only=a,ref_only=b,p_ref=mcn(a,b))
                for c in ('ig','random'):
                    if rk!=c:
                        a2=int((h&~H[c]).sum()); b2=int((~h&H[c]).sum()); d.update({f'a_{c}':a2,f'b_{c}':b2,f'p_{c}':mcn(a2,b2)})
                    else: d[f'p_{c}']=np.nan
                sr.append(d)
            D=pd.DataFrame(sr)
            for c in ('ref','ig','random'): D['q_'+c]=bh(D['p_'+c])
            rows.append(D)
    return pd.concat(rows)
def hotflip(r,prot):
    rows=[]
    for md in MODES:
        sr=[]
        for s in SEEDS:
            u=r.urls500(s); hf=succ(r.hf(s,md,prot).loc[u])
            refn=r.pref(s) if prot else r.ref(s); rh=succ((r.ungp(s,refn) if prot else r.ungm(s,refn)).loc[u])
            comps={'ref':rh}; comps.update({rk:succ(r.var(s,rk,md,prot).loc[u]) for rk in RK})
            for c,h in comps.items():
                a=int((hf&~h).sum()); b=int((~hf&h).sum())
                sr.append(dict(seed=s,mode=md,comp=c,hf=hf.mean(),other=h.mean(),a=a,b=b,p=mcn(a,b)))
        D=pd.DataFrame(sr); D['q']=bh(D.p); rows.append(D)
    return pd.concat(rows)

# ---- from fv.py
MET=[('deletion_auc',-1),('comprehensiveness',1),('sufficiency',-1)]
def pts(r,prot):
    rows=[]
    for s in SEEDS:
        f=r.faith(s,prot).groupby('ranker')[[m for m,_ in MET]].mean()
        u=r.urls500(s)
        for md in MODES:
            for rk in RK:
                rows.append(dict(seed=s,mode=md,ranker=rk,asr=succ(r.var(s,rk,md,prot).loc[u]).mean(),**f.loc[rk].to_dict()))
    return pd.DataFrame(rows)
def corr(P,excl=()):
    out={}
    for md in MODES:
        g=P[(P['mode']==md)&(~P.ranker.isin(excl))]
        for m,sg in MET:
            rho,p=spearmanr(g[m],g.asr); w=[spearmanr(h[m],h.asr)[0] for _,h in g.groupby('seed')]
            out[(md,m)]=(rho,p,np.nanmean(w),sum(1 for x in w if np.sign(x)==sg),len(g))
    return out
def fmt(t): rho,p,w,k,n=t; return f"{rho:+.2f}{' (ns)' if p>=0.05 else ''} / {w:+.2f} [{k}/5]"

# ---- from rep.py
SEEDS3=[0,1,2]
class RR(R):
    def urls500(s,seed): return s.ung(seed).url.unique()
    def ref(s,seed):
        d=s.ung(seed); a={m:succ(d[d.method==m]).mean() for m in UNG}; return max(a,key=a.get)
    def pref(s,seed):
        a={m:succ(s.ungp(seed,m)).mean() for m in UNG}; return max(a,key=a.get)
def get(r,s,rk,md,prot):
    u=r.urls500(s)
    if rk=='hotflip': return succ(r.hf(s,md,prot).loc[u])
    return succ(r.var(s,rk,md,prot).loc[u])
def analyse(r,hot=True):
    rows=[]
    M=RK+(['hotflip'] if hot else [])
    for s in SEEDS3:
        u=r.urls500(s)
        for prot in (False,True):
            rh=succ((r.ungp(s,r.pref(s)) if prot else r.ungm(s,r.ref(s))).loc[u])
            for md in MODES:
                H={m:get(r,s,m,md,prot) for m in M}
                sr=[]
                for m in M:
                    h=H[m]; a=int((h&~rh).sum()); b=int((~h&rh).sum())
                    d=dict(seed=s,prot=prot,mode=md,m=m,asr=h.mean(),ref=rh.mean(),a=a,b=b,p=mcn(a,b))
                    if m!='random':
                        a2=int((h&~H['random']).sum()); b2=int((~h&H['random']).sum()); d.update(ar=a2,br=b2,pr=mcn(a2,b2))
                    else: d['pr']=np.nan
                    if m!='ig':
                        a3=int((h&~H['ig']).sum()); b3=int((~h&H['ig']).sum()); d.update(ai=a3,bi=b3,pi=mcn(a3,b3))
                    else: d['pi']=np.nan
                    sr.append(d)
                D=pd.DataFrame(sr); D['q']=bh(D.p); D['qr']=bh(D.pr); D['qi']=bh(D.pi); rows.append(D)
    return pd.concat(rows)
def cnt(g,a,b,q):
    up=((g[a]>g[b])&(g[q]<.05)).sum(); dn=((g[a]<g[b])&(g[q]<.05)).sum(); return f"{(g[a]>g[b]).sum()}/3 ({up}↑/{dn}↓)", up, dn

NM={'occlusion':'Occlusion','ig':'Integrated Gradients','saliency':'Saliency','gradxinput':'Gradient × input','random':'Random ranking','hotflip':'HotFlip (adapted)'}

def main():
    global args
    ap=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default="results_v5/tables", help="functional-predicate main-corpus results")
    ap.add_argument("--main_syntactic", default="results/tables", help="syntactic-predicate main-corpus results (Table S40)")
    ap.add_argument("--gram", default="results_v6/gram"); ap.add_argument("--bilstm", default="results_v6/bilstm")
    ap.add_argument("--outdir", default="results_v6/paper_tables")
    args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    O=build()
    for k,v in O.items():
        if isinstance(v,str): (out/((("Table"+k[1:]) if k[0]=="T" else k)+".md")).write_text(v+"\n")
    print("wrote", len([v for v in O.values() if isinstance(v,str)]), "tables to", out)

def build():
    O={}
    r=R(args.main)
    NM={'occlusion':'Occlusion','ig':'Integrated Gradients','saliency':'Saliency','gradxinput':'Gradient × input','random':'Random ranking'}
    MD={'one_shot':'One-shot','adaptive':'Adaptive'}
    def m4(x): x=np.asarray(x,float); return f"{x.mean():.4f}±{x.std(ddof=1):.4f}"
    def m3(x): x=np.asarray(x,float); return f"{x.mean():.3f}±{x.std(ddof=1):.3f}"
    def sg(v): return ('−' if v<0 else '+')+f"{abs(v):.4f}"
    # ---- Table 3
    rows,per=table3(args.main)
    L=["| Method | ASR-targeted | Budget-AUC | Effect size | Sig | Beat ref | Underperf. ref |","|---|---|---|---|---|---|---|"]
    lab={'one_shot':'one-shot','adaptive':'adaptive'}
    for n,a,au,es,sig,beat,under,_,esl in rows:
        if sig=='0/0': es='— (reference in every seed)'; sig='—'
        else:
            e=np.array(esl,float); es=f"{sg(e.mean())}±{e.std(ddof=1):.4f}"
        L.append(f"| {lab.get(n,n)} | {a} | {au} | {es} | {sig} | {beat.replace(',',', ')} | {under.replace(',',', ')} |")
    O['T3']="\n".join(L)
    # ---- Table 4 & S8
    A=v0(r)
    L=["| Guidance | Ranking method | ASR (mean±SD) | Above reference (sig.) | Sig. below reference | IG higher (sig.) | Method sig. higher than IG |","|---|---|---|---|---|---|---|"]
    S8=["| Guidance | Ranking method | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean |","|---|---|---|---|---|---|---|---|"]
    for md in MODES:
        for rk in RK:
            g=A[(A['mode']==md)&(A.ranker==rk)].sort_values('seed')
            above=(g.asr>g.ref_asr).sum(); sa=((g.asr>g.ref_asr)&(g.q_ref<.05)).sum(); sb=((g.asr<g.ref_asr)&(g.q_ref<.05)).sum()
            if rk=='ig': igc='—'; msc='—'
            else: igc=f"{(g.ig_only>g.var_only).sum()}/5 ({((g.ig_only>g.var_only)&(g.q_ig<.05)).sum()})"; msc=str(((g.ig_only<g.var_only)&(g.q_ig<.05)).sum())
            L.append(f"| {MD[md]} | {NM[rk]} | {m4(g.asr)} | {above}/5 ({sa}) | {sb} | {igc} | {msc} |")
            S8.append(f"| {MD[md]} | {NM[rk]} | "+" | ".join(f"{x:.3f}" for x in g.asr)+f" | {g.asr.mean():.4f} |")
    refv=A.groupby('seed').ref_asr.first()
    S8.append("| Reference | Reference baseline (domain_path_tld_transform, same 500 URLs) | "+" | ".join(f"{x:.3f}" for x in refv)+f" | {refv.mean():.4f} |")
    O['T4']="\n".join(L); O['S8']="\n".join(S8); O['refv']=list(refv)
    # ---- Table 5, S9, S9b
    B=v1(r)
    L=["| Guidance | Ranking method | V1 ASR (mean±SD) | Change from V0 | Above protected reference baseline (sig.) | Above random (sig.) | IG higher (sig.) | Method sig. higher than IG |","|---|---|---|---|---|---|---|---|"]
    S9=["| Guidance | Ranking method | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean |","|---|---|---|---|---|---|---|---|"]
    for md in MODES:
        for rk in RK:
            g=B[(B['mode']==md)&(B.ranker==rk)].sort_values('seed'); g0=A[(A['mode']==md)&(A.ranker==rk)]
            ch=f"{(g.asr.mean()/g0.asr.mean()-1)*100:+.0f}%".replace('-','−')
            ab=f"{(g.asr>g.pref_asr).sum()}/5 ({((g.asr>g.pref_asr)&(g.q_ref<.05)).sum()})"
            ar='—' if rk=='random' else f"{(g.a_random>g.b_random).sum()}/5 ({((g.a_random>g.b_random)&(g.q_random<.05)).sum()})"
            if rk=='ig': ih='—'; mh='—'
            else: ih=f"{(g.a_ig<g.b_ig).sum()}/5 ({((g.a_ig<g.b_ig)&(g.q_ig<.05)).sum()})"; mh=str(((g.a_ig>g.b_ig)&(g.q_ig<.05)).sum())
            L.append(f"| {MD[md]} | {NM[rk]} | {m4(g.asr)} | {ch} | {ab} | {ar} | {ih} | {mh} |")
            S9.append(f"| {MD[md]} | {NM[rk]} | "+" | ".join(f"{x:.3f}" for x in g.asr)+f" | {g.asr.mean():.4f} |")
    O['T5']="\n".join(L); O['S9']="\n".join(S9)
    pv=B.groupby('seed').pref_asr.first(); O['prefv']=list(pv); O['pref']=list(B.groupby('seed').pref.first())
    # rand below protected ref sig
    O['rand_below_pref']={md:int(((B['mode']==md)&(B.ranker=='random')&(B.asr<B.pref_asr)&(B.q_ref<.05)).sum()) for md in MODES}
    S9b=["| Protected unguided baseline | Seed 0 | Seed 1 | Seed 2 | Seed 3 | Seed 4 | Mean |","|---|---|---|---|---|---|---|"]
    for m in sorted(UNG):
        v=[succ(r.ungp(s,m).loc[r.urls500(s)]).mean() for s in SEEDS]
        S9b.append(f"| {m} | "+" | ".join(f"{x:.3f}" for x in v)+f" | {np.mean(v):.4f} |")
    S9b.append("| Protected reference baseline (best per seed) | "+" | ".join(f"{x:.3f}" for x in pv)+f" | {pv.mean():.4f} |")
    O['S9b']="\n".join(S9b)
    # ---- Table 7 & S12
    def fmt7(t): rho,p,w,k,n=t; return f"{rho:+.2f}{' (ns)' if p>=0.05 else ''} / {w:+.2f} [{k}/5]".replace('+-','−').replace('-','−')
    def fp(t,br=False):
        rho,p,w,k,n=t; ps="p < 0.001" if p<0.001 else f"p = {p:.3f}"
        return (f"{rho:+.2f} ({ps})"+(f" [{k}/5]" if br else "")).replace('-','−')
    L=["| Setting | Guidance | Deletion AUC (all 5 methods) | Comprehensiveness (all) | Sufficiency (all) | Deletion AUC (excl. occlusion) | Comprehensiveness (excl.) | Sufficiency (excl.) |","|---|---|---|---|---|---|---|---|"]
    S12=["| Setting | Guidance | Rankings included | Deletion AUC | Comprehensiveness | Sufficiency |","|---|---|---|---|---|---|"]
    S12b=[]
    M=['deletion_auc','comprehensiveness','sufficiency']
    for prot in (False,True):
        P=pts(r,prot); st='V1' if prot else 'V0'
        ca=corr(P); ce=corr(P,('occlusion',)); cr=corr(P,('random',)); ci=corr(P,('occlusion','random'))
        for md in MODES:
            L.append(f"| {st} | {MD[md]} | "+" | ".join(fmt7(ca[(md,m)]) for m in M)+" | "+" | ".join(fmt7(ce[(md,m)]) for m in M)+" |")
            S12.append(f"| {st} | {MD[md]} | all five | "+" | ".join(fp(ca[(md,m)]) for m in M)+" |")
            S12.append(f"| {st} | {MD[md]} | without occlusion | "+" | ".join(fp(ce[(md,m)]) for m in M)+" |")
            S12b.append(f"| {st} | {MD[md]} | without random ranking | "+" | ".join(fp(cr[(md,m)],1) for m in M)+" |")
            S12b.append(f"| {st} | {MD[md]} | gradient-based rankers only | "+" | ".join(fp(ci[(md,m)],1) for m in M)+" |")
    O['T7']="\n".join(L); O['S12']="\n".join(S12+S12b)
    # ---- Table 8 & S15
    L=["| Method | V0 one-shot | V0 adaptive | V1 one-shot | V1 adaptive |","|---|---|---|---|---|"]
    H={p:hotflip(r,p) for p in (False,True)}
    lab8={'ref':'Reference baseline (V1: protected)','random':'Random ranking (control)','saliency':'Saliency','gradxinput':'Gradient × input','ig':'Integrated Gradients','occlusion':'Occlusion'}
    for c in ['ref','random','saliency','gradxinput','ig','occlusion']:
        cells=[]
        for p in (False,True):
            for md in MODES:
                g=H[p][(H[p]['mode']==md)&(H[p].comp==c)]
                cells.append(f"{m3(g.other)} ({((g.a>g.b)&(g.q<.05)).sum()}/{((g.a<g.b)&(g.q<.05)).sum()})")
        L.append(f"| {lab8[c]} | "+" | ".join(cells)+" |")
    cells=[]
    for p in (False,True):
        for md in MODES:
            g=H[p][(H[p]['mode']==md)&(H[p].comp=='ref')]; cells.append(m3(g.hf))
    L.append("| HotFlip (adapted) | "+" | ".join(cells)+" |")
    O['T8']="\n".join(L)
    S15=["| Setting | Guidance | Seed | HotFlip | Reference | Random | Saliency | Grad×input | IG | Occlusion |","|---|---|---|---|---|---|---|---|---|---|"]
    for p in (False,True):
        for md in MODES:
            for s in SEEDS:
                g=H[p][(H[p]['mode']==md)&(H[p].seed==s)].set_index('comp')
                S15.append(f"| {'V1' if p else 'V0'} | {MD[md]} | {s} | {g.hf.iloc[0]:.3f} | "+" | ".join(f"{g.loc[c,'other']:.3f}" for c in ['ref','random','saliency','gradxinput','ig','occlusion'])+" |")
    O['S15']="\n".join(S15)
    # ---- S2
    P=run(args.main)
    nm={'one_shot':'one-shot','adaptive':'adaptive'}
    def f(p):
        if pd.isna(p): return '—'
        return f"{p:.4f}" if p>=1e-4 else f"{p:.2e}"
    P['mn']=P.method.map(lambda m: nm.get(m,m))
    L=["| Method | Seed | ASR-targeted | McNemar p (FDR) | Wilcoxon p (FDR) | Effect size |","|---|---|---|---|---|---|"]
    for _,x in P.sort_values(['mn','seed']).iterrows():
        if x.ref: L.append(f"| {x.mn} | {x.seed} | {x.asr:.4f} | (reference) | — | — |")
        else: L.append(f"| {x.mn} | {x.seed} | {x.asr:.4f} | {f(x.qm)} | {f(x.qw)} | {(('−' if x.es<0 else '')+f'{abs(x.es):.4f}') if pd.notna(x.es) else '—'} |")
    O['S2']="\n".join(L)
    O['wilc']={n:dict(more=int(((P.method==n)&(P.qw<.05)&(P.es>0)).sum()),fewer=int(((P.method==n)&(P.qw<.05)&(P.es<0)).sum())) for n in MODES}

    r4=R(args.main_syntactic)
    NM={'occlusion':'Occlusion','ig':'Integrated Gradients','saliency':'Saliency','gradxinput':'Gradient × input','random':'Random ranking','hotflip':'HotFlip (adapted)'}
    MD={'one_shot':'One-shot','adaptive':'Adaptive'}
    PRE=re.compile(r"^(?:[^:/?#]*://)?(?:www\.)?", re.I)
    CL=['phishing','defacement','malware']
    def frame(prot,rk,md,s):
        u=r.urls500(s)
        if rk=='ref': d=(r.ungp(s,r.pref(s)) if prot else r.ungm(s,r.ref(s))).loc[u]
        elif rk=='hotflip': d=r.hf(s,md,prot).loc[u]
        else: d=r.var(s,rk,md,prot).loc[u]
        d=d.copy(); d['hit']=succ(d); d['pfx']=[PRE.match(x).group(0) for x in d.index]; d['hp']=d.pfx.str.len()>0
        d['chg']=[not str(f).startswith(p) for f,p in zip(d.final_url,d.pfx)]; return d
    # S39 per-class ASR
    L=["| Setting | Guidance | Method | Phishing | Defacement | Malware | All |","|---|---|---|---|---|---|---|"]
    for prot in (False,True):
        for md in MODES:
            for rk in ['ref','random','saliency','gradxinput','ig','occlusion','hotflip']:
                if rk=='ref' and md=='adaptive': continue
                vals={c:[] for c in CL+['all']}
                for s in SEEDS:
                    d=frame(prot,rk,md,s)
                    for c in CL: vals[c].append(d[d.true_label==c].hit.mean())
                    vals['all'].append(d.hit.mean())
                name=('Protected reference baseline' if prot else 'Reference baseline') if rk=='ref' else NM[rk]
                g='—' if rk=='ref' else MD[md]
                L.append(f"| {'V1' if prot else 'V0'} | {g} | {name} | "+" | ".join(f"{np.mean(vals[c]):.3f}" for c in CL+['all'])+" |")
    O['S39']="\n".join(L)
    # S11 prefix diagnostic, all seeds, V0
    L=["| Method | Guidance | URLs with a prefix | ASR, URLs with a prefix | ASR, URLs without a prefix | Prefix changed: successes | Prefix changed: failures |","|---|---|---|---|---|---|---|"]
    for rk in ['ref','random','saliency','gradxinput','ig','occlusion','hotflip']:
        for md in MODES:
            if rk=='ref' and md=='adaptive': continue
            a=[];b=[];c=[];e=[];f=[]
            for s in SEEDS:
                d=frame(False,rk,md,s); p=d[d.hp]
                a.append(d.hp.mean()); b.append(p.hit.mean()); c.append(d[~d.hp].hit.mean()); e.append(p[p.hit].chg.mean()); f.append(p[~p.hit].chg.mean())
            name='Reference baseline' if rk=='ref' else NM[rk]
            L.append(f"| {name} | {'—' if rk=='ref' else MD[md]} | {np.mean(a):.3f} | {np.mean(b):.3f} | {np.mean(c):.3f} | {np.mean(e):.2f} | {np.mean(f):.2f} |")
    O['S11']="\n".join(L)
    # S40 syntactic vs functional
    L=["| Setting | Guidance | Method | Syntactic predicate | Functional predicate | Relative change |","|---|---|---|---|---|---|"]
    for prot in (False,True):
        for md in MODES:
            for rk in ['ref','random','saliency','gradxinput','ig','occlusion','hotflip']:
                if rk=='ref' and md=='adaptive': continue
                a=[];b=[]
                for s in SEEDS:
                    u=r.urls500(s)
                    if rk=='ref':
                        a.append(succ((r4.ungp(s,r4.pref(s)) if prot else r4.ungm(s,r4.ref(s))).loc[u]).mean()); b.append(succ((r.ungp(s,r.pref(s)) if prot else r.ungm(s,r.ref(s))).loc[u]).mean())
                    elif rk=='hotflip': a.append(succ(r4.hf(s,md,prot).loc[u]).mean()); b.append(succ(r.hf(s,md,prot).loc[u]).mean())
                    else: a.append(succ(r4.var(s,rk,md,prot).loc[u]).mean()); b.append(succ(r.var(s,rk,md,prot).loc[u]).mean())
                name=('Protected reference baseline' if prot else 'Reference baseline') if rk=='ref' else NM[rk]
                L.append(f"| {'V1' if prot else 'V0'} | {'—' if rk=='ref' else MD[md]} | {name} | {np.mean(a):.3f} | {np.mean(b):.3f} | {(np.mean(b)/np.mean(a)-1)*100:+.0f}% |".replace('| -','| −').replace('% |','% |'))
    O['S40']="\n".join(L).replace('| +','| +')
    O['S40']=re.sub(r'\| -(\d)',r'| −\1',O['S40'])
    # S26 pooled McNemar (discordant pairs summed over seeds)
    def pf(p):
        if p<1e-3:
            e=int(np.floor(np.log10(p)))+1; return f"< 10^−{-e}^"
        return f"{p:.3f}"
    L=["| Setting | Guidance | Method | Comparator | Method only | Comparator only | p (exact) |","|---|---|---|---|---|---|---|"]
    for st,D in (('V0',A),('V1',B)):
        for md in MODES:
            g_=D[D['mode']==md]
            if st=='V0': a=g_[g_.ranker=='random'].ig_only.sum(); b=g_[g_.ranker=='random'].var_only.sum()
            else: a=g_[g_.ranker=='ig'].a_random.sum(); b=g_[g_.ranker=='ig'].b_random.sum()
            L.append(f"| {st} | {MD[md]} | Integrated Gradients | Random ranking | {int(a)} | {int(b)} | {pf(mcn(int(a),int(b)))} |")
            for rk in ['ig','occlusion','saliency','gradxinput','random']:
                x=g_[g_.ranker==rk]; a=int(x.vr_only.sum()); b=int(x.ref_only.sum())
                L.append(f"| {st} | {MD[md]} | {NM[rk]} | {'Reference baseline' if st=='V0' else 'Protected reference baseline'} | {a} | {b} | {pf(mcn(a,b))} |")
    O['S26']="\n".join(L)


    # ---- S1, S28, S7 (unguided budget sweep, summary, host-validity sensitivity) and S22 (URL-level)
    L=["| Attack | B=1 | B=2 | B=3 | B=5 | B=8 | B=10 |","|---|---|---|---|---|---|---|"]
    L28=["| Attack | ASR-targeted | ASR-untargeted | Median edits | Mean edits |","|---|---|---|---|---|"]
    for m in sorted(UNG):
        a={b:[succ(r.ungm(s,m),b).mean() for s in SEEDS] for b in BUD}
        L.append(f"| {m} | "+" | ".join(ms(a[b]) for b in BUD)+" |")
        un=[];med=[];mean=[]
        for s in SEEDS:
            d=r.ungm(s,m); un.append(d.untargeted_edits.apply(lambda e: pd.notna(e) and e<=10).mean())
            e=d.targeted_edits[succ(d)]; med.append(e.median()); mean.append(e.mean())
        L28.append(f"| {m} | {ms(a[10])} | {ms(un)} | {np.mean(med):.1f} | {np.mean(mean):.2f} |")
    O['S1']="\n".join(L); O['S28']="\n".join(L28)
    TLD=re.compile(r'^[A-Za-z]{2,24}$')
    def hostok(u):
        try:
            p=urlsplit(u if '://' in u else 'http://'+u); h=p.netloc.split('@')[-1].split(':')[0]
            return bool(TLD.match(h.split('.')[-1])) and len(h.split('.'))>=2
        except ValueError: return False
    L=["| Attack | Full sample | Host-valid URLs only | Relative gain (range over seeds) |","|---|---|---|---|"]
    for m in UNG+MODES:
        full=[];hv=[];gain=[]
        for s in SEEDS:
            d=r.var(s,'ig',m) if m in MODES else r.ungm(s,m)
            ok=np.array([hostok(u) for u in d.index]); sc=succ(d)
            full.append(sc.mean()); hv.append(sc[ok].mean()); gain.append(sc[ok].mean()/sc.mean()-1)
        L.append(f"| {'one-shot' if m=='one_shot' else m} | {ms(full).replace('±',' ± ')} | {ms(hv).replace('±',' ± ')} | {min(gain)*100:.1f}–{max(gain)*100:.1f}% |")
    O['S7']="\n".join(L)
    L=["| Ranking method | V0 one-shot | V0 adaptive | V1 one-shot | V1 adaptive |","|---|---|---|---|---|"]
    for rk in ['ig','occlusion','gradxinput','saliency','random']:
        cells=[]
        for prot in (False,True):
            for md in MODES:
                xs=[];ys=[]
                for s in SEEDS:
                    f_=r.faith(s,prot); f_=f_[f_.ranker==rk].set_index('url'); v=r.var(s,rk,md,prot)
                    u=f_.index.intersection(v.index); xs+=list(f_.loc[u,'deletion_auc']); ys+=list(succ(v.loc[u]))
                cells.append(f"{spearmanr(xs,ys)[0]:+.2f}".replace('-','−'))
        L.append(f"| {NM[rk]} | "+" | ".join(cells)+" |")
    O['S22']="\n".join(L)

    NM={'occlusion':'Occlusion','ig':'Integrated Gradients','saliency':'Saliency','gradxinput':'Gradient × input','random':'Random ranking','hotflip':'HotFlip (adapted)'}
    SN={'occlusion':'Occlusion','ig':'IG','saliency':'Saliency','gradxinput':'Grad×input','random':'Random','hotflip':'HotFlip'}
    MD={'one_shot':'One-shot','adaptive':'Adaptive'}
    def m4r(x): x=np.asarray(x,float); return f"{x.mean():.4f}±{x.std(ddof=1):.4f}"
    rg=RR(args.gram); rb=RR(args.bilstm)
    Ag=analyse(rg,False); Ab=analyse(rb,True)
    def perseed(r,A,hot):
        L=["| Setting | Guidance | Ranking method | Seed 0 | Seed 1 | Seed 2 | Mean |","|---|---|---|---|---|---|---|"]
        for prot in (False,True):
            for md in MODES:
                for m in RK+(['hotflip'] if hot else []):
                    g=A[(A.prot==prot)&(A['mode']==md)&(A.m==m)].sort_values('seed')
                    L.append(f"| {'V1' if prot else 'V0'} | {MD[md]} | {NM[m]} | "+" | ".join(f"{x:.3f}" for x in g.asr)+f" | {g.asr.mean():.4f} |")
        for prot in (False,True):
            g=A[(A.prot==prot)&(A.m=='ig')&(A['mode']=='one_shot')].sort_values('seed')
            names=[(r.pref(s) if prot else r.ref(s)) for s in SEEDS3]
            nm=names[0] if len(set(names))==1 else ", ".join(names)
            L.append(f"| {'V1' if prot else 'V0'} | Reference | {'Protected reference baseline' if prot else 'Reference baseline'} ({nm}) | "+" | ".join(f"{x:.3f}" for x in g.ref)+f" | {g.ref.mean():.4f} |")
        return "\n".join(L)
    O['S14']=perseed(rg,Ag,False); O['S19']=perseed(rb,Ab,True)
    def summ(A,hot,arrows):
        if arrows: L=["| Guidance | Ranking method | V0 ASR (mean±SD) | V0 above reference baseline | V1 ASR (mean±SD) | V1 above protected reference baseline | V1 above random |","|---|---|---|---|---|---|---|"]
        else: L=["| Guidance | Ranking method | V0 ASR (mean±SD) | V0 above reference (sig.) | V1 ASR (mean±SD) | V1 above protected reference baseline (sig.) | V1 above random (sig.) |","|---|---|---|---|---|---|---|"]
        for md in MODES:
            for m in RK+(['hotflip'] if hot else []):
                g0=A[(~A.prot)&(A['mode']==md)&(A.m==m)]; g1=A[(A.prot)&(A['mode']==md)&(A.m==m)]
                def c(g,x,y,q):
                    n=(g[x]>g[y]).sum(); up=((g[x]>g[y])&(g[q]<.05)).sum(); dn=((g[x]<g[y])&(g[q]<.05)).sum()
                    return f"{n}/3 ({up}↑/{dn}↓)" if arrows else f"{n}/3 ({up})"
                cr='—' if m=='random' else c(g1,'ar','br','qr')
                L.append(f"| {MD[md]} | {NM[m]} | {m4r(g0.asr)} | {c(g0,'asr','ref','q')} | {m4r(g1.asr)} | {c(g1,'asr','ref','q')} | {cr} |")
        return "\n".join(L)
    O['S34']=summ(Ag,False,False); O['S35']=summ(Ab,True,True)
    # S20 and S20b
    def fp(q): return "<0.001" if q<0.001 else f"{q:.3f}"
    L=["| Setting | Guidance | Seed | Method | Method only | Reference only | p (FDR) |","|---|---|---|---|---|---|---|"]
    for prot in (False,True):
        for md in MODES:
            for s in SEEDS3:
                for m in ['hotflip','random','saliency','gradxinput','ig','occlusion']:
                    x=Ab[(Ab.prot==prot)&(Ab['mode']==md)&(Ab.seed==s)&(Ab.m==m)].iloc[0]
                    L.append(f"| {'V1' if prot else 'V0'} | {MD[md]} | {s} | {SN[m]} | {x.a} | {x.b} | {fp(x.q)} |")
    O['S20']="\n".join(L)
    L=["| Setting | Guidance | Seed | Method | Method only | Random only | p (FDR) |","|---|---|---|---|---|---|---|"]
    for md in MODES:
        for s in SEEDS3:
            for m in ['saliency','gradxinput','ig','occlusion','hotflip']:
                x=Ab[(Ab.prot)&(Ab['mode']==md)&(Ab.seed==s)&(Ab.m==m)].iloc[0]
                L.append(f"| V1 | {MD[md]} | {s} | {SN[m]} | {int(x.ar)} | {int(x.br)} | {fp(x.qr)} |")
    O['S20b']="\n".join(L)
    return O

if __name__=='__main__':
    main()
