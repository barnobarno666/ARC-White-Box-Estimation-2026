"""P5-04: global-scalar control + carrier selection via analytic distance (lawful)."""
import numpy as np
from whestbench.dataset import load_dataset, resolve_seed_context
from whestbench.domain import MLP
from scipy.stats import norm

def wmc_samples(mlp, n=4200, seed=None):
    d = mlp.width; half=n//2
    rng = np.random.default_rng(seed if seed is not None else mlp.seed)
    xh = rng.standard_normal((half,d)).astype(np.float64)
    x = np.concatenate([xh,-xh],axis=0)
    gram=(x.T@x)/float(n)
    ev,U=np.linalg.eigh(gram); ev=np.maximum(ev,1e-6)
    W=(U*(ev**-0.5))@U.T
    w0=np.array(mlp.weights[0],dtype=np.float64)
    act=np.maximum(x@(W@w0),0.0)
    for l in range(1,mlp.depth):
        w=np.array(mlp.weights[l],dtype=np.float64)
        act=np.maximum(act@w,0.0)
        if l==mlp.depth-2:
            Hpen=act.copy()
    # need preactivations final: recompute last layer pre
    wL=np.array(mlp.weights[-1],dtype=np.float64)
    Z=Hpen@wL; Y=np.maximum(Z,0.0)
    return Y, Z, Hpen

def hadamard_samples(mlp, n=4096):
    d=mlp.width
    # Sylvester Hadamard 1024 with random sign/permutation scramble from mlp.seed
    rng=np.random.default_rng(mlp.seed+999)
    # build Hadamard via Kronecker (2^10=1024)
    H=np.array([[1.0,1.0],[1.0,-1.0]])
    while H.shape[0]<1024:
        H=np.kron(H,np.array([[1.0,1.0],[1.0,-1.0]]))
    H=H/np.sqrt(1024.0)
    perm=rng.permutation(1024); signs=rng.choice([-1.0,1.0],size=1024)
    Q=H[perm,:]*signs[None,:]  # scrambled orthogonal
    Q=np.concatenate([Q,-Q],axis=0)[:n,:]  # antithetic pairs (2048 pairs if n=4096)
    mu_R=31.992187
    # random radius per row (Chi) vs fixed mu_R: test fixed (as before)
    X=Q*mu_R
    act=np.maximum(X@np.array(mlp.weights[0],dtype=np.float64),0.0)
    for l in range(1,mlp.depth):
        act=np.maximum(act@np.array(mlp.weights[l],dtype=np.float64),0.0)
    return act.mean(axis=0)

def blended_c_numpy(mlp):
    width=mlp.width; mu=np.zeros(width); cov=np.eye(width)
    C_C=0.80*0.20*0.07957747154594767; C_T=0.20*0.5
    for w_ in mlp.weights:
        w=np.array(w_,dtype=np.float64)
        mu_pre=w.T@mu; cov_pre=w.T@(cov@w)
        var_pre=np.maximum(np.diag(cov_pre),1e-12); sig=np.sqrt(var_pre)
        a=mu_pre/sig; phi=norm.pdf(a); cdf=norm.cdf(a)
        mu=mu_pre*cdf+sig*phi
        second=(mu_pre**2+var_pre)*cdf+mu_pre*sig*phi
        var_post=np.maximum(second-mu**2,0)
        cov=(cdf[:,None]*cdf[None,:])*cov_pre
        inv=1.0/sig; u=phi/sig
        K=C_C*np.outer(inv,inv)+C_T*np.outer(u,u)
        cov=cov+K*(cov_pre*cov_pre)
        np.fill_diagonal(cov,var_post)
    return mu

def main():
    ds=load_dataset(r"D:\ALL CODES\AICROWD COMPETITION\datasets\mini",split="mini")
    v,s=resolve_seed_context(ds); s0=0.998319
    print("=== P5-04a global-scalar split control ===")
    raws,cvs=[],[]
    for i in range(8):
        row=ds[i]; mlp=MLP.from_row(row,seed_protocol_version=v,seed_salt=s)
        y=np.array(row["final_means"],dtype=np.float64)
        Y,Z,_=wmc_samples(mlp)
        m_raw=Y.mean(axis=0); raw=float(np.mean((m_raw-y)**2))
        idx=np.random.default_rng(0).permutation(4200); A,B=idx[:2100],idx[2100:]
        # pooled global beta across neurons using A-half only (then apply cross-fit both ways, average)
        def pooled_beta(Yh,Zh):
            yc=Yh-Yh.mean(axis=0); zc=Zh-Zh.mean(axis=0)
            return float((yc*zc).sum()/((zc*zc).sum()+1e-12))
        bA=pooled_beta(Y[A],Z[A]); bB=pooled_beta(Y[B],Z[B])
        mA=Y[A].mean(axis=0)-bB*(Z[A].mean(axis=0)-Z[B].mean(axis=0))
        mB=Y[B].mean(axis=0)-bA*(Z[B].mean(axis=0)-Z[A].mean(axis=0))
        m_cv=0.5*(mA+mB); cv=float(np.mean((m_cv-y)**2))
        raws.append(raw); cvs.append(cv)
        print(f" {row['mlp_name']}: raw={raw:.4e} gcv={cv:.4e} ratio={cv/raw:.3f} bA={bA:.3f} bB={bB:.3f}")
    print(f"MEAN raw={np.mean(raws):.4e} gcv={np.mean(cvs):.4e} gain={(1-np.mean(cvs)/np.mean(raws))*100:+.2f}%")
    print("\n=== P5-04b carrier selection WMC vs Had via analytic distance ===")
    for i in range(8):
        row=ds[i]; mlp=MLP.from_row(row,seed_protocol_version=v,seed_salt=s)
        y=np.array(row["final_means"],dtype=np.float64)
        c0=blended_c_numpy(mlp)*s0
        Yw,_,_=wmc_samples(mlp); mw=Yw.mean(axis=0)
        mh=hadamard_samples(mlp)
        ew=float(np.mean((mw-y)**2)); eh=float(np.mean((mh-y)**2))
        dw=float(np.sum((mw-c0)**2)); dh=float(np.sum((mh-c0)**2))
        pick="WMC" if dw<dh else "HAD"; epick=ew if dw<dh else eh
        ebest=min(ew,eh)
        print(f" {row['mlp_name']}: wmc={ew:.4e} had={eh:.4e} |distW={dw:.4e} distH={dh:.4e} lawfulPick={pick} pickMSE={epick:.4e} oracleBest={ebest:.4e}")
if __name__=="__main__":
    main()
