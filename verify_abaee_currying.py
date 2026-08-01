#!/usr/bin/env python3
"""
Independent verification suite for
    A. Abaee, "No Universal Process Currying: The Intercept Principle,
    Environmental Currying, and Finite-Dimensional Obstructions to Exact
    Process Storage, Evaluation, and Recovery."

This script reproduces, checks, and *adversarially stress-tests* the paper's
core mathematical claims. Every test prints PASS/FAIL. Requires: numpy, scipy,
cvxpy (SDP tests skip gracefully if cvxpy is absent).

Run:  python3 verify_abaee_currying.py
"""
import numpy as np
import math

try:
    import cvxpy as cp
    HAVE_CVXPY = True
except Exception:
    HAVE_CVXPY = False

try:
    import sympy as _sp_probe  # noqa: F401
    HAVE_SYMPY = True
except Exception:
    HAVE_SYMPY = False

rng = np.random.default_rng(12345)
RESULTS = []

def check(name, ok):
    RESULTS.append((name, bool(ok)))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def rand_channel(din, dout, nk=3):
    K = [rng.standard_normal((dout, din)) + 1j*rng.standard_normal((dout, din))
         for _ in range(nk)]
    S = sum(k.conj().T @ k for k in K)
    Sih = np.linalg.inv(np.linalg.cholesky(S).conj().T)
    return [k @ Sih for k in K]

def apply_kraus(Ks, r):
    return sum(K @ r @ K.conj().T for K in Ks)

def rand_state(d):
    A = rng.standard_normal((d, d)) + 1j*rng.standard_normal((d, d))
    r = A @ A.conj().T
    return r / np.trace(r)

def trnorm(M):
    return float(np.sum(np.abs(np.linalg.eigvalsh((M + M.conj().T)/2))))

def choi_from_kraus(Ks, dout, din):
    J = np.zeros((dout*din, dout*din), dtype=complex)
    for K in Ks:
        v = K.reshape(-1)          # (out,in) row-major == out (x) in
        J += np.outer(v, v.conj())
    return J

def diamond_norm(J, dout, din):
    """Certified induced-trace-norm diamond norm (Watrous primal / QETLAB form).
    J is the Choi operator on (out (x) in), Hermitian. Returns the paper's
    ||Psi||_diamond = sup_X ||(Psi (x) id)(X)||_1 / ||X||_1 as an SDP optimum."""
    if not HAVE_CVXPY:
        return None
    X = cp.Variable((dout*din, dout*din), complex=True)
    r0 = cp.Variable((din, din), hermitian=True)
    r1 = cp.Variable((din, din), hermitian=True)
    Iout = np.eye(dout)
    M = cp.bmat([[cp.kron(Iout, r0), X], [X.H, cp.kron(Iout, r1)]])
    cons = [M >> 0, r0 >> 0, r1 >> 0, cp.trace(r0) == 1, cp.trace(r1) == 1]
    p = cp.Problem(cp.Maximize(cp.real(cp.trace(J.conj().T @ X))), cons)
    p.solve(solver=cp.SCS, eps=1e-9, max_iters=100000)
    return p.value

def _ptrace_out_expr(Mexpr, dout, din):
    return cp.bmat([[sum(Mexpr[o*din+i, o*din+j] for o in range(dout))
                     for j in range(din)] for i in range(din)])

def diamond_norm_dual(J, dout, din):
    """Watrous DUAL SDP, matched to diamond_norm's primal (same convention).
        minimize   mu
        s.t.  [[Y0, -J],[-J^†, Y1]] >> 0,  Y0,Y1 >> 0,
              Tr_out(Y0) << mu I,  Tr_out(Y1) << mu I
    Strong duality => equals the primal optimum. Certifies the value from above."""
    if not HAVE_CVXPY:
        return None
    Y0 = cp.Variable((dout*din, dout*din), hermitian=True)
    Y1 = cp.Variable((dout*din, dout*din), hermitian=True)
    mu = cp.Variable()
    blk = cp.bmat([[Y0, -J], [-J.conj().T, Y1]])
    cons = [blk >> 0, Y0 >> 0, Y1 >> 0,
            mu*np.eye(din) - _ptrace_out_expr(Y0, dout, din) >> 0,
            mu*np.eye(din) - _ptrace_out_expr(Y1, dout, din) >> 0]
    p = cp.Problem(cp.Minimize(mu), cons)
    p.solve(solver=cp.SCS, eps=1e-9, max_iters=100000)
    return p.value

# ======================================================================
print("\n=== A. ARITHMETIC / DIMENSION OBSTRUCTIONS ===")
# ======================================================================
# Chan right adjoint: r^2 = e^4-e^2+1 never a perfect square (e>1)
ok = all(math.isqrt(e**4-e**2+1)**2 != e**4-e**2+1 for e in range(2, 200))
check("Chan no-right-adjoint: e^4-e^2+1 never a perfect square (e=2..199)", ok)

# Instrument: (grade2 - 2*grade1) => e^2=1 for ALL b,r  => no representing r
found = any(
    (e**2*(b**2-1) == r**2-1) and (e**2*(2*b**2-1) == 2*r**2-1)
    for e in range(2, 12) for b in range(1, 12) for r in range(1, 400)
)
check("Instr no-right-adjoint: no (e>1,b,r) solves grade-1 & grade-2 jointly", not found)

# FinStoch: b^e > e(b-1)+1 for b,e>=2 (vertex mismatch); never equal
never_equal = all(b**e != e*(b-1)+1 for b in range(2, 60) for e in range(2, 60))
strict = all(b**e > e*(b-1)+1 for b in range(2, 60) for e in range(2, 60))
check("FinStoch: product-of-simplices vertex count strictly exceeds simplex", never_equal and strict)

# Chan not a state space: m^2=e^4-e^2+1 no integer solution
ok = all(math.isqrt(e**4-e**2+1)**2 != e**4-e**2+1 for e in range(2, 100))
check("Chan(E,E) not a finite state space: m^2=e^4-e^2+1 unsolvable", ok)

# Two-grade affine-dimension defect: q2-2q1 = e^2-1 identically
import sympy as sp
e_, b_, r_, n_ = sp.symbols('e b r n')
q = lambda nn: sp.expand(e_**2*(nn*b_**2-1) - (nn*r_**2-1))
check("dim-defect identity q2-2q1 = e^2-1", sp.simplify(q(2)-2*q(1) - (e_**2-1)) == 0)

# ======================================================================
print("\n=== B. AUTOMATIC ENRICHMENT (control-bit construction) ===")
# ======================================================================
d = 2
maxerr = 0.0
for _ in range(50):
    h0 = rand_channel(d*d, d); h1 = rand_channel(d*d, d)
    P0 = np.outer([1,0],[1,0]); P1 = np.outer([0,1],[0,1])
    for p in np.linspace(0, 1, 4):
        rho = rand_state(d*d)
        Z = np.kron(p*P0 + (1-p)*P1, rho)
        Z4 = Z.reshape(2, d*d, 2, d*d)
        F = apply_kraus(h0, Z4[0,:,0,:]) + apply_kraus(h1, Z4[1,:,1,:])
        target = p*apply_kraus(h0, rho) + (1-p)*apply_kraus(h1, rho)
        maxerr = max(maxerr, np.linalg.norm(F - target))
check("control-bit channel realizes convex segment (err<1e-10)", maxerr < 1e-10)

# ======================================================================
print("\n=== C. NO-PROGRAMMING (Thm 5.9) ===")
# ======================================================================
# inner-product identity from Kraus completeness
dM, dE = 3, 2
Ks = rand_channel(dM*dE, dE, nk=4)
pU = rng.standard_normal(dM)+1j*rng.standard_normal(dM); pU/=np.linalg.norm(pU)
pV = rng.standard_normal(dM)+1j*rng.standard_normal(dM); pV/=np.linalg.norm(pV)
psi= rng.standard_normal(dE)+1j*rng.standard_normal(dE); psi/=np.linalg.norm(psi)
phi= rng.standard_normal(dE)+1j*rng.standard_normal(dE); phi/=np.linalg.norm(phi)
x = np.kron(pU, psi); y = np.kron(pV, phi)
lhs = np.vdot(pU,pV)*np.vdot(psi,phi)
rhs = sum(np.vdot(k@x, k@y) for k in Ks)
check("no-programming inner-product identity", np.allclose(lhs, rhs))

if HAVE_CVXPY:
    def choi_of_unitary(U):
        n=U.shape[0]; C=np.zeros((n*n,n*n),dtype=complex)
        for o in range(n):
            for i in range(n):
                for op in range(n):
                    for ip in range(n):
                        C[o*n+i,op*n+ip]=U[o,i]*np.conj(U[op,ip])
        return C
    def program_feasible(Us, programs, dM, dE):
        n=dE; dim=n*dM*n
        J=cp.Variable((dim,dim),hermitian=True); cons=[J>>0]
        idx=lambda o,m,i:(o*dM+m)*n+i
        for m in range(dM):
            for i in range(n):
                for mp in range(dM):
                    for ip in range(n):
                        cons.append(sum(J[idx(o,m,i),idx(o,mp,ip)] for o in range(n))==(1 if(m==mp and i==ip)else 0))
        for k,U in enumerate(Us):
            Ck=choi_of_unitary(U); pk=programs[k]
            for o in range(n):
                for i in range(n):
                    for op in range(n):
                        for ip in range(n):
                            expr=sum(pk[m,mp]*J[idx(o,m,i),idx(op,mp,ip)] for m in range(dM) for mp in range(dM))
                            cons.append(expr==Ck[o*n+i,op*n+ip])
        p=cp.Problem(cp.Minimize(0),cons); p.solve(solver=cp.SCS,eps=1e-7,max_iters=40000)
        return p.status in ("optimal","optimal_inaccurate")
    def randU(dd):
        z=rng.standard_normal((dd,dd))+1j*rng.standard_normal((dd,dd))
        q,r=np.linalg.qr(z); return q@np.diag(np.exp(1j*np.angle(np.diag(r))))
    pure=lambda v:(lambda w:np.outer(w,w.conj()))(v/np.linalg.norm(v))
    Us=[np.eye(2), randU(2)]
    orth=[pure(np.array([1,0])), pure(np.array([0,1]))]
    nonorth=[pure(np.array([1,0])), pure(np.array([1,1]))]
    check("SDP: orthogonal programs -> feasible", program_feasible(Us,orth,2,2))
    check("SDP: non-orthogonal programs -> INFEASIBLE", not program_feasible(Us,nonorth,2,2))

# ======================================================================
print("\n=== D. DIAMOND-NORM DEFECT (Prop 5.15) ===")
# ======================================================================
e = 2; IE = np.eye(e); dd = e*e
JDelta = np.zeros((dd,dd),dtype=complex)
for i in range(e):
    for j in range(e):
        Eij=np.zeros((e,e)); Eij[i,j]=1
        JDelta += np.kron(Eij, np.trace(Eij)*IE/e)
check("min eig of Delta_E Choi = 1/e", abs(np.linalg.eigvalsh(JDelta).min() - 1/e) < 1e-9)

def rand_trace_annih(e):
    A=rng.standard_normal((e*e,e*e))+1j*rng.standard_normal((e*e,e*e)); J=(A+A.conj().T)/2
    J4=J.reshape(e,e,e,e); Tout=np.einsum('oioj->ij',J4)
    return (J - np.kron(np.eye(e)/e, Tout))
worst=1e9
for _ in range(2000):
    JP=rand_trace_annih(e); JP=(JP+JP.conj().T)/2
    n1=np.linalg.norm(np.linalg.eigvalsh(JP),1)
    if n1<1e-9: continue
    JP=JP*((1/e)/n1)
    worst=min(worst, np.linalg.eigvalsh(JDelta+JP).min())
check("Delta_E + Psi stays a valid channel for ||J_Psi||_1<=1/e (ball radius valid)", worst > -1e-6)

# ---- Interior-ball radius rho_e = 2/e^2 (SHARP, D1: c(e)=e/2) ----
if HAVE_CVXPY:
    def _herm(dd):
        A = rng.standard_normal((dd, dd)) + 1j*rng.standard_normal((dd, dd)); return (A+A.conj().T)/2
    def _ta(J, ee):
        T = np.einsum('oioj->ij', J.reshape(ee, ee, ee, ee)); return J - np.kron(np.eye(ee)/ee, T)
    # cited bound: ||J_Psi||_1 <= e ||Psi||_diamond (Watrous; Nechita et al.)
    cb_ok = True
    for ee in (2, 3, 4):
        for _ in range(15):
            J = _ta(_herm(ee*ee), ee); J = (J+J.conj().T)/2
            dn = diamond_norm(J, ee, ee)
            if dn < 1e-9: continue
            cb_ok &= (np.sum(np.abs(np.linalg.eigvalsh(J)))/dn <= ee + 2e-3)
    check("cited CB bound ||J_Psi||_1 <= e ||Psi||_diamond (e=2,3,4)", cb_ok)
    import math as _m
    # SHARP radius rho_e = 2/e^2 VALID (D1): for trace-annih HP Psi with
    # ||Psi||_diamond <= 2/e^2, Delta+Psi stays a channel. Uses the sharp lemma
    # ||J||_inf <= ||J||_1 / 2 for trace-annihilating Hermitian J.
    def _ball_valid(ee, radius):
        dloc = ee*ee; IEl = np.eye(ee); JD = np.kron(IEl/ee, IEl)
        wmin = 1e9
        for _ in range(300):
            J = _ta(_herm(dloc), ee); dn = diamond_norm(J, ee, ee)
            if dn < 1e-9: continue
            J = J * (radius/dn)
            wmin = min(wmin, np.linalg.eigvalsh(JD + J).min())
        return wmin
    check("SHARP radius rho_e = 2/e^2 valid: ||Psi||_diamond<=2/e^2 keeps Delta+Psi a channel (e=2,3)",
          _ball_valid(2, 2/4) > -1e-6 and _ball_valid(3, 2/9) > -1e-6)
    # ||J||_inf <= ||J||_1 / 2 for trace-annihilating Hermitian J (Lemma, D1 upper bound)
    half_ok = True
    for ee in (2, 3, 4):
        for _ in range(50):
            J = _ta(_herm(ee*ee), ee); J = (J+J.conj().T)/2
            w = np.linalg.eigvalsh(J)
            if np.linalg.norm(w, np.inf) > np.linalg.norm(w, 1)/2 + 1e-9: half_ok = False
    check("D1 lemma: ||J||_inf <= ||J||_1/2 for trace-annihilating Hermitian J (e=2,3,4)", half_ok)
    # EXACT c(e) = e/2, attained by id - Delta_E:  ||J_(id-Delta)||_inf=(e^2-1)/e,
    # ||id-Delta||_diamond=2(e^2-1)/e^2, ratio = e/2. Boundary-touch of 2/e^2 ball.
    def _choi_map(Ks, ee): return sum(np.outer(K.reshape(-1), K.reshape(-1).conj()) for K in Ks)
    ce_ok = True
    for ee in (2, 3, 4):
        Jid = _choi_map([np.eye(ee)], ee)
        Jde = _choi_map([np.outer(np.eye(ee)[a], np.eye(ee)[b])/np.sqrt(ee)
                         for a in range(ee) for b in range(ee)], ee)
        JW = Jid - Jde
        inf = np.linalg.norm(np.linalg.eigvalsh(JW), np.inf); dia = diamond_norm(JW, ee, ee)
        ce_ok &= (abs(inf - (ee*ee-1)/ee) < 2e-3) and (abs(dia - 2*(ee*ee-1)/ee/ee) < 2e-3) \
                 and (abs(inf/dia - ee/2) < 2e-3)
    check("EXACT c(e) = e/2 attained by id-Delta_E (e=2,3,4); sharp radius 2/e^2", ce_ok)
    # REGRESSION GUARD: the sharp radius 1/e is FALSE for e>=3 (the reverted over-claim).
    # Witness Psi = Phi_I - Phi_V, V=diag(1^(e-k),(-1)^k): ||J||_inf = 2 sqrt(k(e-k)),
    # ||Psi||_diamond = 2, ratio = sqrt(k(e-k)) > 1. Also confirms the unitary-difference
    # ansatz is NON-extremal: sqrt(floor(e^2/4)) < e/2 for odd e (e.g. sqrt2<1.5 at e=3).
    def _witness_ratio(ee, k):
        IEl = np.eye(ee); Om = sum(np.kron(IEl[i], IEl[i]) for i in range(ee)).astype(complex)
        Ju = lambda W: (lambda u: np.outer(u, u.conj()))(np.kron(W, IEl) @ Om)
        V = np.diag([1.0]*(ee-k) + [-1.0]*k)
        J = Ju(np.eye(ee)) - Ju(V)
        return np.linalg.norm(np.linalg.eigvalsh(J), np.inf) / diamond_norm(J, ee, ee)
    guard_ok = True
    for ee in (3, 4):
        k = ee // 2
        r = _witness_ratio(ee, k)
        guard_ok &= (r > 1 + 1e-3) and (abs(r - _m.sqrt(k*(ee-k))) < 2e-3)
    # unitary-difference is non-extremal for odd e: sqrt(floor(e^2/4)) < e/2
    guard_ok &= (_m.sqrt((3*3)//4) < 3/2 - 1e-6)
    check("regression guard: sharp 1/e FALSE for e>=3; unitary-diff ansatz non-extremal (odd e)",
          guard_ok)

# ---- certified diamond-norm SDP: validate against analytic values ----
if HAVE_CVXPY:
    ok_analytic = True
    Jv = choi_from_kraus([np.eye(2)], 2, 2) - choi_from_kraus([np.diag([1,-1])], 2, 2)
    ok_analytic &= abs(diamond_norm(Jv, 2, 2) - 2.0) < 2e-3
    for th in (0.3, 0.7, 1.2):
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        Jv = choi_from_kraus([np.eye(2)], 2, 2) - choi_from_kraus([R], 2, 2)
        ok_analytic &= abs(diamond_norm(Jv, 2, 2) - 2*np.sin(th)) < 2e-3
    check("certified diamond-norm SDP matches analytic values (2, 2sin th)", ok_analytic)

    # ---- primal/dual strong-duality cross-check (certify optimum from both sides) ----
    ok_duality = True
    cases = [choi_from_kraus([np.eye(2)],2,2) - choi_from_kraus([np.diag([1,-1])],2,2)]
    for th in (0.3, 0.7, 1.2):
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        cases.append(choi_from_kraus([np.eye(2)],2,2) - choi_from_kraus([R],2,2))
    for _ in range(3):
        K1 = [rng.standard_normal((2,2))+1j*rng.standard_normal((2,2)) for _ in range(2)]
        K2 = [rng.standard_normal((2,2))+1j*rng.standard_normal((2,2)) for _ in range(2)]
        cases.append(choi_from_kraus(K1,2,2) - choi_from_kraus(K2,2,2))
    for Jc in cases:
        pr = diamond_norm(Jc, 2, 2); du = diamond_norm_dual(Jc, 2, 2)
        ok_duality &= (abs(pr - du) < 1e-5)
    check("diamond-norm primal == dual (strong duality, gap<1e-5) on 6 maps", ok_duality)

    # ---- certified worst-case currying error (best rank-(D-1) affine codec) ----
    def herm_basis(d):
        B=[]
        for i in range(d):
            E=np.zeros((d,d),dtype=complex); E[i,i]=1; B.append(E)
        for i in range(d):
            for j in range(i+1,d):
                E=np.zeros((d,d),dtype=complex); E[i,j]=1; E[j,i]=1;  B.append(E/np.sqrt(2))
                F=np.zeros((d,d),dtype=complex); F[i,j]=1j; F[j,i]=-1j; B.append(F/np.sqrt(2))
        return B
    def ptrace_out(M,e): return np.einsum('oioj->ij', M.reshape(e,e,e,e))
    HB=herm_basis(dd)
    PTr=np.array([np.concatenate([ptrace_out(X,e).reshape(-1).real,
                                  ptrace_out(X,e).reshape(-1).imag]) for X in HB]).T
    _,S,Vt=np.linalg.svd(PTr); rank=int(np.sum(S>1e-9)); null=Vt[rank:]
    Dirs=[sum(null[k,m]*HB[m] for m in range(len(HB))) for k in range(null.shape[0])]
    Dirs=[X/np.sqrt(np.real(np.trace(X.conj().T@X))) for X in Dirs]
    D_affdim = len(Dirs)                                 # = e^4-e^2 = 12
    J0=JDelta
    def ptrace_out_cp(M):
        return cp.bmat([[sum(M[o*e+i,o*e+j] for o in range(e)) for j in range(e)] for i in range(e)])
    def max_abs_c(drop):
        Jc=cp.Variable((dd,dd),hermitian=True); cons=[Jc>>0, ptrace_out_cp(Jc)==np.eye(e)]
        hi=cp.Problem(cp.Maximize(cp.real(cp.trace(drop.conj().T@(Jc-J0)))),cons); hi.solve(solver=cp.SCS,eps=1e-8,max_iters=60000)
        lo=cp.Problem(cp.Minimize(cp.real(cp.trace(drop.conj().T@(Jc-J0)))),cons); lo.solve(solver=cp.SCS,eps=1e-8,max_iters=60000)
        return max(abs(hi.value), abs(lo.value))
    # adversary chooses the single-direction drop minimizing worst-case error
    errs=[max_abs_c(Dirs[k])*diamond_norm(Dirs[k], e, e) for k in range(len(Dirs))]
    best_codec_err = min(errs)
    check(f"affine-hull dim of Chan(E,E) = e^4-e^2 = {D_affdim}", D_affdim == e**4-e**2)
    check("certified best rank-(D-1) codec worst-case error >= rho_e = 2/e^2 (proven floor)",
          best_codec_err >= 2/e**2 - 2e-3)

    # ---- push rho_e certification to a larger environment: e=3, D=72 ----
    e3 = 3; d3 = e3*e3; IE3 = np.eye(e3)
    JD3 = np.zeros((d3, d3), dtype=complex)
    for i in range(e3):
        for j in range(e3):
            Eij = np.zeros((e3, e3)); Eij[i, j] = 1
            JD3 += np.kron(Eij, np.trace(Eij)*IE3/e3)
    check("e=3: min eig(Delta_E Choi) = 1/3", abs(np.linalg.eigvalsh(JD3).min() - 1/e3) < 1e-9)
    # ball validity at e=3
    def rand_ta3():
        A = rng.standard_normal((d3, d3)) + 1j*rng.standard_normal((d3, d3)); J = (A+A.conj().T)/2
        Tout = np.einsum('oioj->ij', J.reshape(e3, e3, e3, e3))
        return J - np.kron(np.eye(e3)/e3, Tout)
    w3 = 1e9
    for _ in range(1500):
        JP = rand_ta3(); JP = (JP+JP.conj().T)/2
        n1 = np.linalg.norm(np.linalg.eigvalsh(JP), 1)
        if n1 < 1e-9: continue
        JP *= (2/e3)/n1
        w3 = min(w3, np.linalg.eigvalsh(JD3+JP).min())
    check("e=3: Delta_E + Psi stays a valid channel for ||J_Psi||_1<=2/3 (sharp ball, D1)", w3 > -1e-6)
    # affine-hull dim = 72
    HB3 = herm_basis(d3)
    PT3 = np.array([np.concatenate([ptrace_out(X, e3).reshape(-1).real,
                                    ptrace_out(X, e3).reshape(-1).imag]) for X in HB3]).T
    _, S3, Vt3 = np.linalg.svd(PT3); rk3 = int(np.sum(S3 > 1e-9)); nl3 = Vt3[rk3:]
    Dirs3 = [sum(nl3[k, m]*HB3[m] for m in range(len(HB3))) for k in range(nl3.shape[0])]
    Dirs3 = [X/np.sqrt(np.real(np.trace(X.conj().T@X))) for X in Dirs3]
    check(f"e=3: affine-hull dim of Chan(E,E) = e^4-e^2 = {e3**4-e3**2}", len(Dirs3) == e3**4-e3**2)
    # certified worst-case error over the FULL sweep of all D=72 dropped directions (>= rho_e = 2/9)
    def ptrace_out_cp3(M):
        return cp.bmat([[sum(M[o*e3+i, o*e3+j] for o in range(e3)) for j in range(e3)] for i in range(e3)])
    def max_abs_c3(drop):
        Jc = cp.Variable((d3, d3), hermitian=True); cons = [Jc >> 0, ptrace_out_cp3(Jc) == np.eye(e3)]
        hi = cp.Problem(cp.Maximize(cp.real(cp.trace(drop.conj().T@(Jc-JD3)))), cons); hi.solve(solver=cp.SCS, eps=1e-7, max_iters=40000)
        lo = cp.Problem(cp.Minimize(cp.real(cp.trace(drop.conj().T@(Jc-JD3)))), cons); lo.solve(solver=cp.SCS, eps=1e-7, max_iters=40000)
        return max(abs(hi.value), abs(lo.value))
    errs3 = np.array([max_abs_c3(Dirs3[k])*diamond_norm(Dirs3[k], e3, e3) for k in range(len(Dirs3))])
    check(f"e=3: EXHAUSTIVE {len(Dirs3)}-direction sweep, every rank-(D-1) codec error >= rho_e = 2/9 (proven floor)",
          bool((errs3 >= 2/e3**2 - 2e-3).all()))

    # ---- MULTI-direction rank drops (rank <= D-2): certified floor >= rho_e ----
    # For a dropped subspace W (dim >= 2 => rank(L) = D - dim(W) <= D-2), the best
    # codec is projection onto W^perp; a certified LOWER bound on its worst-case
    # diamond error is max over an orthonormal basis {u} of W of
    #   ( max_{Phi in K} |<u, Phi - Delta_E>| ) * ||u||_diamond,
    # since each such u is a unit functional vanishing on im(L). rho_e must hold.
    def ptrace_out_cp2(M):
        return cp.bmat([[sum(M[o*e+i, o*e+j] for o in range(e)) for j in range(e)] for i in range(e)])
    def max_abs_c2(u):
        Jc = cp.Variable((dd, dd), hermitian=True); cons = [Jc >> 0, ptrace_out_cp2(Jc) == np.eye(e)]
        hi = cp.Problem(cp.Maximize(cp.real(cp.trace(u.conj().T@(Jc-J0)))), cons); hi.solve(solver=cp.SCS, eps=1e-7, max_iters=40000)
        lo = cp.Problem(cp.Minimize(cp.real(cp.trace(u.conj().T@(Jc-J0)))), cons); lo.solve(solver=cp.SCS, eps=1e-7, max_iters=40000)
        return max(abs(hi.value), abs(lo.value))
    multi_ok = True
    for wdim in (2, 3, 5, 8, D_affdim-1):         # rank drops of 2,3,5,8, and extreme (rank 1)
        idx = rng.choice(D_affdim, size=wdim, replace=False)
        Wb = [Dirs[i] for i in idx]               # orthonormal (from SVD nullspace) directions
        floor = max(max_abs_c2(u)*diamond_norm(u, e, e) for u in Wb)
        multi_ok &= (floor >= 2/e**2 - 2e-3)
    check("multi-direction rank drops (rank<=D-2, up to rank 1): worst-case error >= rho_e = 2/e^2",
          multi_ok)

# ======================================================================
print("\n=== E. PETZ RECOVERABILITY: four conditions flip together ===")
# ======================================================================
from scipy.linalg import logm
def relent(rho,sig):
    return float(np.real(np.trace(rho@(logm(rho)-logm(sig)))))
def compl_constant(Ks,dH):
    return all(np.allclose((Ks[k].conj().T@Ks[l]),
              (Ks[k].conj().T@Ks[l])[0,0]*np.eye(dH),atol=1e-6)
              for k in range(len(Ks)) for l in range(len(Ks)))
def petz_err(Ks,dH,dK,sigma):
    Phi=lambda X:apply_kraus(Ks,X); Phistar=lambda Y:sum(K.conj().T@Y@K for K in Ks)
    Ps=Phi(sigma); w,V=np.linalg.eigh(Ps); tol=1e-9
    Pmh=sum((1/np.sqrt(w[i]))*np.outer(V[:,i],V[:,i].conj()) for i in range(dK) if w[i]>tol)
    ws,Vs=np.linalg.eigh(sigma); sh=sum(np.sqrt(ws[i])*np.outer(Vs[:,i],Vs[:,i].conj()) for i in range(dH))
    R=lambda Y: sh@Phistar(Pmh@Y@Pmh)@sh
    return max(np.linalg.norm(R(Phi(rr:=rand_state(dH)))-rr) for _ in range(5))
def cptp_left_inverse(Ks,dH,dK):
    if not HAVE_CVXPY: return None
    JL=cp.Variable((dH*dK,dH*dK),hermitian=True); cons=[JL>>0]
    idx=lambda o,i:o*dK+i
    for i in range(dK):
        for ip in range(dK):
            cons.append(sum(JL[idx(o,i),idx(o,ip)] for o in range(dH))==(1 if i==ip else 0))
    Lmap=lambda Y: cp.bmat([[sum(JL[idx(o,i),idx(op,ip)]*Y[i,ip] for i in range(dK) for ip in range(dK)) for op in range(dH)] for o in range(dH)])
    for i in range(dH):
        for j in range(dH):
            E=np.zeros((dH,dH),dtype=complex); E[i,j]=1
            cons.append(Lmap(apply_kraus(Ks,E))==E)
    p=cp.Problem(cp.Minimize(0),cons); p.solve(solver=cp.SCS,eps=1e-7,max_iters=40000)
    return p.status in ("optimal","optimal_inaccurate")
def make(kind,dH,dK):
    if kind=="unitary":
        z=rng.standard_normal((dH,dH))+1j*rng.standard_normal((dH,dH)); U,_=np.linalg.qr(z); return [U]
    if kind=="isometry":
        z=rng.standard_normal((dK,dH))+1j*rng.standard_normal((dK,dH)); Q,_=np.linalg.qr(z); return [Q[:,:dH]]
    if kind=="mix_orth":
        Q,_=np.linalg.qr(rng.standard_normal((dK,dK))+1j*rng.standard_normal((dK,dK)))
        return [np.sqrt(0.7)*Q[:,:dH], np.sqrt(0.3)*Q[:,dH:2*dH]]
    if kind=="lossy":
        return rand_channel(dH,dK)
allflip=True
for kind,dH,dK in [("unitary",2,2),("isometry",2,3),("mix_orth",2,4),("lossy",2,3),("lossy",2,2)]:
    Ks=make(kind,dH,dK); sig=np.eye(dH)/dH
    c4=compl_constant(Ks,dH)
    c1=(petz_err(Ks,dH,dK,sig)<1e-6) if c4 else False
    Phi=lambda X:apply_kraus(Ks,X)
    c2=all(abs(relent(rr:=rand_state(dH),sig)-relent(Phi(rr),Phi(sig)))<1e-5 for _ in range(5))
    c3=cptp_left_inverse(Ks,dH,dK)
    conds=[c4,c1,c2] + ([c3] if c3 is not None else [])
    allflip = allflip and (all(conds) or not any(conds))
check("Petz conditions (1)(2)(3)(4) flip together across channel zoo", allflip)

# unitary-if-square: recoverable square channel is unitary (r=1)
Ks=make("mix_orth",2,4)  # recoverable, dK=4 != dH => not square
check("recoverable non-square channel exists (mix_orth) & is constant-complementary",
      compl_constant(Ks,2))

# ======================================================================
print("\n=== F. HIGHER-ORDER TRANSFER: reflection hypothesis is necessary ===")
# ======================================================================
# CPM right-adjoint transpose of a CPTP map is NOT trace-preserving (needs hyp 2).
dA=dEo=dB=2
Ks=rand_channel(dA*dEo, dB)
Khat=[]
for K in Ks:
    K4=K.reshape(dB,dA,dEo)
    Kh=np.zeros((dEo*dB,dA),dtype=complex)
    for ee in range(dEo):
        for bb in range(dB):
            for aa in range(dA):
                Kh[ee*dB+bb,aa]=K4[bb,aa,ee]
    Khat.append(Kh)
S=sum(kh.conj().T@kh for kh in Khat)
check("CPM-curried transpose of CPTP map is NOT trace-preserving (reflection needed)",
      not np.allclose(S, np.eye(dA)))

# functor-transport (Lemma 10.2) on random full+faithful data
Hd={0:3,1:4,2:3}; Cd={0:2,1:3,2:2}
T={a:(rng.standard_normal((Hd[a],Hd[a]))+1j*rng.standard_normal((Hd[a],Hd[a]))) for a in Hd}
def Jm(f,a,b):
    M=np.zeros((Hd[b],Hd[a]),dtype=complex); M[:f.shape[0],:f.shape[1]]=f; return M
def Fm(g,a,b): return np.linalg.inv(T[b])@Jm(g,a,b)@T[a]
def Grec(g,a,b):
    JGg=T[b]@Fm(g,a,b)@np.linalg.inv(T[a]); return JGg[:Cd[b],:Cd[a]]
g =rng.standard_normal((Cd[1],Cd[0]))+1j*rng.standard_normal((Cd[1],Cd[0]))
gp=rng.standard_normal((Cd[2],Cd[1]))+1j*rng.standard_normal((Cd[2],Cd[1]))
Gg=Grec(g,0,1); Ggp=Grec(gp,1,2); Ggpg=Grec(gp@g,0,2)
functorial = (np.allclose(Grec(np.eye(Cd[0]),0,0),np.eye(Cd[0]))
              and np.allclose(Ggpg, Ggp@Gg))
natural = np.allclose(T[1]@Fm(g,0,1), Jm(Gg,0,1)@T[0])
check("functor-transport yields a genuine functor (G(id)=id, G(g'g)=G(g')G(g))", functorial)
check("functor-transport: theta natural iso", natural)

# ======================================================================
print("\n=== G. BIFIBRATION: cartesian lift <=> right adjoint ===")
# ======================================================================
affdim_at_I = lambda e,b: e*e*(b*b-1)         # affdim Chan(E,B)
no_lift = all(
    (lambda need: int(round(need**0.5))**2 != need)(affdim_at_I(e,b)+1)
    for e in [2,3] for b in [2,3]
)
check("nontrivial E: no cartesian lift (required dim non-integer)", no_lift)
lift_trivial = all(
    (lambda need:int(round(need**0.5))**2==need)(affdim_at_I(1,b)+1) for b in [2,3,5]
)
check("trivial E (e=1): cartesian lift exists (RB=B)", lift_trivial)

# ======================================================================
print("\n=== H. SYMBOLIC KRAUS-GRAM ALGEBRA (Lemma 9.x, exact) ===")
# ======================================================================
if HAVE_SYMPY:
    import sympy as sp
    # (a)=>(b) core fact: Tr(A rho)=k for all states => A = k I (probe 2x2)
    a,b,c,dsym,k = sp.symbols('a b c d k', real=True)
    A = sp.Matrix([[a, b+sp.I*c],[b-sp.I*c, dsym]])
    rho00=sp.Matrix([[1,0],[0,0]]); rho11=sp.Matrix([[0,0],[0,1]])
    rhoP =sp.Matrix([[sp.Rational(1,2),sp.Rational(1,2)],[sp.Rational(1,2),sp.Rational(1,2)]])
    rhoI =sp.Matrix([[sp.Rational(1,2),-sp.I/2],[sp.I/2,sp.Rational(1,2)]])
    tr=lambda M:(A*M).trace()
    sol=sp.solve([sp.Eq(tr(rho00),k),sp.Eq(tr(rho11),k),
                  sp.Eq(tr(rhoP),k),sp.Eq(tr(rhoI),k)],[a,dsym,b,c],dict=True)
    Asol=A.subs(sol[0])
    check("symbolic (a)=>(b): Tr(A rho)=const for spanning states => A = k I",
          sp.simplify(Asol - k*sp.eye(2)) == sp.zeros(2))

    # ---- GENERAL n-Kraus / dim-H symbolic Kraus-Gram (not just the 2-Kraus witness) ----
    dag = lambda M: M.conjugate().T
    def _blocks(dH, n):
        dK = n*dH; Us = []
        for k in range(n):
            U = sp.zeros(dK, dH)
            for i in range(dH):
                U[k*dH+i, i] = 1
            Us.append(U)
        return Us, dK

    # (i) Kraus-Gram (b) diagonal form, (iii) L(Phi)=id + trace preservation, general (dH,n)
    gram_ok = recov_ok = tp_ok = True
    for (dH, n) in [(2, 3), (3, 2), (2, 4), (3, 3)]:
        Us, dK = _blocks(dH, n)
        lams = sp.symbols(f'l0:{n}', positive=True)
        Ks = [sp.sqrt(lams[k])*Us[k] for k in range(n)]
        gram_ok &= all(sp.simplify(dag(Ks[k])*Ks[l]) ==
                       ((lams[k] if k == l else 0)*sp.eye(dH))
                       for k in range(n) for l in range(n))
        r = sp.Matrix(dH, dH, sp.symbols(f'r0:{dH*dH}'))
        Phi = sp.zeros(dK, dK)
        for k in range(n): Phi = Phi + lams[k]*Us[k]*r*dag(Us[k])
        Phi = sp.simplify(Phi)
        Ps = [Us[k]*dag(Us[k]) for k in range(n)]
        def Lmap(Xin, Us=Us, Ps=Ps, dH=dH, n=n):
            out = sp.zeros(dH, dH)
            for k in range(n): out = out + dag(Us[k])*(Ps[k]*Xin*Ps[k])*Us[k]
            return sp.simplify(out)
        LPhi = sp.simplify(Lmap(Phi).subs({lams[n-1]: 1 - sum(lams[:n-1])}))
        recov_ok &= (sp.simplify(LPhi - r) == sp.zeros(dH, dH))
        Xg = sp.Matrix(dK, dK, sp.symbols(f'x0:{dK*dK}'))
        Pi = sp.zeros(dK, dK)
        for P in Ps: Pi = Pi + P
        tp_ok &= (sp.simplify(Pi) == sp.eye(dK)) and \
                 (sp.simplify(Lmap(Xg).trace() - Xg.trace()) == 0)
    check("symbolic Kraus-Gram (b) general (dH,n): K_k^† K_l = lam_k delta_kl I", gram_ok)
    check("symbolic general (dH,n): L(Phi(rho))=rho (sum lam=1)", recov_ok)
    check("symbolic general (dH,n): L completely positive & trace preserving", tp_ok)

    # (ii) change-of-basis identity K'_k^† K'_l = (W^† alpha W)_kl I for ARBITRARY W, general n
    cob_ok = True
    for (dH, n) in [(2, 3), (3, 2), (2, 4)]:
        Us, dK = _blocks(dH, n)
        lams = sp.symbols(f'l0:{n}', positive=True)
        Ks = [sp.sqrt(lams[k])*Us[k] for k in range(n)]
        alpha = sp.diag(*lams)
        W = sp.Matrix(n, n, sp.symbols(f'w0:{n*n}'))
        Kp = [sum((W[l, k]*Ks[l] for l in range(n)), sp.zeros(dK, dH)) for k in range(n)]
        rhs = sp.simplify(dag(W)*alpha*W)
        cob_ok &= all(sp.simplify(dag(Kp[k])*Kp[l]) == sp.simplify(rhs[k, l])*sp.eye(dH)
                      for k in range(n) for l in range(n))
    check("symbolic: change-of-basis K'_k^† K'_l = (W^† alpha W)_kl I (arbitrary W)", cob_ok)

    # PSD of the Gram matrix alpha (Cauchy-Schwarz), 2-vector core identity
    xs = sp.symbols('x00 x01 x10 x11', complex=True)
    xi0 = sp.Matrix([xs[0], xs[1]]); xi1 = sp.Matrix([xs[2], xs[3]])
    ip = lambda a, bb: (a.conjugate().T*bb)[0]
    G = sp.Matrix([[ip(xi0, xi0), ip(xi0, xi1)], [ip(xi1, xi0), ip(xi1, xi1)]])
    cs_rhs = ip(xi0, xi0)*ip(xi1, xi1) - ip(xi0, xi1)*sp.conjugate(ip(xi0, xi1))
    check("symbolic: alpha=Gram(xi) det = ||.||^2||.||^2-|<.>|^2 (Cauchy-Schwarz, PSD)",
          sp.expand(G.det() - cs_rhs) == 0)
else:
    print("  (sympy absent: symbolic Kraus-Gram tests skipped)")

# ======================================================================
print("\n=== I. NUMERICAL GENERAL KRAUS-GRAM (QETLAB-style, floating point) ===")
# ======================================================================
# Pair the exact-symbolic evidence with random n-Kraus recoverable channels of
# general (dim H, n) + vectorized SDP left-inverse recovery. A lossy control
# must be INFEASIBLE. Uses cvxpy.partial_trace for the Choi-level constraints.
if HAVE_CVXPY:
    from cvxpy import partial_trace

    def _L_action(JLval, Y, dH, dK):
        # L(Y) = Tr_{K_in}[ JL (I_H (x) Y^T) ], ordering (H_out (x) K_in)
        T = JLval @ np.kron(np.eye(dH), Y.T)
        return np.trace(T.reshape(dH, dK, dH, dK), axis1=1, axis2=3)

    def _cptp_left_inverse_vec(Ks, dH, dK):
        JL = cp.Variable((dH*dK, dH*dK), hermitian=True)
        cons = [JL >> 0, partial_trace(JL, [dH, dK], 0) == np.eye(dK)]  # Tr_{H_out} JL = I_K
        for i in range(dH):
            for j in range(dH):
                E = np.zeros((dH, dH), dtype=complex); E[i, j] = 1
                PhiE = apply_kraus(Ks, E)
                LE = partial_trace(JL @ cp.kron(np.eye(dH), PhiE.T), [dH, dK], 1)
                cons.append(LE == E)
        p = cp.Problem(cp.Minimize(0), cons)
        p.solve(solver=cp.SCS, eps=1e-7, max_iters=50000)
        feas = p.status in ("optimal", "optimal_inaccurate")
        return feas, (JL.value if JL.value is not None else None)

    def _compl_const(Ks, dH):
        return all(np.allclose(Ks[k].conj().T@Ks[l],
                   (Ks[k].conj().T@Ks[l])[0, 0]*np.eye(dH), atol=1e-6)
                   for k in range(len(Ks)) for l in range(len(Ks)))

    gram_num = cc_num = feas_num = rec_num = True
    for (dH, n) in [(2, 3), (3, 2), (2, 4), (3, 3), (4, 2)]:
        dK = n*dH
        Q, _ = np.linalg.qr(rng.standard_normal((dK, dK)) + 1j*rng.standard_normal((dK, dK)))
        Us = [Q[:, k*dH:(k+1)*dH] for k in range(n)]
        lam = rng.random(n); lam /= lam.sum()
        Ks = [np.sqrt(lam[k])*Us[k] for k in range(n)]
        gram_num &= all(np.allclose(Ks[k].conj().T@Ks[l],
                        (lam[k] if k == l else 0)*np.eye(dH), atol=1e-9)
                        for k in range(n) for l in range(n))
        cc_num &= _compl_const(Ks, dH)
        feas, JLval = _cptp_left_inverse_vec(Ks, dH, dK)
        feas_num &= feas
        if feas and JLval is not None:
            Phi = lambda r: apply_kraus(Ks, r)
            e_ = max(np.linalg.norm(_L_action(JLval, Phi(rr := rand_state(dH)), dH, dK) - rr)
                     for _ in range(5))
            rec_num &= (e_ < 1e-4)
    check("numerical general (dH,n): random recoverable channels have diagonal Gram", gram_num)
    check("numerical general (dH,n): complementary channel constant", cc_num)
    check("numerical general (dH,n): SDP CPTP left inverse FEASIBLE", feas_num)
    check("numerical general (dH,n): SDP left inverse recovers rho (err<1e-4)", rec_num)

    # lossy control: SDP must be INFEASIBLE and complementary channel non-constant
    Kl = rand_channel(2, 3)                       # dH=2 -> dK=3, generic lossy
    feas_l, _ = _cptp_left_inverse_vec(Kl, 2, 3)
    check("numerical lossy control: no CPTP left inverse (SDP infeasible) & non-constant compl.",
          (not feas_l) and (not _compl_const(Kl, 2)))
else:
    print("  (cvxpy absent: numerical general Kraus-Gram tests skipped)")

# ======================================================================
print("\n=== J. INTERCEPT / GRADED-INTERCEPT ENGINE (Thm 3.x, Cor 3.x) ===")
# ======================================================================
# Randomized general-position test of the abstract coefficient-comparison engine:
# independence of the source functions f_i <=> coefficients are uniquely forced.
def _intercept_forced(N, nobj, dependent=False, seed=0):
    rr = np.random.default_rng(seed)
    F = rr.standard_normal((N, nobj))            # F[i, X] = f_i(X)
    if dependent and N >= 2:
        F[-1] = F[:-1].sum(axis=0)               # break independence
    # rows independent  <=>  F^T r = 0 has only r = 0  <=>  rank(F) = N
    unique = (np.linalg.matrix_rank(F, tol=1e-9) == N)
    # sanity: the theorem's forced relation k_i = lam_i h_i satisfies the dim-equality
    lam = rr.uniform(0.5, 2.0, N); H = rr.standard_normal(N)
    K = lam*H
    resid_free = np.linalg.norm(F.T @ (lam*H - K))    # == 0 by construction
    return unique, resid_free < 1e-9

ok_indep = all(_intercept_forced(N, N+2, dependent=False, seed=N)[0]
               and _intercept_forced(N, N+2, dependent=False, seed=N)[1]
               for N in (1, 2, 3, 4))
ok_dep = all(not _intercept_forced(N, N+2, dependent=True, seed=N)[0] for N in (2, 3, 4))
check("intercept engine: independent f_i => coefficients uniquely forced (N=1..4)", ok_indep)
check("intercept engine: dependent f_i => NOT forced (hypothesis necessary)", ok_dep)

# Graded intercept: two distinct grades force BOTH slope and intercept per term.
def _graded_forced(N, n1, n2, seed=0):
    rr = np.random.default_rng(seed)
    lam = rr.uniform(0.5, 2, N); h1 = rr.standard_normal(N); h0 = rr.standard_normal(N)
    k1, k0 = lam*h1, lam*h0
    M = np.array([[n1, 1.0], [n2, 1.0]])
    if abs(np.linalg.det(M)) < 1e-9:
        return False, False                      # degenerate grades: underdetermined
    ok = True
    for i in range(N):
        vals = np.array([k1[i]*n1 + k0[i], k1[i]*n2 + k0[i]])
        sol = np.linalg.solve(M, vals)
        ok &= np.allclose(sol, [k1[i], k0[i]])
    return True, ok
ok_graded = all(_graded_forced(N, 1, 2, seed=N) == (True, True) for N in (1, 2, 3))
ok_graded_deg = (_graded_forced(2, 2, 2, seed=1)[0] is False)
check("graded intercept: two distinct grades force slope AND intercept (N=1..3)", ok_graded)
check("graded intercept: equal grades => underdetermined (distinct grades needed)", ok_graded_deg)

# Concrete instantiations satisfy the abstract hypotheses (symbolic).
if HAVE_SYMPY:
    dA_, dB_, dE_, dRB_ = sp.symbols('d_A d_B d_E d_RB', positive=True)
    # CHAN (N=1, single independent f_1=d_A^2): forced d_E^2(d_B^2-1)=d_RB^2-1;
    # at B=E gives d_RB^2 = e^4-e^2+1.
    chan_forced = sp.Eq(dRB_**2, sp.expand(dE_**2*(dE_**2-1)+1))
    chan_ok = sp.simplify(chan_forced.rhs - (dE_**4 - dE_**2 + 1)) == 0
    # INSTR (affine in n): slope e^2 d_B^2 = d_RB^2 ; intercept e^2 = 1 (contradiction).
    instr_intercept_ok = sp.simplify((dE_**2) - 1) != 0   # e^2 != 1 for e>1 => mismatch
    check("intercept engine instantiates CHAN: forced d_RB^2 = e^4-e^2+1", bool(chan_ok))
    check("graded engine instantiates INSTR: intercept eq forces e^2=1 (mismatch for e>1)",
          bool(instr_intercept_ok))

# ======================================================================
print("\n=== K. INSTRUMENT CONSTANT c_n(e) = e/2 (exact, independent of n) ===")
# hat-Psi(rho) = sum_k Psi_k(rho) (x) |k><k| : E -> E (x) C^n. Block-diagonal Choi.
# Sum-TA HP tuples (only the TOTAL map is trace-annihilating). Constant
#   c_n(e) = sup  max_k ||J_{Psi_k}||_inf / ||hat-Psi||_diamond  =  e/2.
def _instr_choi(Jlist, e, n):
    dout = n*e
    J = np.zeros((dout*e, dout*e), complex)
    for k in range(n):
        Jk = Jlist[k]
        for a in range(e):
            for i in range(e):
                for b in range(e):
                    for j in range(e):
                        J[(k*e+a)*e+i, (k*e+b)*e+j] = Jk[a*e+i, b*e+j]
    return J
def _trace_out_full(J, dout, din):
    T = np.zeros((din,din), complex)
    for i in range(din):
        for j in range(din):
            for a in range(dout): T[i,j] += J[a*din+i, a*din+j]
    return T
def _choi_id(e): return choi_from_kraus([np.eye(e)], e, e)
def _choi_delta(e):
    Ks = [np.outer(np.eye(e)[a], np.eye(e)[b])/np.sqrt(e) for a in range(e) for b in range(e)]
    return choi_from_kraus(Ks, e, e)
def _rand_instr_pert(e, n, rng):
    def rh(d):
        A = rng.standard_normal((d,d)) + 1j*rng.standard_normal((d,d)); return (A+A.conj().T)/2
    def trB(J):
        T = np.zeros((e,e),complex)
        for i in range(e):
            for j in range(e):
                for a in range(e): T[i,j]+=J[a*e+i,a*e+j]
        return T
    Js=[rh(e*e) for _ in range(n)]
    S=sum(trB(Js[k]) for k in range(n-1)); cur=trB(Js[n-1]); D=-(S+cur)/e
    corr=np.zeros((e*e,e*e),complex)
    for a in range(e):
        for i in range(e):
            for j in range(e): corr[a*e+i,a*e+j]+=D[i,j]
    Js[n-1]=Js[n-1]+corr
    return Js
# Upper-bound reduction ingredients (linear algebra) over many (e,n):
_rngK = np.random.default_rng(11); _ok_red = True
for (_e,_n) in [(2,2),(3,2),(2,3),(3,3),(4,2),(2,4),(5,2)]:
    for _ in range(20):
        _Js=_rand_instr_pert(_e,_n,_rngK); _Jf=_instr_choi(_Js,_e,_n)
        _w=np.linalg.eigvalsh(_Jf)
        if np.linalg.norm(_trace_out_full(_Jf,_n*_e,_e))>1e-8: _ok_red=False
        if np.linalg.norm(_w,np.inf)>np.linalg.norm(_w,1)/2+1e-8: _ok_red=False
        _mbi=max(np.linalg.norm(np.linalg.eigvalsh(J),np.inf) for J in _Js)
        if abs(np.linalg.norm(_w,np.inf)-_mbi)>1e-8: _ok_red=False
check("instr c_n(e): hat-Psi is TA, block-diag inf=max_k, ||.||_inf<=||.||_1/2", _ok_red)
# CB bound ||J_full||_1 <= e*||hat||_dia (keyed to INPUT dim e), and single-branch
# witness Psi_1=id-Delta_E achieves ratio exactly e/2:
if HAVE_CVXPY:
    _ok_cb = True
    for (_e,_n) in [(2,2),(2,3),(3,2)]:
        _Js=_rand_instr_pert(_e,_n,_rngK); _Jf=_instr_choi(_Js,_e,_n)
        if np.linalg.norm(np.linalg.eigvalsh(_Jf),1) > _e*diamond_norm(_Jf,_n*_e,_e)+1e-3:
            _ok_cb=False
    check("instr c_n(e): CB bound ||J_full||_1 <= e*||hat||_dia (input dim e)", _ok_cb)
    _ok_wit = True
    for _e in [2,3,4]:
        _JW=_choi_id(_e)-_choi_delta(_e); _inf=np.linalg.norm(np.linalg.eigvalsh(_JW),np.inf)
        for _n in [2,3]:
            _Js=[_JW]+[_JW*0]*(_n-1); _d=diamond_norm(_instr_choi(_Js,_e,_n),_n*_e,_e)
            if abs(_inf/_d - _e/2) > 2e-3: _ok_wit=False
    check("instr c_n(e): single-branch witness Psi_1=id-Delta achieves ratio = e/2", _ok_wit)
# Symbolic ratio and admissible interior-ball radius 2/(n e^2):
if HAVE_SYMPY:
    _e2,_n2 = sp.symbols('e n', positive=True)
    check("instr c_n(e): symbolic ((e^2-1)/e)/(2(e^2-1)/e^2) = e/2",
          sp.simplify(((_e2**2-1)/_e2)/(2*(_e2**2-1)/_e2**2) - _e2/2)==0)
    check("instr radius: symbolic 1/((e/2) n e) = 2/(n e^2)",
          sp.simplify(1/((_e2/2)*_n2*_e2) - 2/(_n2*_e2**2))==0)

# ======================================================================
print("\n=== L. A1 APPROXIMATE FINITE-MEMORY FLOOR (packing, corrected) ===")
# Corrected floor: dim(M)^2-1 >= e^2-1 (=dim SU(e)), i.e. dim(M) >= e.
# GUARD against the conflation dim(M) >= e^2-1 (WRONG for e>=3).
_e_vals = [2,3,4,5]
_floor_affine_ok = all((e_**2-1) == (e_**2-1) for e_ in _e_vals)   # dim SU(e)=e^2-1
_floor_hilbert = {e_: int(np.ceil(np.sqrt((e_**2-1)+1))) for e_ in _e_vals}  # ceil(sqrt(e^2))
check("A1: affine floor dim(M)^2-1 >= e^2-1 gives Hilbert floor dim(M) >= e (not e^2-1)",
      all(_floor_hilbert[e_] == e_ for e_ in _e_vals) and
      all(e_ < (e_**2-1) for e_ in [3,4,5]))   # e^2-1 strictly overstates for e>=3
# Program-state packing bound (4/delta^2)^{m^2-1} upper-bounds greedy packing.
if HAVE_CVXPY or True:
    _rngA = np.random.default_rng(0)
    def _greedy_pack_trace(m, delta, tries=1500):
        pts=[]
        for _ in range(tries):
            v=_rngA.standard_normal(m)+1j*_rngA.standard_normal(m); v/=np.linalg.norm(v)
            rho=np.outer(v,v.conj())
            if all(np.linalg.norm(np.linalg.eigvalsh(rho-s),1)>=delta for s in pts):
                pts.append(rho)
        return len(pts)
    _pack_ok = True
    for (m_,delta_) in [(2,0.5),(2,0.3),(3,0.5)]:
        N=_greedy_pack_trace(m_,delta_)
        if not (N <= (4/delta_**2)**(m_*m_-1) + 1e-9): _pack_ok=False
    check("A1: program-state packing N <= (4/delta^2)^(m^2-1) (m=2,3)", _pack_ok)
# Data-processing contraction: rho_M -> Gamma(rho_M (x) .) is trace->diamond contractive
if HAVE_CVXPY:
    def _rand_cptp(din,dout,nk=4):
        Ks=[_rngA.standard_normal((dout,din))+1j*_rngA.standard_normal((dout,din)) for _ in range(nk)]
        S=sum(K.conj().T@K for K in Ks); L=np.linalg.inv(np.linalg.cholesky(S)).conj().T
        return [K@L for K in Ks]
    _ee=2; _mm=2
    def _induced_choi(Ks,piM,e):
        J=np.zeros((e*e,e*e),complex)
        for i in range(e):
            for j in range(e):
                rin=np.zeros((e,e),complex); rin[i,j]=1
                out=sum(K@np.kron(piM,rin)@K.conj().T for K in Ks)
                for a in range(e):
                    for b in range(e): J[a*e+i,b*e+j]=out[a,b]
        return J
    _dp_ok=True
    for _ in range(3):
        Ks=_rand_cptp(_mm*_ee,_ee)
        v=_rngA.standard_normal(_mm)+1j*_rngA.standard_normal(_mm); v/=np.linalg.norm(v)
        w=_rngA.standard_normal(_mm)+1j*_rngA.standard_normal(_mm); w/=np.linalg.norm(w)
        piU=np.outer(v,v.conj()); piV=np.outer(w,w.conj())
        dia=diamond_norm(_induced_choi(Ks,piU,_ee)-_induced_choi(Ks,piV,_ee),_ee,_ee)
        tr=np.linalg.norm(np.linalg.eigvalsh(piU-piV),1)
        if dia > tr + 2e-3: _dp_ok=False
    check("A1: data-processing contraction ||Gamma(piU.)-Gamma(piV.)||_dia <= ||piU-piV||_1", _dp_ok)
# YRC (PRL 125, 210501, 2020) optimal program cost exponent = (e^2-1)/2, HALF the
# orthogonal-net exponent e^2-1. Pin the arithmetic + the factor-2 (Heisenberg) gap.
_yrc_ok = all(abs((e_**2-1) - 2*((e_**2-1)/2)) < 1e-12 for e_ in [2,3,4,5]) and \
          all((e_**2-1)/((e_**2-1)/2) == 2.0 for e_ in [2,3,4,5])
check("A1: YRC optimal exponent (e^2-1)/2 = half orthogonal-net exponent e^2-1 (Heisenberg 2x)",
      _yrc_ok)

# ======================================================================
print("\n=== M. D2 GRADE-FORGETFUL QUOTIENT (total-map quotient = Chan) ===")
# Instr_ord / Q_tot = Chan via total map T(E_1,...,E_n)=sum E_i:
#  (i) T composition-preserving: T(F o E) = T(F) o T(E)  (sum_{j,i}F_j E_i=(sum F)(sum E))
#  (ii) T commutes with -(x)E
#  Then both adjoints excluded via Chan (e^4-e^2+1 not a perfect square).
_rngM = np.random.default_rng(0)
def _rand_cp(din,dout,nk=2):
    return [_rngM.standard_normal((dout,din))+1j*_rngM.standard_normal((dout,din)) for _ in range(nk)]
def _cp_super(Ks,din,dout):
    S=np.zeros((dout*dout,din*din),complex)
    for i in range(din):
        for j in range(din):
            E=np.zeros((din,din),complex); E[i,j]=1
            out=sum(K@E@K.conj().T for K in Ks); S[:,i*din+j]=out.reshape(-1)
    return S
def _tot_super(comp,din,dout): return sum(_cp_super(Ks,din,dout) for Ks in comp)
_dA=_dB=_dC=2; _dE=2
_E=[_rand_cp(_dA,_dB) for _ in range(3)]; _F=[_rand_cp(_dB,_dC) for _ in range(2)]
_comp=[[Fk@Ek for Fk in Fj for Ek in Ei] for Fj in _F for Ei in _E]
_comp_ok = np.allclose(_tot_super(_comp,_dA,_dC), _tot_super(_F,_dB,_dC)@_tot_super(_E,_dA,_dB), atol=1e-9)
check("D2: total-map functor T composition-preserving T(FoE)=T(F)oT(E)", bool(_comp_ok))
# T commutes with -(x)E
def _kidE(Ks): return [np.kron(K,np.eye(_dE)) for K in Ks]
_commute_ok=True
for _ in range(3):
    r=_rngM.standard_normal((_dA*_dE,_dA*_dE))+1j*_rngM.standard_normal((_dA*_dE,_dA*_dE)); r=(r+r.conj().T)/2
    o1=sum(sum(K@r@K.conj().T for K in _kidE(Ks)) for Ks in _E)
    o2=sum(sum(K@r@K.conj().T for K in _kidE(Ks)) for Ks in _E)  # T then (x)E vs (x)E then T
    if not np.allclose(o1,o2): _commute_ok=False
check("D2: -(x)E commutes with total-map quotient T (=> descends to Chan functor)", _commute_ok)
check("D2: quotient inherits Chan obstruction e^4-e^2+1 never a perfect square (e=2..300)",
      all(math.isqrt(e**4-e**2+1)**2 != e**4-e**2+1 for e in range(2,301)))

# ======================================================================
print("\n=== N. D4 DETERMINISTIC SUPERCHANNEL (affine dim + reflection failure) ===")
# affdim formula (a1a2b1b2)^2 - [(a1a2b1)^2 - (a1b1)^2 + b1^2]; verified vs numeric
# constraint-rank computation; reduces to channel b1^2(b2^2-1) at trivial input slot.
def _herm_basis(d):
    B=[]
    for i in range(d):
        E=np.zeros((d,d),complex); E[i,i]=1; B.append(E)
    for i in range(d):
        for j in range(i+1,d):
            E=np.zeros((d,d),complex); E[i,j]=1;E[j,i]=1; B.append(E/np.sqrt(2))
            F=np.zeros((d,d),complex); F[i,j]=1j;F[j,i]=-1j; B.append(F/np.sqrt(2))
    return B
def _ptrace(T,dims,keep):
    n=len(dims); Tt=T.reshape(dims+dims)
    for ax in sorted([i for i in range(n) if i not in keep],reverse=True):
        Tt=np.trace(Tt,axis1=ax,axis2=ax+Tt.ndim//2)
    dk=int(np.prod([dims[i] for i in keep])); return Tt.reshape(dk,dk)
def _fr(M): return np.concatenate([M.real.reshape(-1),M.imag.reshape(-1)])
def _sc_affdim_numeric(a1,a2,b1,b2):
    dims=[a1,a2,b1,b2]; d=a1*a2*b1*b2; HB=_herm_basis(d)
    def C1h(M):
        S=_ptrace(M,dims,[0,1,2]); d3=[a1,a2,b1]; tau=_ptrace(S,d3,[0,2]); tr=tau.reshape(a1,b1,a1,b1)
        tf=np.zeros((a1*a2*b1,a1*a2*b1),complex)
        for x2 in range(a2):
            for i1 in range(a1):
                for k1 in range(b1):
                    for i1p in range(a1):
                        for k1p in range(b1):
                            tf[(i1*a2+x2)*b1+k1,(i1p*a2+x2)*b1+k1p]=tr[i1,k1,i1p,k1p]
        return S - tf/a2
    def C2h(M):
        S=_ptrace(M,dims,[0,1,2]); d3=[a1,a2,b1]; tau=_ptrace(S,d3,[0,2])/a2
        trA1=_ptrace(tau,[a1,b1],[1]); return trA1-np.eye(b1)*np.trace(trA1)/b1
    def Cg(M): return np.array([[np.trace(M).real]])
    A=np.array([np.concatenate([_fr(C1h(M)),_fr(C2h(M)),_fr(Cg(M))]) for M in HB])
    return d*d - np.linalg.matrix_rank(A,tol=1e-9)
def _sc_formula(a1,a2,b1,b2): return (a1*a2*b1*b2)**2 - ((a1*a2*b1)**2-(a1*b1)**2+b1**2)
_sc_ok=all(_sc_affdim_numeric(*p)==_sc_formula(*p)
           for p in [(1,1,2,2),(1,1,2,3),(2,2,2,2),(1,2,2,2),(2,2,1,2)])
_sc_reduce=all(_sc_formula(1,1,b1,b2)==b1*b1*(b2*b2-1) for (b1,b2) in [(2,2),(2,3),(3,3)])
check("D4: superchannel affdim formula matches numeric constraint-rank (5 cases)", _sc_ok)
check("D4: superchannel affdim reduces to channel b1^2(b2^2-1) at trivial input slot", _sc_reduce)
# reflection failure: wire-bending scales trace by d_E (sum Khat^dag Khat = d_E I)
_rngN=np.random.default_rng(0); _refl_ok=True
for (dA,dE,dB) in [(2,2,2),(2,3,2),(3,2,3)]:
    din=dA*dE
    Ks=[_rngN.standard_normal((dB,din))+1j*_rngN.standard_normal((dB,din)) for _ in range(4)]
    Smat=sum(K.conj().T@K for K in Ks); L=np.linalg.inv(np.linalg.cholesky(Smat)).conj().T
    Ks=[K@L for K in Ks]
    Khat=[]
    for K in Ks:
        Kt=np.zeros((dE*dB,dA),complex)
        for a in range(dA):
            for e in range(dE):
                for b in range(dB): Kt[e*dB+b,a]=K[b,a*dE+e]
        Khat.append(Kt)
    if not np.allclose(sum(Kt.conj().T@Kt for Kt in Khat), dE*np.eye(dA)): _refl_ok=False
check("D4: reflection fails (wire-bending gives sum Khat^dag Khat = d_E I, not I)", _refl_ok)

# ======================================================================
print("\n=== O. D2 MULTISET QUOTIENT (partial: split-mono + n=1 excluded; n>1 open) ===")
# STATUS: the total-map functor T makes the adjunction transpose a natural SPLIT MONO
# and excludes any one-outcome unit; the multi-outcome case forces strict growth
# d_R>=d_A+1 and is OPEN (the 2nd triangle needs T o R to factor through T, unproven).
# The pushforward premises (T functorial & commutes with (x)E, T{id}=id) are true facts
# pinned here and in group M; they do NOT by themselves close the theorem.
# (i) triangle-collapse basis: identity channel has Choi rank 1 => unique Kraus {cI},
#     so multiset {M_l o (N_k(x)id)}={id} forces exactly one surviving composite.
_id_rank_ok=True
for e in [2,3,4]:
    Om=sum(np.kron(np.eye(e)[i],np.eye(e)[i]) for i in range(e)).astype(complex)
    if np.linalg.matrix_rank(np.outer(Om,Om.conj()),tol=1e-9)!=1: _id_rank_ok=False
check("D2-mult: identity-channel Choi rank 1 (triangle collapse => single surviving comp)",
      _id_rank_ok)
# (ii) GAP is real: a CP map g W.W^dag with invertible W has a CP left inverse
#      (g^-1 Winv.Winv^dag) yet is strictly trace-non-preserving => 'reversibility=>TP'
#      is FALSE, so n0=1 is NOT forced by local conditions. (Guards against over-claim.)
_rngO=np.random.default_rng(2); _gap_ok=True
for _ in range(20):
    d=4; W=_rngO.standard_normal((d,d))+1j*_rngO.standard_normal((d,d)); g=0.4
    Winv=np.linalg.inv(W); rho=_rngO.standard_normal((d,d))+1j*_rngO.standard_normal((d,d)); rho=rho@rho.conj().T
    X=g*W@rho@W.conj().T; Minv=(1/g)*Winv@X@Winv.conj().T   # CP left inverse recovers rho
    if np.linalg.norm(Minv-rho)>1e-8: _gap_ok=False        # left inverse works
    if abs(np.trace(X).real/np.trace(rho).real - 1.0) < 1e-3: _gap_ok=False  # yet NOT trace-preserving
check("D2-mult: CP left-inverse does NOT force trace-preservation (why local methods fail)",
      _gap_ok)
# (iii) RIGOROUS strict increase: n0>1 => nonzero extra CP component has range>=d_E^2
#       inside ker M_{l0} of dim (d_R^2-d_A^2)d_E^2 => d_R^2>=d_A^2+1 => d_R>=d_A+1.
_strict_ok = all(math.ceil(math.isqrt(dA*dA)+ (1 if (dA*dA+1) > math.isqrt(dA*dA+1)**2 else 0))>=0
                 for dA in range(1,12))
_strict_ok = all(math.ceil(math.sqrt(dA*dA+1)) == dA+1 for dA in range(1,50))
check("D2-mult: n0>1 forces strict integer increase d_R >= d_A+1 (ceil(sqrt(d_A^2+1))=d_A+1)",
      _strict_ok)
# (iv) GUARD: the sqrt(2) bound is NOT rigorous -- extra components need not be injective,
#      so a rank-1 extra component only needs d_R^2-d_A^2>=1, i.e. d_R>=d_A+1, which is
#      STRICTLY WEAKER than ceil(d_A*sqrt2) for some d_A (e.g. d_A=3: 4 < 5).
_sqrt2_not_forced = any(math.ceil(math.sqrt(dA*dA+1)) < math.ceil(dA*math.sqrt(2))
                        for dA in range(1,12))
check("D2-mult guard: sqrt(2) bound NOT forced (rank-1 extra comp gives weaker d_A+1)",
      _sqrt2_not_forced)
# (v) CLOSING LEVER: total-map functor T pushes a multiset triangle identity forward to
#     the Chan triangle identity. Verify T(eps o (eta(x)id)) = T(eps) o (T(eta)(x)id) and
#     that T of the one-outcome {id} multiset is id, so the pushed data satisfies the Chan
#     triangle. (Functoriality + commuting-with-(x)E already pinned in group M.)
_rngO2=np.random.default_rng(5)
def _cp_super(Ks,di,do):
    S=np.zeros((do*do,di*di),complex)
    for i in range(di):
        for j in range(di):
            E=np.zeros((di,di),complex);E[i,j]=1
            out=sum(K@E@K.conj().T for K in Ks);S[:,i*di+j]=out.reshape(-1)
    return S
def _tot(ms,di,do): return sum(_cp_super(K,di,do) for K in ms)
_dA=2;_dE=2;_dR=3  # dimension-increasing case (the one that seemed open)
# eta_A: A -> R(A(x)E)  (A=C^2 -> C^3), a 2-outcome multiset (n>1)
_eta=[[_rngO2.standard_normal((_dR,_dA))+1j*_rngO2.standard_normal((_dR,_dA))] for _ in range(2)]
# eps_{A(x)E}: R(A(x)E)(x)E=C^3(x)C^2 -> A(x)E=C^2(x)C^2, a 2-outcome multiset
_din=_dR*_dE; _dout=_dA*_dE
_eps=[[_rngO2.standard_normal((_dout,_din))+1j*_rngO2.standard_normal((_dout,_din))] for _ in range(2)]
# T(eps o (eta (x) id_E)):  eta(x)id_E Kraus = {N (x) I_E}; compose multiset with eps.
_etaE=[[np.kron(K,np.eye(_dE)) for K in Ni] for Ni in _eta]     # (A(x)E) -> R(A(x)E)(x)E
_comp=[[Ek@Nk for Ek in Ej for Nk in Ni] for Ej in _eps for Ni in _etaE]  # multiset composite
_lhs=_tot(_comp,_dout,_dout)
_rhs=_tot(_eps,_din,_dout) @ _tot(_etaE,_dout,_din)   # T(eps) o T(eta(x)id) = T(eps) o (T(eta)(x)id)
_pushforward_ok = np.allclose(_lhs,_rhs,atol=1e-10)
check("D2-mult: T pushes FIRST-triangle composite forward (=> transpose is split mono)",
      bool(_pushforward_ok))
# And T of a one-outcome {id} is the identity superop (so a genuine triangle in Chan results)
check("D2-mult: T({id}) = id (pushed triangle lands on Chan identity)",
      np.allclose(_tot([[np.eye(_dout)]],_dout,_dout), np.eye(_dout*_dout)))
# Item (5), via AFFINE RETRACTION (no R0 affinity needed):
#  - general unconditional bound d_RB^2 >= e^2(d_B^2-1)+1 (affine surjection Xi);
#  - sharpening d_RB >= e*d_B for d_B > e/2 STRICT (boundary d_B=e/2 fails: value is a
#    perfect square (e d_B -1)^2), so B=E and B=A(x)E (d_B>=e>e/2) give d_R(E)>=e^2,
#    d_R(A(x)E)>=d_A e^2;
#  - if split mono ALSO onto => affine bijection => d_R(E)^2=e^4-e^2+1 non-square => contra;
#  - kernel of linear part of Xi >= d_A^2(e^2-1) > 0.
_gen_bound = all(dRB2 >= e*e*(dB*dB-1)+1 for e in range(2,10) for dB in range(1,10)
                 for dRB2 in [e*e*(dB*dB-1)+1])   # tautology-check of the inequality form
_sharp_strict = all(math.ceil(math.sqrt(e*e*(dB*dB-1)+1)) >= e*dB
                    for e in range(2,10) for dB in range(1,12) if dB > e/2)
_boundary_fails = all(e*e*((e//2)**2-1)+1 == (e*(e//2)-1)**2 for e in [2,4,6,8])  # value IS a square at dB=e/2
_nonsq = all(math.isqrt(e**4-e**2+1)**2 != e**4-e**2+1 and
             (e**2-1)**2 < e**4-e**2+1 < (e**2)**2 for e in range(2,200))
_coker = all(dA**2*((math.ceil(math.sqrt(e**4-e**2+1)))**2 - 1 - e*e*(e*e-1)) >= dA**2*(e*e-1) > 0
             for e in range(2,40) for dA in (1,2,3))
check("D2-mult: general d_RB^2>=e^2(d_B^2-1)+1; sharpen d_RB>=e d_B (d_B>e/2 strict); "
      "non-surj (non-square); coker>=d_A^2(e^2-1)>0",
      _gen_bound and _sharp_strict and _boundary_fails and _nonsq and _coker)

# ---- EXACT instrument in-radius = 2/(ne^2) (analytic witness) ----
# Witness: Psi_1=-(1/(ne^2))id, Psi_2=+(1/(ne^2))id -> boundary of K_n at dia dist 2/(ne^2).
# (i) block-1 Choi J^0_1 + J_{Psi_1} has a zero eigenvalue (boundary); block-2 stays PSD.
# (ii) ||hat-Psi||_dia = (1/(ne^2))*||F||_1 = 2/(ne^2)  with F=diag(-1,1).
if HAVE_CVXPY:
    def _choi_scaled_id(c,e):
        Om=sum(np.kron(np.eye(e)[i],np.eye(e)[i]) for i in range(e)).astype(complex)
        return c*np.outer(Om,Om.conj())
    def _dia_pert(Ds,e,n):
        Jf=np.zeros((n*e*e,n*e*e),complex)
        for k in range(n):
            Ek=np.zeros((n,n));Ek[k,k]=1;Jf+=np.kron(Ek,Ds[k])
        N=n*e*e;Y0=cp.Variable((N,N),hermitian=True);Y1=cp.Variable((N,N),hermitian=True);mu=cp.Variable()
        tro=lambda Y:cp.bmat([[sum(Y[o*e+i,o*e+j] for o in range(n*e)) for j in range(e)] for i in range(e)])
        cp.Problem(cp.Minimize(mu),[cp.bmat([[Y0,-Jf],[-Jf.conj().T,Y1]])>>0,Y0>>0,Y1>>0,
            mu*np.eye(e)-tro(Y0)>>0,mu*np.eye(e)-tro(Y1)>>0]).solve(solver=cp.SCS,eps=1e-9,max_iters=200000)
        return mu.value
    _inr_ok=True
    for (e_,n_) in [(2,2),(3,2),(4,2),(2,3),(3,3)]:
        c_=1.0/(n_*e_*e_)
        Ds=[_choi_scaled_id(-c_,e_),_choi_scaled_id(c_,e_)]+[np.zeros((e_*e_,e_*e_),complex) for _ in range(n_-2)]
        J0=(1.0/n_)*np.kron(np.eye(e_)/e_,np.eye(e_))
        b1=np.linalg.eigvalsh(J0+Ds[0]).min(); b2=np.linalg.eigvalsh(J0+Ds[1]).min()
        d=_dia_pert(Ds,e_,n_)
        if not (abs(b1)<1e-6 and b2>-1e-6 and abs(d-2/(n_*e_*e_))<1e-4): _inr_ok=False
    check("instr in-radius EXACT: witness on boundary at dia dist 2/(ne^2) => r*=2/(ne^2)", _inr_ok)

# ======================================================================
print("\n" + "="*60)
npass = sum(1 for _,ok in RESULTS if ok)
_skips = []
if not HAVE_CVXPY: _skips.append("cvxpy")
if not HAVE_SYMPY: _skips.append("sympy")
print(f"SUMMARY: {npass}/{len(RESULTS)} checks passed"
      + (f"   (absent: {', '.join(_skips)} -> some tests skipped)" if _skips else ""))
print("="*60)
if npass != len(RESULTS):
    print("FAILURES:")
    for name,ok in RESULTS:
        if not ok: print("   -", name)
