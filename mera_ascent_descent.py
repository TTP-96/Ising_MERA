"""
A pedagogical code implementing a MERA-like ansatz for the transverse-field Ising model (TFIM)

with both ascent (operator renormalization) and descent (state construction)

This code performs binary MERA-like optimization for the 1D TFIM using
a simple variational ansatz consisting of layers of two-qubit disentanglers
and isometries, topped by a single-site state vector.

We assume a translation-invariant, scale invariant ansatz, so only one
disentangler U and one isometry W are used at each layer, along with
a single top vector t.
The assumption of criticality/scale invariance is the reason why we
have only one U and W, reused at each layer. 

The U and W tensors implement an RG transformation that best
preserves the ground state of the TFIM Hamiltonian. The top tensor t
represents the ground state at the coarsest scale (a single site).

For now the code allows you to compute the energy either via descent (the full state route)
or via ascent (the operator renormalization route). This serves as a consistency check.

TODO:
- Implement calculation of scaling dimensions from the ascended operators, operator product expansion coefficients, central charge, etc.
- Implement more advanced optimizers (e.g., gradient-based).
- Implement calculation of correlation functions.


--- Usage example ---

python mera_ascent_descent.py --L 16 --J 1.0 --g 1.0 --steps 100 --tries 10 --route ascent
"""




import numpy as np
from numpy.linalg import norm
from scipy.linalg import polar, expm
from scipy.sparse.linalg import LinearOperator, eigsh

import argparse


rng = np.random.default_rng(10)

# Paulis
I = np.array([[1,0],[0,1]], dtype=complex)
X = np.array([[0,1],[1,0]], dtype=complex)
Z = np.array([[1,0],[0,-1]], dtype=complex)

# utility functions 
def kron_n(*ops):
    out = np.array([[1]], dtype=complex)
    for A in ops:
        out = np.kron(out, A)
    return out

def reshape_u(U):
    """4x4 -> u[a,b,ap,bp] with (ab)->row, (ap bp)->col."""
    return U.reshape(2,2,2,2)

def reshape_w(W):
    """4x2 -> w[a,b,s] with (ab)->row, s->col."""
    return W.reshape(2,2,2)

#  ASCENT (center channel) 
def ascend_1site_center(O, U, W, which="left"):
    """
    Ascend a 1-site operator O through a single MERA layer in the *center channel*:
      O' = w^\dagger u^\dagger (O ⊗ I  or  I ⊗ O) u w
    Args:
      O: (2,2)
      U: (4,4)  two-qubit disentangler
      W: (4,2)  isometry (2->1 site)
      which: "left" places O on the left of the 2-site block; "right" on the right
    Returns:
      O': (2,2)
    """
    u = reshape_u(U)          # (a,b,ap,bp)
    w = reshape_w(W)          # (a,b,s)
    # embed O on the 2-site block:
    OO = np.kron(O, I) if which == "left" else np.kron(I, O)   # (4,4)
    # Sandwich in pair space: u† (OO) u
    tmp = U.conj().T @ OO @ U                                  # (4,4)
    tmp4 = tmp.reshape(2,2,2,2)                                 # [a',b'; a,b]
    # Project 2->1 via w† ... w:
    # O'[s', s] = sum_{a',b',a,b} w^*[a',b',s'] tmp4[a',b',a,b] w[a,b,s]
    left = np.tensordot(w.conj(), tmp4, axes=([0,1],[0,1]))     # (s', a, b)
    Oprime = np.tensordot(left, w, axes=([1,2],[0,1]))          # (s', s)
    return Oprime

def ascend_2site_center(OO, U, W):
    """
    Ascend a *2-site* operator OO acting on the fused pair (center channel):
      OO' = w^\dagger u^\dagger OO u w
    (Here OO already acts on the 2-site block.)
    Args:
      OO: (4,4)
        U: (4,4)  two-qubit disentangler
        W: (4,2)  isometry (2->1 site)
    Returns:
      (2,2) operator on the coarse site.
    """
    # print sizes as sanity check
    # print("ascend_2site_center: OO.shape =", OO.shape)
    # print("U.shape =", U.shape)
    # print("W.shape =", W.shape)
    u = reshape_u(U) # change shape for tensordot. (a,b,ap,bp)
    w = reshape_w(W) # (a,b,s)
    tmp = U.conj().T @ OO @ U                                   # (4,4)
    tmp4 = tmp.reshape(2,2,2,2)
    left = np.tensordot(w.conj(), tmp4, axes=([0,1],[0,1]))     # (s', a, b)
    Oprime = np.tensordot(left, w, axes=([1,2],[0,1]))          # (s', s)
    return Oprime

def ascend_layers_1site(O, U, W, layers):
    """
    Push a 1-site operator to the top through 'layers' identical center-channel layers.
    To mimic translation invariance, average 'left' and 'right' placements at each layer
    """
    Op = O.copy()
    for _ in range(layers):
        O_L = ascend_1site_center(Op, U, W, which="left")
        O_R = ascend_1site_center(Op, U, W, which="right")
        Op = 0.5*(O_L + O_R)   #  TI average
    return Op

def ascend_layers_2site(OO, U, W, layers):
    """
    Push a 2-site operator OO up through `layers` layers.

    First layer: use 2-site -> 1-site ascender.
    Remaining layers: use 1-site ascender (avg left/right for TI).
    The reason is that after the first layer, the operator is 1-site because
    the isometry W maps 2 sites -> 1 site.
    """
    assert OO.shape == (4, 4), "OO must be a 2-site operator (4x4)."
    if layers == 0:
        return OO  # nothing to do

    # Layer 1: 2-site -> 1-site
    Op = ascend_2site_center(OO, U, W)     # now Op is (2,2)

    # Layers 2..layers: 1-site -> 1-site
    for _ in range(layers - 1):
        O_L = ascend_1site_center(Op, U, W, which="left")
        O_R = ascend_1site_center(Op, U, W, which="right")
        Op  = 0.5 * (O_L + O_R)

    return Op
# DESCENT  
def apply_twoqubit_gate_state(psi, U, i):
    """
    Apply a two-qubit gate U to sites (i, i+1) of the full state psi.
    psi: (2,2,2,...,2) rank-L state
    U: (4,4) two-qubit gate
    i: site index (first of the two)
    Returns: new state psi' with U applied at (i,i+1)
    """
    n = psi.ndim
    axes = (i, i+1) + tuple(k for k in range(n) if k not in (i, i+1)) # bring (i,i+1) to front
    inv = np.argsort(axes) # to restore order later
    phi = np.transpose(psi, axes=axes).reshape(2,2,-1)     # (2,2,R) where R=2^(L-2)
    U4 = U.reshape(2,2,2,2)                             # (a,b,ap,bp)
    out = np.tensordot(U4, phi, axes=([2,3],[0,1]))        # (2,2,R)
    out = out.reshape((2,2) + phi.shape[2:]) # (2,2,2,2,...,2)
    out = np.transpose(out, axes=inv) # restore original order
    return out

def apply_isometry_expand_state(psi, W, s):
    """
    Apply isometry W to site s of the full state psi to expand 1->2 sites.
    psi: (2,2,2,...,2) rank-L state
    W: (4,2) isometry
    s: site index to expand
    Returns: new state psi' with site s expanded to two sites
    """
    W2 = W.reshape(2,2,2)                                  # (a,b,s)
    axes = (s,) + tuple(k for k in range(psi.ndim) if k != s) # bring s to front
    inv = np.argsort(axes) # to restore order later
    phi = np.transpose(psi, axes=axes).reshape(2, -1)      # (s, R)
    new = np.tensordot(W2, phi, axes=([2],[0]))            # (a,b,R)
    new = new.reshape(2,2, *psi.shape[:s], *psi.shape[s+1:]) # (2,2,2,...,2) as *psi.shape[:s] unpacks the tuple of leading dims and *psi.shape[s+1:] unpacks the trailing dims

    # restore original order with new axes
    before = list(range(2, 2+s)) # axes before s
    after  = list(range(2+s, new.ndim)) # axes after s
    perm = before + [0,1] + after # new order: before, new sites, after (the 0 and 1 are the two new sites from W which are now at front but need to go to position s and s+1)
    new = np.transpose(new, axes=perm)
    return new

def descend_state(U, W, t, L):
    assert (L & (L-1)) == 0 and L >= 2
    layers = int(np.log2(L))
    psi = (t / norm(t)).reshape(2,)
    # Descend through layers and build full state
    for ell in range(layers):
        n = psi.ndim # number of sites before expansion
        for s in range(n):
            psi = apply_isometry_expand_state(psi, W, 2*s)     # expand 1->2 at each site
        Ut = U.conj().T
        n2 = psi.ndim # number of sites after expansion
        offset = 0 if (ell % 2 == 0) else 1 # staggered pattern of disentanglers. Need to offset by 1 on odd layers so that disentanglers always act on (even,odd) pairs 
        for i in range(offset, n2-1, 2):
            psi = apply_twoqubit_gate_state(psi, Ut, i)
        psi = psi / norm(psi.ravel()) # normalize to avoid numerical issues
    return psi

#  Local expectations (state route) 
def one_site_expectation(psi, O, i):
    """
    Compute <O_i> from full state psi.
    psi: (2,2,2,...,2) rank-L state
    O: (2,2) operator
    i: site index
    Returns: float <O_i>

    Construct the 1-site reduced density matrix at site i and evaluate Tr(rho O).
    Axes are rearranged so that site i is first, then reshaped.

    """
    axes = (i,) + tuple(k for k in range(psi.ndim) if k != i) # bring i to front
    phi = np.transpose(psi, axes=axes).reshape(2, -1)      # 2 x (rest) Now i is first axis 
    rho = phi @ phi.conj().T                               # 2x2 RDM. This does Tr_rest |psi><psi|
    return float(np.real(np.trace(rho @ O)))  # Tr(rho O) (expval of O acting on site i)

def two_site_expectation(psi, O2, i):
    """
    Compute <O_{i,i+1}> from full state psi.
    psi: (2,2,2,...,2) rank-L state
    O2: (4,4) operator acting on sites (i,i+1)
    i: site index (first of the two)
    Returns: float <O_{i,i+1}>

    Mode of operation is analogous to one_site_expectation above.
    Construct the 2-site reduced density matrix at sites (i,i+1) and evaluate Tr(rho O2).
    Axes are rearranged so that sites (i,i+1) are first, then reshaped.
    """
    
    axes = (i, i+1) + tuple(k for k in range(psi.ndim) if k not in (i, i+1))
    phi = np.transpose(psi, axes=axes).reshape(4, -1)
    rho2 = phi @ phi.conj().T                              # 4x4 RDM
    return float(np.real(np.trace(rho2 @ O2)))

def tfim_energy_from_state(psi, J=1.0, g=1.0, pbc=False):
    """
    Compute the TFIM energy from the full state psi:
      H = -J ∑ Z_i Z_{i+1} - g ∑ X_i
    Args:
        psi: (2,2,2,...,2) rank-L state
        J, g: TFIM parameters
        pbc: bool, periodic BC

    Returns:
        E: float, energy expectation <psi|H|psi>

    This approach is O(L * 2^L) in time and memory, so only
    feasible for small L (e.g., L <= 16).
    """

    
    L = psi.ndim
    ZZ = np.kron(Z, Z)
    E = 0.0
    for i in range(L-1):
        E += -J * two_site_expectation(psi, ZZ, i)
    if pbc and L > 2:
        psi_rot = np.rollaxis(psi, 0, psi.ndim) # roll site 0 to the end
        psi_rot = np.ascontiguousarray(psi_rot) # ensure memory layout to improve performance
        E += -J * two_site_expectation(psi_rot, ZZ, psi_rot.ndim-2) # last two sites
    for i in range(L):
        E += -g * one_site_expectation(psi, X, i)
    return E

# Projections
def project_unitary(M):
    """
    Project a general complex matrix M onto the nearest unitary matrix U.
    The polar decomposition gives you the closest unitary in Frobenius norm sense.
    """
    Uu, H = polar(M)  # polar decompose M. U is a unitary matrix whilst H is hermitian positive-semidefinite. This is like a matrix version of the complex number decomposition z = exp(i*theta)*r 
    det = np.linalg.det(Uu)
    if abs(det) > 1e-12:
        Uu = Uu / (det**(1/4)) # divide by det^(1/4) as for an nxn matrix det(cA)=c^n det(A)
    return Uu

def project_isometry(W):
    """
    Project a general complex matrix W onto the nearest isometry V (V†V=I).
    QR decomposition achieves this in the same Frobenius norm sense as above.
    """
    Q, R = np.linalg.qr(W)
    return Q[:, :W.shape[1]]

def random_su4_perturb(eps, rng):
    """
    Generate a small random perturbation in SU(4) via exponentiating a random
    anti-Hermitian matrix.
    """
    A = rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4)) # random complex matrix
    K = A - A.conj().T # make it anti-Hermitian
    return project_unitary(np.eye(4, dtype=complex) + eps * K / (norm(K)+1e-12)) # linearization of the exponential map (I + eps K) projected back to unitary

def random_isometry_perturb(W, eps, rng):
    """
    Generate a small random perturbation of an isometry W by adding
    a small random matrix and re-projecting.
    Note that unlike the above case of a unitary perturbation, isometries live on
    a Stiefel manifold, which does not have a simple Lie group structure. Hence, the
    exponential map approach is not natural here.
    """
    D = rng.normal(size=W.shape) + 1j*rng.normal(size=W.shape)
    D /= (norm(D) + 1e-12)
    return project_isometry(W + eps * D)

def random_top_rotation(t, eps, rng):
    """
    Generate a small random rotation of the top vector t by exponentiating
    a random anti-Hermitian 2x2 matrix. We use an exponential map here since
    the set of normalized vectors is isomorphic to the unitary group U(2).
    
    """
    A = rng.normal(size=(2,2)) + 1j*rng.normal(size=(2,2))
    K = A - A.conj().T
    U2 = expm(eps * K / (norm(K)+1e-12))
    v = U2 @ t
    return v / norm(v)

#  Energies: descent vs ascent 
def energy_via_descent(U, W, t, L, J=1.0, g=1.0, pbc=False):
    psi = descend_state(U, W, t, L)
    return tfim_energy_from_state(psi, J=J, g=g, pbc=pbc)

def energy_via_ascent(U, W, t, L, J=1.0, g=1.0, pbc=False):
    """
      1) Push a representative X (1-site) and ZZ (2-site) up log2(L) layers.
      2) Evaluate at the top with ρ_top = |t><t| and multiply by counts (L and L-1 or L).
    NOTE: Uses the *center channel* each layer (offset=0) and averages left/right
          placement for 1-site operators to mimic translation invariance.
    """
    layers = int(np.log2(L))

    # Push local operators to the top
    X_top  = ascend_layers_1site(X,       U, W, layers)         # (2,2)
    ZZ_top = ascend_layers_2site(np.kron(Z, Z), U, W, layers)    # (2,2)

    # Top density
    t = t / norm(t)
    rho_top = np.outer(t, t.conj())                              # (2,2)

    # Expectations at top
    exp_X  = np.real(np.trace(rho_top @ X_top))
    exp_ZZ = np.real(np.trace(rho_top @ ZZ_top))

    # Count of terms (open chain)
    nZZ = L-1 if not pbc else L
    nX  = L

    E = -J * nZZ * exp_ZZ - g * nX * exp_X
    return float(E)


def energy(U, W, t, L, J=1.0, g=1.0, pbc=False, route="descent"):
    if route == "descent":
        return energy_via_descent(U, W, t, L, J, g, pbc)
    elif route == "ascent":
        return energy_via_ascent(U, W, t, L, J, g, pbc)
    else:
        raise ValueError("route must be 'descent' or 'ascent'")

# Optimizer 
def optimize(U, W, t, L, J=1.0, g=1.0, pbc=False,
             steps=140, tries=15, epsU=0.12, epsW=0.12, epst=0.12,
             decay=0.985, seed=10, route="descent"):
    """
    Optimize the tensors U, W, t to minimize the TFIM energy via random
    perturbations and selection. We use a simple adaptive scheme to adjust
    the perturbation sizes. For each iteration we try `tries` random perturbations
    and keep the best one if it improves the energy. If no improvement is found
    in an iteration, we slightly increase the perturbation sizes. Otherwise, we
    induce a decay in the perturbation sizes (like a learning rate schedule).
    Args:
      U: (4,4) disentangler
      W: (4,2) isometry
      t: (2,)   top vector
      L: int    system size (power of 2)
      J, g: TFIM parameters
      pbc: bool, periodic BC
      steps: int, optimization steps
      tries: int, random tries per step
      epsU, epsW, epst: float, initial perturbation sizes
      decay: float, decay factor for perturbation sizes
      seed: int, random seed
      route: "descent" or "ascent" for energy evaluation

    Returns:
        Ub, Wb, tb: optimized tensors
        e_best: best energy found
    """
    rng = np.random.default_rng(seed)
    e_best = energy(U, W, t, L, J, g, pbc, route=route) # initial energy
    Ub, Wb, tb = U.copy(), W.copy(), t.copy() # best tensors
    print(f"init E={e_best:.6f}  (L={L}, route={route})")
    for it in range(steps):
        improved = False
        for _ in range(tries): # random tries
            U_try = project_unitary(Ub @ random_su4_perturb(epsU, rng))
            W_try = random_isometry_perturb(Wb, epsW, rng)
            t_try = random_top_rotation(tb, epst, rng)
            e = energy(U_try, W_try, t_try, L, J, g, pbc, route=route)
            if e < e_best: # improvement found
                e_best, Ub, Wb, tb = e, U_try, W_try, t_try
                improved = True
        # decay (or increase) perturbation sizes depending on improvement
        epsU *= decay
        epsW *= decay
        epst *= decay
        if (it+1) % 10 == 0:
            print(f"iter {it+1:3d}  E={e_best:.6f}")
        if not improved:
            epsU *= 1.05
            epsW *= 1.05
            epst *= 1.05
    return Ub, Wb, tb, e_best


def tfim_linear_operator(L, J=1.0, g=1.0, pbc=False, dtype=np.float64):
    """
    Returns a scipy.sparse.linalg.LinearOperator that applies the TFIM Hamiltonian
    H = -J ∑ Z_i Z_{i+1} - g ∑ X_i  to a vector v, without forming H.
    Uses a (2,)*L view for vectorized matvec (O(L * 2^L) time, O(2^L) memory).
    """
    N = 1 << L                      # 2**L
    z = np.array([1.0, -1.0], dtype=dtype)  # Z eigenvalues

    def matvec(v):
        v = np.asarray(v)
        # reshape into rank-L tensor
        phi = v.reshape((2,)*L)
        out = np.zeros_like(phi)

        
        # -J ∑ Z_i Z_{i+1} term (diagonal, via broadcasted signs for efficiency)
        for i in range(L-1):
            zi = z.reshape((1,)*i + (2,) + (1,)*(L-i-1)) # This creates a shape like (1,1,2,1,1,...) with 2 at position i
            zj = z.reshape((1,)*(i+1) + (2,) + (1,)*(L-i-2)) # This creates a shape like (1,1,1,2,1,...) with 2 at position i+1

            # elementwise multiply and accumulate. This works because zi and zj broadcast to the full shape of phi, giving the correct sign for each basis state. zi*zj  is the diagonal of Z_i Z_{i+1} in the computational basis. It's value is +1 if spins i and i+1 are aligned, -1 if anti-aligned. zi*zj 
            out += -J * (zi * zj) * phi  # elementwise multiply. Broadcast happens here (resulting shape is (2,2,...,2))
        if pbc and L > 2:

            # if periodic BC, add term connecting last and first sites
            zi = z.reshape((2,) + (1,)*(L-1))           # site 0
            zj = z.reshape((1,)*(L-1) + (2,))           # site L-1
            out += -J * (zi * zj) * phi

        # -g ∑ X_i term (bit-flip along axis i)


        for i in range(L):
            out += -g * np.flip(phi, axis=i) # flip along axis i flips the i-th qubit (applies X_i)

        return out.reshape(-1)

    # return LinearOperator instance
    # LinearOperator allows us to define a matrix-like object via its matvec function without explicitly forming the matrix which can be too large to store
    return LinearOperator((N, N), matvec=matvec, dtype=dtype) 


def ground_energy_lanczos(L, J=1.0, g=1.0, pbc=False, k=1, maxiter=None, tol=1e-10):
    """
    Compute the ground-state energy with ARPACK (eigsh) using only H·v.
    """
    Hlin = tfim_linear_operator(L, J=J, g=g, pbc=pbc, dtype=np.float64)
    # For Hermitian operators: which='SA' -> smallest algebraic
    vals, _ = eigsh(Hlin, k=k, which='SA', maxiter=maxiter, tol=tol)
    return float(vals.min())

#  Exact TFIM (for benchmarking) 
# def TFIM(L, J=1.0, g=1.0, pbc=False):
#     H = np.zeros((2**L, 2**L), dtype=complex)
#     # -J sum Z_i Z_{i+1}
#     for i in range(L-1):
#         term = [I]*L; term[i]=Z; term[i+1]=Z
#         H += -J * kron_n(*term)
#     if pbc and L>2:
#         term = [I]*L; term[-1]=Z; term[0]=Z
#         H += -J * kron_n(*term)
#     # -g sum X_i
#     for i in range(L):
#         term = [I]*L; term[i]=X
#         H += -g * kron_n(*term)
#     return H

# -------------------- Main --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MERA-like ansatz with ascent and descent")
    parser.add_argument("--L", type=int, default=16, help="System size (power of 2)")
    parser.add_argument("--J", type=float, default=1.0, help="Coupling J")
    parser.add_argument("--g", type=float, default=1.0, help="Transverse field g")
    parser.add_argument("--pbc", action="store_true", help="Use periodic BC")
    parser.add_argument("--steps", type=int, default=80, help="Optimization steps")
    parser.add_argument("--tries", type=int, default=12, help="Random tries per step")
    parser.add_argument("--epsU", type=float, default=0.12, help="Initial U perturbation size")
    parser.add_argument("--epsW", type=float, default=0.12, help="Initial W perturbation size")
    parser.add_argument("--epst", type=float, default=0.12, help="Initial t perturbation size")
    parser.add_argument("--route", choices=["descent","ascent"], default="ascent",
                        help="Compute energy via 'descent' (full state) or 'ascent' (coarse operators)")
    args = parser.parse_args()

    L     = args.L
    assert (L & (L-1)) == 0 and L >= 2, "L must be a power of 2 >= 2"
    J     = args.J
    g     = args.g
    pbc   = args.pbc
    steps = args.steps
    tries = args.tries
    epsU  = args.epsU
    epsW  = args.epsW
    epst  = args.epst
    route = args.route

    # initialize tensors
    U = project_unitary(rng.normal(size=(4,4)) + 1j*rng.normal(size=(4,4)))
    W = project_isometry(rng.normal(size=(4,2)) + 1j*rng.normal(size=(4,2)))
    t = rng.normal(size=(2,)) + 1j*rng.normal(size=(2,))
    t /= norm(t)

    U_opt, W_opt, t_opt, Ebest = optimize(U, W, t, L, J=J, g=g, pbc=pbc,
                                          steps=steps, tries=tries,
                                          epsU=epsU, epsW=epsW, epst=epst,
                                          route=route)

    print("\nOptimized energy (route={}): {:.6f}".format(route, Ebest))

    # Optional: benchmark exact GS energy for small L
    if L <= 18:
        try:
            E_lanczos = ground_energy_lanczos(L, J=J, g=g, pbc=pbc, k=1, maxiter=None, tol=1e-10)
            print("Lanczos ground energy:", E_lanczos)
        except Exception as e:
            print("Lanczos failed:", e)
        # H = TFIM(L, J=J, g=g, pbc=pbc)
        # E_exact = np.linalg.eigvalsh(H).min()
        # print("Exact ground energy:        {:.6f}".format(E_exact))
