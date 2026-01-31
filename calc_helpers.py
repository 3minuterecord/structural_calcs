import sys
import os
import pandas as pd
import ast
import math
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


def getPythonLoc():
    return Path(__file__).resolve().parent

def d():
    print('-'*75)
    
def errStatement(statement):
    print()
    d()
    print(f'-> ***ERROR***')
    print(f'-> ***{statement}***')
    print(f'-> ***PROGRAMME ABORTED***')
    d()
    print()
    sys.exit()

def checkPath(path):
    if path.exists() == False:
        errStatement(f'Cannot locate: "{path}"')

def checkFileExists(file_path):
    if os.path.exists(file_path) == False:
        errStatement(f'Cannot locate: "{file_path}"')

def check_and_load_life_data(pos, case, YRS, DFF, KDF):
    for step in ['step7', 'step6']:
        filename = f'{currentDir}\\LifetimeFatigueSummary\\{step}_lifetime_pos{pos}_{case}.xlsx'
        if os.path.exists(filename):
            df = kf.life_to_df(filename, YRS=YRS, DFF=DFF, KDF=KDF)
            return df
    raise FileNotFoundError(f"No valid lifetime file found for pos {pos} and case {case}")

def analyze_beam(spans, loads, supports, E, I,
                 nel_per_span=4, n_plot_per_span=200):
    """
    Beam analysis (Euler-Bernoulli, prismatic, UDL per span).

    Parameters
    ----------
    spans : list[float]
        Span lengths [L1, L2, ..., Ln] in m.
    loads : list[float]
        UDL on each span [w1, w2, ..., wn] in kN/m (positive downward).
    supports : list[str]
        Boundary conditions at original nodes (len = n+1).
        Each entry: 'pin', 'fixed', or 'free'.
        e.g. ['pin', 'pin', 'free'] -> pin-pin with overhang.
    E : float
        Young’s modulus (kN/m²).
    I : float
        Second moment of area (m⁴).
    nel_per_span : int, optional
        Number of FE elements per span (>=2 is usually fine).
    n_plot_per_span : int, optional
        Number of plotting/diagram points per span.

    Returns
    -------
    result : dict
        {
          "x": np.array,   # position along beam (m)
          "v": np.array,   # deflection (m, downward positive)
          "V": np.array,   # shear (kN, upward positive)
          "M": np.array,   # bending moment (kNm, sagging +ve)
          "support_positions": np.array,
          "support_reactions": np.array,  # vertical (kN, upward +ve)
        }
    """
    S = len(spans)
    assert len(loads) == S
    assert len(supports) == S + 1

    # --- refine spans into smaller FE elements ---
    elem_lengths = []
    elem_loads = []
    for Lspan, wspan in zip(spans, loads):
        Le = Lspan / nel_per_span
        for _ in range(nel_per_span):
            elem_lengths.append(Le)
            elem_loads.append(wspan)

    n_elem = len(elem_lengths)
    n_nodes = n_elem + 1
    ndofs = 2 * n_nodes
    EI = E * I

    K = np.zeros((ndofs, ndofs))
    F = np.zeros(ndofs)

    # --- assemble global K and equivalent nodal loads F ---
    for e, (L, w) in enumerate(zip(elem_lengths, elem_loads)):
        L2 = L * L
        L3 = L2 * L
        # 4x4 beam element stiffness matrix
        k = EI / L3 * np.array([
            [ 12,    6*L, -12,    6*L],
            [  6*L,  4*L2, -6*L,  2*L2],
            [-12,   -6*L,  12,   -6*L],
            [  6*L,  2*L2, -6*L,  4*L2],
        ])
        # Consistent nodal loads for UDL w (downward)
        fe = w * L / 2 * np.array([1, L/6, 1, -L/6])

        dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
        for i in range(4):
            F[dofs[i]] += fe[i]
            for j in range(4):
                K[dofs[i], dofs[j]] += k[i, j]

    # --- map original supports -> refined nodes ---
    supports_ref = ['free'] * n_nodes
    for j, bc in enumerate(supports):
        node_idx = j * nel_per_span
        supports_ref[node_idx] = bc

    fixed_dofs = []
    for n, bc in enumerate(supports_ref):
        if bc in ('pin', 'fixed'):
            fixed_dofs.append(2 * n)       # vertical displacement = 0
        if bc == 'fixed':
            fixed_dofs.append(2 * n + 1)   # rotation = 0

    free_dofs = [i for i in range(ndofs) if i not in fixed_dofs]

    # --- solve for nodal displacements ---
    Kff = K[np.ix_(free_dofs, free_dofs)]
    Ff = F[free_dofs]
    d = np.zeros(ndofs)
    d_free = np.linalg.solve(Kff, Ff)
    d[free_dofs] = d_free

    # --- reactions at all refined nodes ---
    R = K.dot(d) - F

    # --- original node positions (span boundaries) ---
    node_pos_orig = [0.0]
    cum = 0.0
    for Lspan in spans:
        cum += Lspan
        node_pos_orig.append(cum)

    # support reactions at original support nodes
    support_positions = []
    support_reactions = []
    for j, bc in enumerate(supports):
        node_idx = j * nel_per_span
        if bc in ('pin', 'fixed'):
            pos = node_pos_orig[j]
            Vj = -R[2 * node_idx]  # upward reaction positive
            support_positions.append(pos)
            support_reactions.append(Vj)

    support_positions = np.array(support_positions)
    support_reactions = np.array(support_reactions)

    # --- deflection field v(x) from FE (Hermite shape in each element) ---
    total_L = sum(spans)
    n_plot = S * n_plot_per_span + 1
    xs = np.linspace(0.0, total_L, n_plot)
    vs = np.zeros_like(xs)

    # element start positions along beam
    elem_starts = [0.0]
    c = 0.0
    for L in elem_lengths:
        c += L
        elem_starts.append(c)
    elem_starts = np.array(elem_starts)

    for i, x in enumerate(xs):
        # find element e where x lies
        e = max(0, min(n_elem-1,
                       np.searchsorted(elem_starts, x, side='right') - 1))
        x0 = elem_starts[e]
        L = elem_lengths[e]
        xi = (x - x0) / L

        dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
        d_e = d[dofs]

        # Hermite shape functions
        N1 = 1 - 3*xi**2 + 2*xi**3
        N2 = L * (xi - 2*xi**2 + xi**3)
        N3 = 3*xi**2 - 2*xi**3
        N4 = L * (-xi**2 + xi**3)

        vs[i] = N1*d_e[0] + N2*d_e[1] + N3*d_e[2] + N4*d_e[3]

    # --- shear V(x) and moment M(x) from statics ---
    Vs = np.zeros_like(xs)
    Ms = np.zeros_like(xs)

    # span boundaries (original grid)
    span_bounds = [0.0]
    c = 0.0
    for Lspan in spans:
        c += Lspan
        span_bounds.append(c)
    span_bounds = np.array(span_bounds)

    for i, x in enumerate(xs):
        # shear: sum of support reactions to the left minus UDL to the left
        mask_r = support_positions <= x + 1e-9
        V_reac = support_reactions[mask_r].sum()

        V_load = 0.0
        for si, Lspan in enumerate(spans):
            span_start = span_bounds[si]
            span_end = span_bounds[si+1]
            if x <= span_start:
                continue
            a = span_start
            b = min(span_end, x)
            if b > a:
                Lseg = b - a
                V_load += loads[si] * Lseg

        Vs[i] = V_reac - V_load  # upward positive

        # bending moment: Σ(Rj * (x - xj)) - ∫ w (x - s) ds
        M_reac = ((x - support_positions[mask_r]) *
                  support_reactions[mask_r]).sum()

        M_load = 0.0
        for si, Lspan in enumerate(spans):
            span_start = span_bounds[si]
            span_end = span_bounds[si+1]
            if x <= span_start:
                continue
            a = span_start
            b = min(span_end, x)
            if b > a:
                Lseg = b - a
                wspan = loads[si]
                # integral of w (x - s) ds from s=a to b
                M_load += wspan * (x*Lseg - (b**2 - a**2)/2.0)

        Ms[i] = M_reac - M_load  # sagging positive

    return {
        "x": xs,
        "v": vs,
        "V": Vs,
        "M": Ms,
        "support_positions": support_positions,
        "support_reactions": support_reactions,
    }

def tip_deflection_overhang(L1, L2, w, E, I):
    """
    Pin at 0, pin at L1, free overhang L2, UDL w over full length.
    Returns downward tip deflection in metres.
    """
    k = L2 * (L1**3 - 4*L1*L2**2 - 3*L2**3) / 24.0
    return k * w / (E * I)\


def plot_shear_moment(x, V, M):
    """
    Plot shear force and bending moment diagrams for a beam.

    Parameters
    ----------
    x : array-like
        Positions along the beam (m).
    V : array-like
        Shear force values (kN). Positive up.
    M : array-like
        Bending moment values (kNm). Positive sagging
        (will be flipped so hogging plots positive).
    """

    x = np.asarray(x)
    V = np.asarray(V)
    M = np.asarray(M)

    # -------------------------
    # Shear force diagram
    # -------------------------
    plt.figure(figsize=(8, 3))

    plt.plot(x, V, linewidth=1.5, color="black")
    plt.fill_between(
        x, 0, V,
        where=(V >= 0),
        interpolate=True,
        color="maroon",
        alpha=0.3,
    )
    plt.fill_between(
        x, 0, V,
        where=(V <= 0),
        interpolate=True,
        color="grey",
        alpha=0.3,
    )

    plt.axhline(0, linewidth=0.8)
    plt.xlabel("x (m)")
    plt.ylabel("Shear V (kN)")
    plt.title("Shear Force Diagram")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Bending moment diagram
    # -------------------------

    # Flip sign so hogging is positive on the plot
    M_plot = -M

    plt.figure(figsize=(8, 3))

    plt.plot(x, M_plot, linewidth=1.5, color="black")
    plt.fill_between(
        x, 0, M_plot,
        where=(M_plot >= 0),
        interpolate=True,
        color="maroon",
        alpha=0.3,
    )
    plt.fill_between(
        x, 0, M_plot,
        where=(M_plot <= 0),
        interpolate=True,
        color="grey",
        alpha=0.3,
    )

    plt.axhline(0, linewidth=0.8)
    plt.xlabel("x (m)")
    plt.ylabel("Moment M (kNm)  (+ve hogging)")
    plt.title("Bending Moment Diagram")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_shear_moment_deflection(x, V, M, d):
    """
    Plot shear force, bending moment, and deflected shape diagrams for a beam.

    Parameters
    ----------
    x : array-like
        Positions along the beam (m).
    V : array-like
        Shear force values (kN). Positive up.
    M : array-like
        Bending moment values (kNm). Positive sagging
        (will be flipped so hogging plots positive).
    d : array-like
        Vertical deflection values (mm). Positive upward.
    """

    x = np.asarray(x)
    V = np.asarray(V)
    M = np.asarray(M)
    d = np.asarray(d)

    # -------------------------
    # Shear force diagram
    # -------------------------
    plt.figure(figsize=(8, 3))

    plt.plot(x, V, linewidth=1.5, color="black")
    plt.fill_between(
        x, 0, V,
        where=(V >= 0),
        interpolate=True,
        color="maroon",
        alpha=0.3,
    )
    plt.fill_between(
        x, 0, V,
        where=(V <= 0),
        interpolate=True,
        color="grey",
        alpha=0.3,
    )

    plt.axhline(0, linewidth=0.8)
    plt.xlabel("x (m)")
    plt.ylabel("Shear V (kN)")
    plt.title("Shear Force Diagram")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Bending moment diagram
    # -------------------------

    # Flip sign so hogging is positive on the plot
    M_plot = -M

    plt.figure(figsize=(8, 3))

    plt.plot(x, M_plot, linewidth=1.5, color="black")
    plt.fill_between(
        x, 0, M_plot,
        where=(M_plot >= 0),
        interpolate=True,
        color="maroon",
        alpha=0.3,
    )
    plt.fill_between(
        x, 0, M_plot,
        where=(M_plot <= 0),
        interpolate=True,
        color="grey",
        alpha=0.3,
    )

    plt.axhline(0, linewidth=0.8)
    plt.xlabel("x (m)")
    plt.ylabel("Moment M (kNm)  (+ve hogging)")
    plt.title("Bending Moment Diagram")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Deflected shape
    # -------------------------

    # Find max positive and negative deflections
    i_max = np.argmax(d)
    i_min = np.argmin(d)

    x_max, d_max = x[i_max], round(d[i_max], 1)
    x_min, d_min = x[i_min], round(d[i_min], 1)

    fig, ax = plt.subplots(figsize=(9, 3.8))

    ax.plot(x, d, linewidth=1.5, color="black")
    ax.axhline(0, linewidth=0.8, color="black")

    # Mark extrema
    ax.plot(x_max, d_max, "o", color="maroon", ms=10, zorder=5)
    ax.plot(x_min, d_min, "o", color="grey", ms=10, zorder=5)

    # --- Axis padding (prevents border clash) ---
    y_range = d.max() - d.min()
    ax.set_ylim(
        d.min() - 0.25 * y_range,
        d.max() + 0.25 * y_range
    )

    x_range = x.max() - x.min()
    ax.set_xlim(
        x.min() - 0.02 * x_range,
        x.max() + 0.05 * x_range
    )

    # --- Smart annotation placement ---
    ax.annotate(
        f"{d_max:.1f} mm",
        (x_max, d_max),
        xytext=(-8, 10),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="maroon",
        clip_on=False,
    )

    ax.annotate(
        f"{d_min:.1f} mm",
        (x_min, d_min),
        xytext=(6, -14),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
        color="black",
        clip_on=False,
    )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("Deflection (mm)")
    ax.set_title("Deflected Shape")

    ax.grid(True)
    fig.tight_layout(pad=1.6)
    plt.show()

