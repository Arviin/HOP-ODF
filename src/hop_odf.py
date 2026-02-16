"""
HOP–ODF visualization (synthetic)

Generates synthetic fibre configurations in a disk and their corresponding
orientation distribution functions (ODF) for prescribed Hermans Orientation
Parameter (HOP) values (including negative values for perpendicular alignment).

Author: Arvin (Fazel) Mirzaei
License: MIT 
"""

import numpy as np
import matplotlib as mpl
mpl.rcParams.update({
    "font.family": "DejaVu Sans",  # ships with matplotlib -> consistent across OS
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

import matplotlib.pyplot as plt


# ============================================================
# 1) Sample fibre angles (undirected)
# ============================================================
def sample_fibre_angles(kappa: float, n: int, seed: int, mu: float = 0.0) -> np.ndarray:
    """
    Sample undirected fibre orientation angles theta in [0, pi).

    Parameters
    ----------
    kappa : float
        Concentration of the von Mises distribution.
        kappa=0 => uniform/random; larger => tighter clustering around mu.
    n : int
        Number of fibres.
    seed : int
        Random seed for reproducibility.
    mu : float
        Mean direction (radians). Use 0 for parallel; pi/2 for perpendicular.

    Returns
    -------
    theta : ndarray, shape (n,)
        Undirected angles in [0, pi).
    """
    rng = np.random.default_rng(seed)
    ang = rng.vonmises(mu=mu, kappa=kappa, size=n)  # [-pi, pi]
    ang = np.mod(ang, 2 * np.pi)                    # [0, 2pi)
    theta = np.mod(ang, np.pi)                      # undirected -> [0, pi)
    return theta


# ============================================================
# 2) HOP (Hermans Orientation Parameter)
# ============================================================
def hop_hermans_3d(theta: np.ndarray) -> float:
    """
    Hermans orientation parameter using 3D solid-angle weighting.

        HOP = 0.5 * (3 <cos^2(theta)>_w - 1)

    where <...>_w is weighted by sin(theta), consistent with 3D averaging.

    Range:
      -0.5  => perfect perpendicular
       0.0  => isotropic/random
       1.0  => perfect parallel alignment
    """
    w = np.sin(theta)
    w = np.clip(w, 1e-12, None)
    A = np.sum(w * (np.cos(theta) ** 2)) / np.sum(w)
    return 0.5 * (3.0 * A - 1.0)


def kappa_for_target_hop_abs(target_abs: float, n_probe: int = 22000, seed: int = 123) -> float:
    """
    Numerically invert kappa for a target |HOP| by bisection.

    We solve under the convention mu=0 (parallel case), so the returned kappa
    corresponds to a *positive* HOP of approximately target_abs.
    For negative HOP panels we set mu=pi/2 (perpendicular) but keep the same kappa
    magnitude to represent the same "strength" of ordering.

    Parameters
    ----------
    target_abs : float
        Desired magnitude of HOP in [0, 1).
    n_probe : int
        Number of samples used during inversion.
    seed : int
        Seed for inversion reproducibility.

    Returns
    -------
    kappa : float
        Concentration parameter producing approximately target_abs.
    """
    target_abs = float(np.clip(target_abs, 0.0, 0.999))
    if target_abs <= 0.0:
        return 0.0
    if target_abs >= 0.999:
        return 1000.0

    lo, hi = 0.0, 1000.0
    for _ in range(36):
        mid = 0.5 * (lo + hi)
        th = sample_fibre_angles(mid, n_probe, seed=seed, mu=0.0)
        hop_mid = hop_hermans_3d(th)
        if hop_mid > target_abs:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ============================================================
# 3) ODF (Orientation Distribution Function)
# ============================================================
def odf_density(theta: np.ndarray, nbins: int = 180, ngrid: int = 720):
    """
    Compute ODF as a probability density histogram over theta in [0, pi),
    then map to a full polar 0..2pi using alpha = 2*theta (undirected fibres).

    Returns
    -------
    alpha : ndarray
        Polar angles in [0, 2pi).
    y : ndarray
        ODF density values (area over theta is 1). Autoscaled in plotting.
    """
    edges = np.linspace(0, np.pi, nbins + 1)
    counts, _ = np.histogram(theta, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    dens = counts.astype(float)
    dens = dens / np.trapezoid(dens, centers)  # probability density over theta

    alpha = np.linspace(0, 2 * np.pi, ngrid, endpoint=False)  # alpha = 2*theta
    y = np.interp(alpha / 2, centers, dens, left=dens[0], right=dens[-1])

    # mild smoothing for nicer visual curves (optional)
    k = 7
    kernel = np.ones(k) / k
    y_pad = np.r_[y[-k:], y, y[:k]]
    y = np.convolve(y_pad, kernel, mode="same")[k:-k]

    return alpha, y


# ============================================================
# 4) Fibre cross-section drawing (segments in a disk)
# ============================================================
def random_points_in_disk(n: int, seed: int, r: float = 1.0):
    """Uniform random points in a disk of radius r."""
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    ang = rng.random(n) * 2 * np.pi
    rad = r * np.sqrt(u)
    return rad * np.cos(ang), rad * np.sin(ang)


def draw_disk(ax, radius=1.0):
    ax.add_patch(plt.Circle((0, 0), radius, fill=False, lw=1.4, color="black"))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-1.08 * radius, 1.08 * radius)
    ax.set_ylim(-1.08 * radius, 1.08 * radius)
    ax.set_xticks([])
    ax.set_yticks([])


def draw_fibres_in_disk(ax, theta_plane, seed, n_fibres, L=0.12, lw=0.8, radius=1.0):
    x, y = random_points_in_disk(n_fibres, seed=seed, r=radius * 0.94)
    rng = np.random.default_rng(seed + 1)

    lengths = L * rng.uniform(0.90, 1.10, n_fibres)
    dx = 0.5 * lengths * np.cos(theta_plane)
    dy = 0.5 * lengths * np.sin(theta_plane)

    for xi, yi, dxi, dyi in zip(x, y, dx, dy):
        ax.plot(
            [xi - dxi, xi + dxi],
            [yi - dyi, yi + dyi],
            lw=lw,
            alpha=0.85,
            color="black",
            solid_capstyle="round",
        )


def add_reference_arrow(ax, radius=1.0):
    ax.annotate(
        "",
        xy=(0, 0.92 * radius),
        xytext=(0, 0.60 * radius),
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black"),
    )


def panel_label(ax, text):
    ax.text(
        0.98, 1.05,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=13,
        fontweight="bold",
    )


# ============================================================
# 5) Main plotting routine
# ============================================================
def plot_hop_sweep(
    hop_values=None,
    n_fibres=800,
    L=0.12,
    lw=0.8,
    nbins_odf=180,
    ngrid_odf=720,
    ncols=2,
    wspace=0.20,
    hspace=0.30,base_seed = 12345,
):
    """
    Plot fibre cross-sections + ODFs for a list of HOP values.

    Notes
    -----
    - Positive HOP => preferential alignment with reference (mu=0)
    - Negative HOP => preferential alignment perpendicular (mu=pi/2)
    - kappa is solved from |HOP| to control "ordering strength"
    - ODF radius is autoscaled per panel for readability
    """
    if hop_values is None:
        hop_values = [-0.5, -0.3, -0.1] + list(np.round(np.linspace(0, 1, 11), 1))

    hop_values = list(hop_values)
    n_panels = len(hop_values)
    nrows = int(np.ceil(n_panels / ncols))

    # Precompute data
    records = []
    for idx, hop_target in enumerate(hop_values):
        kappa = kappa_for_target_hop_abs(abs(hop_target), n_probe=22000, seed= base_seed + 100 + idx)
        mu = 0.0 if hop_target >= 0 else np.pi / 2
        theta = sample_fibre_angles(kappa, n_fibres, seed=base_seed + 5000 + idx * 111, mu=mu)
        alpha, y = odf_density(theta, nbins=nbins_odf, ngrid=ngrid_odf)
        records.append((hop_target, theta, alpha, y))

    # Figure
    fig_h = 3.0 * nrows
    fig = plt.figure(figsize=(14, fig_h))

    gs = fig.add_gridspec(
        nrows, ncols,
        left=0.05, right=0.98, top=0.98, bottom=0.04,
        wspace=wspace, hspace=hspace
    )


    # ---------------------------------------------------------
    # Column headers (added once for the entire figure)
    # ---------------------------------------------------------
    fig.text(
        0.13, 1.01,
        "Fibre Cross-Section",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold"
    )

    fig.text(
        0.53, 1.01,
        "Orientation Distribution Function (ODF)",
        ha="center",
        va="top",
        fontsize=14,
        fontweight="bold"
    )


    for i, (hop_target, theta, alpha, y) in enumerate(records):
        r = i // ncols
        c = i % ncols

        sub = gs[r, c].subgridspec(1, 2, width_ratios=[1.0, 1.05], wspace=0.08)
        ax_disk = fig.add_subplot(sub[0, 0])
        ax_polar = fig.add_subplot(sub[0, 1], projection="polar")

        # Disk with fibres
        theta_plane = (np.pi / 2) - theta
        draw_disk(ax_disk)
        draw_fibres_in_disk(ax_disk, theta_plane, seed=base_seed + 9000 + i, n_fibres=n_fibres, L=L, lw=lw)
        add_reference_arrow(ax_disk)
        panel_label(ax_disk, f"HOP = {hop_target:.1f}")

        # Polar ODF
        ax_polar.plot(alpha, y, lw=2.4, color="black")
        ax_polar.fill(alpha, y, alpha=0.10, color="black")
        ax_polar.set_theta_zero_location("N")
        ax_polar.set_theta_direction(-1)

        # reference arrow in polar plot
        ax_polar.annotate(
            "",
            xy=(0, ax_polar.get_rmax() * 0.90),
            xytext=(0, ax_polar.get_rmax() * 0.60),
            arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black"),
        )

        ax_polar.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2])
        ax_polar.set_xticklabels(["0°", "90°", "180°", "270°"], fontsize=11, fontweight="bold")
        ax_polar.set_yticks([])
        ax_polar.set_yticklabels([])
        ax_polar.spines["polar"].set_linewidth(1.2)
        ax_polar.set_title("")

    # Disable unused cells
    for j in range(n_panels, nrows * ncols):
        rr = j // ncols
        cc = j % ncols
        ax_off = fig.add_subplot(gs[rr, cc])
        ax_off.axis("off")

    plt.show()


def main():
    # You can customize hop_values here if needed:
    # hop_values = [-0.5, -0.3, -0.1, 0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    plot_hop_sweep()


if __name__ == "__main__":
    main()
