"""Sequence visualization: inline notebook plots, frame grids and GIFs.

show_sequence draws a batch from generate_sequences inline in a notebook.
sequence_grid / sequence_gif write files and can mark target positions with
red circles: ground-truth trajectories when `traj` is given (real/cached
sequences), otherwise detected peaks from src.eval.metrics.

No backend is forced here: notebooks keep their inline backend, and scripts
only save files, which works on any backend (headless machines default to Agg).
"""
import matplotlib.pyplot as plt
import numpy as np


def _frame_marks(x, traj=None, n_targets=None):
    """Per-frame (m, 2) arrays of (range_bin, doppler_bin) marker positions.

    Returns (marks, source_label). traj is (max_targets, L, 2) as stored in
    the cache; rows past n_targets are zero padding and are dropped.
    """
    L = x.shape[0]
    if traj is not None:
        t = traj.detach().cpu()
        m = int(n_targets) if n_targets is not None else t.shape[0]
        return [t[:m, l].numpy() for l in range(L)], "GT targets"
    from src.eval.metrics import detect_peaks
    return [detect_peaks(f).numpy() for f in x], "detected peaks"


def _draw_marks(ax, marks):
    if marks is not None and len(marks):
        ax.scatter(marks[:, 1], marks[:, 0], s=140, facecolors="none",
                   edgecolors="red", linewidths=1.6)


def sequence_grid(x, path, ncols=4, traj=None, n_targets=None, mark=True):
    x = x.detach().cpu()
    L = x.shape[0]
    marks, src = (_frame_marks(x, traj, n_targets) if mark
                  else ([None] * L, None))
    nrows = (L + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.4 * ncols, 3.2 * nrows),
                             constrained_layout=True)
    vmin, vmax = float(x.min()), float(x.max())
    arr = x.numpy()
    im = None
    for i, ax in enumerate(np.atleast_1d(axes).flatten()):
        ax.axis("off")
        if i < L:
            im = ax.imshow(arr[i], vmin=vmin, vmax=vmax, cmap="viridis",
                           origin="lower", aspect="equal")
            _draw_marks(ax, marks[i])
            ax.set_title(f"t={i}", fontsize=11)
    if im is not None:
        cbar = fig.colorbar(im, ax=np.atleast_1d(axes).ravel().tolist(),
                            fraction=0.02, pad=0.01)
        cbar.set_label("dB")
    if src:
        fig.suptitle(f"red circles: {src}   |   x: Doppler bin, y: range bin",
                     fontsize=12)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def sequence_gif(x, path, fps=4, traj=None, n_targets=None, mark=True,
                 size=5.0):
    import imageio.v2 as imageio
    x = x.detach().cpu()
    L = x.shape[0]
    marks, src = (_frame_marks(x, traj, n_targets) if mark
                  else ([None] * L, None))
    vmin, vmax = float(x.min()), float(x.max())
    frames = []
    for i in range(L):
        fig, ax = plt.subplots(figsize=(size, size), dpi=100)
        ax.imshow(x[i].numpy(), vmin=vmin, vmax=vmax, cmap="viridis",
                  origin="lower", aspect="equal")
        _draw_marks(ax, marks[i])
        title = f"frame {i + 1}/{L}"
        if src:
            title += f"   (red: {src})"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Doppler bin")
        ax.set_ylabel("range bin")
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    # imageio's GIF (pillow) writer deprecated `fps` in favor of `duration`
    # (ms per frame); loop=0 makes the GIF repeat indefinitely.
    imageio.mimsave(path, frames, duration=1000 / fps, loop=0)


def show_sequence(data, i=0, frames=(0, 5, 10, 15), title=""):
    """Plot frames of sequence i from a generate_sequences batch, inline.

    Red circles mark the true target positions. One shared colour scale
    across the frames, so brightness is comparable between them.
    """
    x = data["x"][i].detach().cpu().numpy()
    m = int(data["n_targets"][i])
    traj = data["traj"][i, :m].detach().cpu().numpy()
    fig, axes = plt.subplots(1, len(frames), figsize=(3 * len(frames), 3.2),
                             constrained_layout=True, squeeze=False)
    axes = axes[0]
    for ax, t in zip(axes, frames):
        im = ax.imshow(x[t], origin="lower", vmin=x.min(), vmax=x.max())
        ax.scatter(traj[:, t, 1], traj[:, t, 0], s=120,
                   facecolors="none", edgecolors="red")
        ax.set(title=f"frame {t}", xlabel="Doppler bin")
    axes[0].set_ylabel("range bin")
    fig.colorbar(im, ax=axes, label="dB", shrink=0.8)
    fig.suptitle(title)
    plt.show()


def show_rows(datasets, labels, i=0, frames=(0, 5, 10, 15), title="", hist=False):
    """Several scenes as rows on ONE shared colour scale, so they compare directly.

    Red circles mark the true targets. With hist=True a last column shows each
    scene's distribution of dB values (log density), e.g. the floor noise creates.
    """
    xs = [d["x"][i].detach().cpu().numpy() for d in datasets]
    lo = min(float(x.min()) for x in xs)
    hi = max(float(x.max()) for x in xs)
    ncols = len(frames) + (1 if hist else 0)
    fig, axes = plt.subplots(len(datasets), ncols,
                             figsize=(3 * ncols, 2.9 * len(datasets)),
                             constrained_layout=True, squeeze=False)
    im = None
    for row, (d, x, label) in enumerate(zip(datasets, xs, labels)):
        # unconditional generated samples have maps but no target labels
        traj = (d["traj"][i, :int(d["n_targets"][i])].detach().cpu().numpy()
                if "traj" in d else None)
        for col, t in enumerate(frames):
            ax = axes[row, col]
            im = ax.imshow(x[t], origin="lower", vmin=lo, vmax=hi)
            if traj is not None:
                ax.scatter(traj[:, t, 1], traj[:, t, 0], s=100,
                           facecolors="none", edgecolors="red")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"frame {t}")
            if col == 0:
                ax.set_ylabel(label)
        if hist:
            ax = axes[row, -1]
            ax.hist(x.ravel(), bins=100, range=(lo, hi), density=True, color="#3b6ea5")
            ax.set_yscale("log")
            ax.set_xlabel("dB")
            if row == 0:
                ax.set_title("value distribution")
    fig.colorbar(im, ax=axes[:, :len(frames)].ravel().tolist(), label="dB", shrink=0.8)
    fig.suptitle(title)
    plt.show()


def _target_peaks(x, traj, window=3):
    """Brightest pixel within `window` bins of the true position, per frame.

    x: (L, N, K) dB map; traj: (L, 2) true (range, Doppler) bins.
    Returns positions (L, 2) and their dB values (L,).
    """
    L, N, K = x.shape
    pos, val = np.zeros((L, 2)), np.zeros(L)
    for t in range(L):
        r, d = np.rint(traj[t]).astype(int)
        r0, r1 = max(r - window, 0), min(r + window + 1, N)
        d0, d1 = max(d - window, 0), min(d + window + 1, K)
        patch = x[t, r0:r1, d0:d1]
        k = np.unravel_index(np.argmax(patch), patch.shape)
        pos[t] = (r0 + k[0], d0 + k[1])
        val[t] = patch[k]
    return pos, val


def show_kinematics(datasets, labels, i=0, target=0):
    """How one target moves, and whether the map agrees with its label.

    Four panels, one line per scene:
      track        the true path through the RD plane (line) and the brightest
                   pixel near it in each frame (x)
      pixel vs     that pixel's distance from the label; a target between bins
      label        puts its peak in a neighbour, so up to ~1.5 bins is expected
      second       |r[t+1] - 2 r[t] + r[t-1]|: ~0 for physical motion, large
      difference   for unrelated bright pixels
      brightness   the target's peak dB per frame (flat, or flickering)
    """
    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6), constrained_layout=True)
    for k, (d, label) in enumerate(zip(datasets, labels)):
        c = colours[k % len(colours)]
        x = d["x"][i].detach().cpu().numpy()
        traj = d["traj"][i, target].detach().cpu().numpy()
        pos, val = _target_peaks(x, traj)
        axes[0].plot(traj[:, 1], traj[:, 0], "-", color=c, lw=1.6, label=label)
        axes[0].plot(pos[:, 1], pos[:, 0], "x", color=c, ms=6)
        axes[1].plot(np.linalg.norm(pos - traj, axis=1), "o-", color=c, ms=3, label=label)
        second = np.linalg.norm(traj[2:] - 2 * traj[1:-1] + traj[:-2], axis=1)
        axes[2].plot(range(1, len(traj) - 1), second, "o-", color=c, ms=3, label=label)
        axes[3].plot(val, "o-", color=c, ms=3, label=label)
    axes[0].set(xlabel="Doppler bin", ylabel="range bin",
                title="track: line = label, x = brightest pixel")
    axes[1].axhspan(0, 1.5, color="grey", alpha=0.2)
    axes[1].set(xlabel="frame", ylabel="bins", title="brightest pixel vs label")
    axes[2].set(xlabel="frame", ylabel="bins", title="second difference of the track")
    axes[2].set_ylim(bottom=0)   # else float jitter on a constant auto-scales into fake wiggles
    axes[3].set(xlabel="frame", ylabel="dB", title="target peak brightness")
    for ax in axes:
        ax.legend(fontsize=8)
    plt.show()


def _grid():
    """Range (m) and radial-velocity (m/s) axes of the RD map."""
    from src.simulator import TemporalRadarSimulator   # deferred: keeps viz import light
    sim = TemporalRadarSimulator(seq_len=1)
    return sim.R.numpy(), sim.V.numpy()


def _nearest_bin(data, i, target, t):
    rb, db = np.rint(data["traj"][i, target, t].detach().cpu().numpy()).astype(int)
    return int(np.clip(rb, 0, 63)), int(np.clip(db, 0, 63))


def show_cuts(data, i=0, t=0, target=0):
    """One frame in physical units, plus cuts through the target along range and
    along Doppler: the sidelobe shape that draws a single target as a cross."""
    R, V = _grid()
    x = data["x"][i, t].detach().cpu().numpy()
    rb, db = _nearest_bin(data, i, target, t)
    top = float(x.max())
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.3, 1, 1]})
    im = axes[0].imshow(x, origin="lower", aspect="auto",
                        extent=[V[0], V[-1], R[0], R[-1]])
    axes[0].set(xlabel="radial velocity (m/s)", ylabel="range (m)", title=f"frame {t}")
    fig.colorbar(im, ax=axes[0], label="dB")
    axes[1].plot(R, x[:, db], color="#3b6ea5")
    axes[1].axvline(R[rb], color="red", ls="--", lw=1)
    axes[1].set(xlabel="range (m)", ylabel="dB", title="cut along range",
                ylim=(top - 140, top + 5))
    axes[2].plot(V, x[rb, :], color="#3b6ea5")
    axes[2].axvline(V[db], color="red", ls="--", lw=1)
    axes[2].set(xlabel="radial velocity (m/s)", ylabel="dB", title="cut along Doppler",
                ylim=(top - 140, top + 5))
    plt.show()


def _lag1(maps):
    """Mean correlation between consecutive frames of real (L, N, K) maps."""
    a, b = maps[:-1].reshape(len(maps) - 1, -1), maps[1:].reshape(len(maps) - 1, -1)
    a = a - a.mean(1, keepdims=True)
    b = b - b.mean(1, keepdims=True)
    return float(((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))).mean())


def show_clutter_correlation(rhos=(0.0, 0.2, 0.4, 0.6, 0.8, 0.95), nu=0.5,
                             n_seq=8, seq_len=16, seed=0):
    """What rho controls, measured on clutter alone.

    Two lines per requested rho: the clutter signal's frame-to-frame correlation,
    which follows rho, and the dB map's, which is high even at rho = 0 because the
    clutter's Doppler band and range texture are fixed for the whole sequence;
    only the fine speckle on top decorrelates.
    """
    import torch
    from src.simulator import TemporalRadarSimulator, create_rd_map
    torch.manual_seed(seed)
    sim = TemporalRadarSimulator(seq_len=seq_len)
    iq_corr, db_corr = [], []
    for rho in rhos:
        iq_vals, db_vals = [], []
        for _ in range(n_seq):
            C = sim._clutter_frames(rho=rho, nu=nu)                 # (L, N, K) complex IQ
            a, b = C[:-1].flatten(1), C[1:].flatten(1)
            iq_vals.append(float(((a.conj() * b).sum(1).real
                                  / (a.abs().pow(2).sum(1).sqrt()
                                     * b.abs().pow(2).sum(1).sqrt())).mean()))
            db = torch.stack([20 * torch.log10(create_rd_map(c).abs() + 1e-6) for c in C])
            db_vals.append(_lag1(db.float().numpy()))
        iq_corr.append(np.mean(iq_vals))
        db_corr.append(np.mean(db_vals))
    fig, ax = plt.subplots(figsize=(5.8, 3.8), constrained_layout=True)
    ax.plot(rhos, iq_corr, "o-", color="#3b6ea5", label="clutter signal")
    ax.plot(rhos, db_corr, "s-", color="#d1791e", label="dB map, as plotted")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="y = rho")
    ax.set(xlabel="requested rho", ylabel="consecutive-frame correlation",
           title="what rho controls", ylim=(-0.1, 1.05))
    ax.legend(fontsize=8)
    plt.show()


def show_range_profile(datasets, labels, i=0, t=0, target=0, span=5):
    """dB along range through the target's Doppler bin, near the target.

    At a bin centre a point target fills one range bin; an extended target,
    three scatterers 3 m apart, fills three.
    """
    R, _ = _grid()
    fig, ax = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
    top = -np.inf
    for d, label in zip(datasets, labels):
        x = d["x"][i, t].detach().cpu().numpy()
        rb, db = _nearest_bin(d, i, target, t)
        lo, hi = max(rb - span, 0), min(rb + span + 1, x.shape[0])
        ax.plot(R[lo:hi], x[lo:hi, db], "o-", label=label)
        top = max(top, float(x[lo:hi, db].max()))
    ax.set(xlabel="range (m)", ylabel="dB", title="range profile through the target",
           ylim=(top - 80, top + 5))
    ax.legend(fontsize=8)
    plt.show()


def show_brightness(datasets, labels, i=0, target=0):
    """The target's peak brightness in each frame, one line per scene.

    A steady target dips only a little when it falls between bins; a
    Swerling-1 target's power is redrawn every frame, so it jumps.
    """
    fig, ax = plt.subplots(figsize=(6.5, 3.6), constrained_layout=True)
    for d, label in zip(datasets, labels):
        x = d["x"][i].detach().cpu().numpy()
        traj = d["traj"][i, target].detach().cpu().numpy()
        _, val = _target_peaks(x, traj)
        ax.plot(val, "o-", ms=3, label=label)
    ax.set(xlabel="frame", ylabel="dB", title="target peak brightness")
    ax.legend(fontsize=8)
    plt.show()


def show_normalization(x_db, stats):
    """Pixel values before and after normalisation, side by side.

    Left: raw dB, with the training mean (black) and one std either side (grey).
    Right: normalised values, with 0 and +/-1 marked, and where the brightest
    1% and 0.1% of pixels start (dashed): the tail where the targets live.
    """
    x = x_db.detach().cpu().numpy().ravel()[::7]
    xn = (x - stats["mean"]) / stats["std"]
    p99, p999 = np.quantile(xn, [0.99, 0.999])
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), constrained_layout=True)

    ax = axes[0]
    ax.hist(x, bins=150, density=True, color="#3b6ea5")
    ax.axvline(stats["mean"], color="black", ls="--", lw=1.5,
               label=f"mean {stats['mean']:.1f} dB")
    for sign in (-1, 1):
        ax.axvline(stats["mean"] + sign * stats["std"], color="grey", ls=":", lw=1.5,
                   label=f"one std ({stats['std']:.1f} dB)" if sign == 1 else None)
    ax.set(xlabel="dB", ylabel="density", yscale="log", title="raw")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(xn, bins=150, density=True, color="#3b6ea5")
    ax.axvline(0, color="black", ls="--", lw=1.5, label="0")
    for sign in (-1, 1):
        ax.axvline(sign, color="grey", ls=":", lw=1.5, label="+/-1" if sign == 1 else None)
    ax.axvline(p99, color="#d1791e", ls="--", lw=1.5, label=f"brightest 1% from {p99:.2f}")
    ax.axvline(p999, color="#b02418", ls="--", lw=1.5, label=f"brightest 0.1% from {p999:.2f}")
    ax.set(xlabel="normalised value", yscale="log", title="normalised")
    ax.legend(fontsize=8)
    plt.show()


def show_real_vs_sim(real_db, sim_db, targets=None, title="",
                     percentiles=(5, 99.5)):
    """One real RADIal range-Doppler map beside a simulated one.

    Both are drawn on their own colour scale, set by percentiles rather than
    min/max: the two differ by tens of dB in absolute level and in dynamic
    range, so a shared scale would render one of them flat and hide the
    structure the comparison is about. Axes are physical (metres, Doppler bin).
    """
    real_db, sim_db = np.asarray(real_db), np.asarray(sim_db)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, m, name in zip(axes, (real_db, sim_db), ("real RADIal", "simulated")):
        lo, hi = np.percentile(m, percentiles)
        im = ax.imshow(m, aspect="auto", origin="lower", vmin=lo, vmax=hi,
                       cmap="viridis",
                       extent=[0, m.shape[1], 0, m.shape[0] * 0.2])
        ax.set_title(f"{name}\n{m.min():.0f} to {m.max():.0f} dB "
                     f"(span {m.max() - m.min():.0f})")
        ax.set_xlabel("Doppler bin")
        ax.set_ylabel("range (m)")
        fig.colorbar(im, ax=ax, label="dB")
    if targets is not None:
        for rb, db in targets:
            axes[0].plot(db, rb * 0.2, "o", mfc="none", mec="red", ms=11, mew=1.6)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def show_fidelity_curve(deltas, arms, title="simulator mismatch vs detector AP"):
    """Detector accuracy against simulator mismatch, one line per arm.

    The claim under test is the *interaction*: the raw-simulator line should
    fall as the simulator gets worse while the generator line stays flat.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    styles = {"real": ("grey", "--", "real only"),
              "real_sim": ("tab:red", "-o", "real + raw simulator"),
              "real_synth": ("tab:blue", "-o", "real + DiT (pretrained on it, fine-tuned)")}
    for name, values in arms.items():
        colour, style, label = styles.get(name, ("black", "-o", name))
        ax.plot(deltas, values, style, color=colour, label=label)
    ax.set_xlabel("simulator mismatch (dB of target brightness)")
    ax.set_ylabel("detector mAP")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
