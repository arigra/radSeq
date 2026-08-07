"""Sequence visualization: frame grids and GIFs.

Both renderers can mark target positions with red circles: ground-truth
trajectories when `traj` is given (real/cached sequences), otherwise
detected peaks from src.eval.metrics (generated sequences have no GT).
"""
import matplotlib
matplotlib.use("Agg")
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