"""Sequence visualization: frame grids and GIFs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def sequence_grid(x, path, ncols=8):
    x = x.detach().cpu().numpy()
    L = x.shape[0]
    nrows = (L + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    vmin, vmax = x.min(), x.max()
    for i, ax in enumerate(np.atleast_1d(axes).flatten()):
        ax.axis("off")
        if i < L:
            ax.imshow(x[i], vmin=vmin, vmax=vmax, cmap="viridis",
                      origin="lower", aspect="auto")
            ax.set_title(f"t={i}", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def sequence_gif(x, path, fps=4):
    import imageio.v2 as imageio
    x = x.detach().cpu().numpy()
    lo, hi = x.min(), x.max()
    frames = ((x - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)
    # imageio's GIF (pillow) writer deprecated `fps` in favor of `duration`
    # (ms per frame); convert to keep output warning-free.
    imageio.mimsave(path, list(frames), duration=1000 / fps)
