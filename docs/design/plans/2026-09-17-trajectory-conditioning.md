# Trajectory-Conditioned DiT Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the working full-data DiT generate the targets it is asked for (per-target range/Doppler path and class), so every synthetic sequence comes with exact labels.

**Architecture:** Requested targets are drawn as a 4-channel condition map aligned with the RD map (a Gaussian blob per target in its class channel, plus a presence channel). The DiT patchifies the noisy map and the condition channels together; only its input projection grows, warm-started from the passing unconditional checkpoint with zero weights on the new columns. Classifier-free guidance by dropping the whole condition for 10% of training sequences. Scored with a trajectory hit-rate test relative to the simulator plus Ari's four checks.

**Tech Stack:** PyTorch 2.12 (`/truenas/home/arigra/.venv/bin/python`), pytest, existing radSeq modules (`src/dit.py`, `src/train.py`, `src/sample.py`, `src/eval/metrics.py`).

Spec: `docs/design/specs/2026-09-17-trajectory-conditioning-design.md`.

## Global Constraints

- Run everything from the repo root `/truenas/home/arigra/permuter/ariGranevich/radSeq`; tests with `CUDA_VISIBLE_DEVICES= /truenas/home/arigra/.venv/bin/python -m pytest -q <path>`.
- Condition map shape: `(B, L, 4, 64, 64)`; channels 0-2 = classes 0 steady, 1 Swerling-1, 2 extended; channel 3 = presence; blob sigma = 1 bin, peak value 1.
- `traj[..., 0]` is the range bin (map row), `traj[..., 1]` the Doppler bin (map column).
- Conditioning is on iff `model.cond_channels > 0` (config key `model.cond_channels: 4`).
- Warm start: `train.init_from` = `checkpoints/e3_long_bs32/last.pt`, EMA weights if present; ignored when resuming.
- `train.cond_dropout: 0.1`; guidance `pred = uncond + w * (cond - uncond)`, uncond = zero condition map.
- Adherence: evaluator detector (`detect_peaks`, 12 dB over median, top 5), radius 2 bins, lasting track = linked track of >= 8 frames.
- Decision rule: hit rate >= simulator hit rate - 0.05; unrequested lasting tracks/seq <= simulator's; Ari's four checks (std within 0.1, marginal L1 <= 0.15, target tracks within 15%, persistence within 25% of real) pass. Null check: unconditional `e3_long` hit rate at least 0.20 below the conditional model's.
- Guidance sweep w in {1, 2, 3} on val 0-95 / seeds 1-3; verdict on val 96-287 / seeds 4-9; DDIM 30, EMA weights.
- Tests and runs must not write to `archive/ablations_aug/logs/phase1.log` (always set `train.log_file`).
- Commit messages end with `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.

---

### Task 1: Condition map

**Files:**
- Create: `src/trajectory_condition.py`
- Test: `tests/test_trajectory_condition.py`

**Interfaces:**
- Produces: `COND_CHANNELS = 4`; `render_condition(traj, n_targets, cls, n_range=64, n_doppler=64, sigma=1.0) -> Tensor (B, L, 4, n_range, n_doppler)`; `drop_condition(cond, p, generator=None) -> Tensor` (same shape).

- [ ] **Step 1: Write the failing tests** — `tests/test_trajectory_condition.py`:

```python
import torch

from src.trajectory_condition import COND_CHANNELS, drop_condition, render_condition


def _one_target(r, d, cls_id, n_frames=2):
    traj = torch.zeros(1, 5, n_frames, 2)
    traj[0, 0, :, 0], traj[0, 0, :, 1] = r, d
    return traj, torch.tensor([1]), torch.tensor([[cls_id, 0, 0, 0, 0]])


def test_blob_peaks_at_requested_bin_in_its_class_channel():
    traj, n, cls = _one_target(20.0, 41.0, cls_id=2)
    c = render_condition(traj, n, cls)
    assert c.shape == (1, 2, COND_CHANNELS, 64, 64)
    peak = int(c[0, 0, 2].argmax())
    assert (peak // 64, peak % 64) == (20, 41)
    assert float(c[0, 0, 2, 20, 41]) == 1.0
    assert float(c[0, 0, 0].max()) == 0.0 and float(c[0, 0, 1].max()) == 0.0


def test_padded_targets_are_ignored():
    traj, n, cls = _one_target(20.0, 41.0, cls_id=0)
    traj[0, 1, :, 0], traj[0, 1, :, 1] = 50.0, 10.0      # slot 1 is padding (n_targets = 1)
    assert float(render_condition(traj, n, cls)[0, 0, 0, 50, 10]) < 1e-6


def test_presence_channel_is_ones():
    traj, n, cls = _one_target(5.0, 5.0, cls_id=1)
    assert torch.equal(render_condition(traj, n, cls)[:, :, 3], torch.ones(1, 2, 64, 64))


def test_drop_condition_zeroes_whole_sequences():
    torch.manual_seed(0)
    cond = torch.ones(200, 2, COND_CHANNELS, 4, 4)
    per_seq = drop_condition(cond, 0.5).flatten(1)
    assert torch.equal(per_seq.min(1).values, per_seq.max(1).values)   # all or nothing
    assert 60 < int((per_seq.max(1).values == 0).sum()) < 140
    assert torch.equal(drop_condition(cond, 0.0), cond)
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES= /truenas/home/arigra/.venv/bin/python -m pytest -q tests/test_trajectory_condition.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.trajectory_condition'`.

- [ ] **Step 3: Implement** — `src/trajectory_condition.py`:

```python
"""Trajectory condition maps for the conditional DiT.

A requested scene is a list of targets, each with a (range, Doppler) bin per
frame and a class. It is drawn as 4 channels per frame, aligned with the RD
map: a Gaussian blob per target in its class's channel (0 steady,
1 Swerling-1, 2 extended), and a presence channel of ones meaning "a
condition was given". All-zero channels are the "no condition" input used
for classifier-free guidance.
"""
import torch
import torch.nn.functional as F

N_CLASSES = 3
COND_CHANNELS = N_CLASSES + 1


def render_condition(traj, n_targets, cls, n_range=64, n_doppler=64, sigma=1.0):
    """traj (B, M, L, 2) bins [range, Doppler]; n_targets (B,); cls (B, M).

    Returns (B, L, 4, n_range, n_doppler) with values in [0, 1].
    """
    B, M, L, _ = traj.shape
    device = traj.device
    r = torch.arange(n_range, device=device, dtype=torch.float32).view(1, 1, 1, n_range, 1)
    d = torch.arange(n_doppler, device=device, dtype=torch.float32).view(1, 1, 1, 1, n_doppler)
    tr = traj[..., 0].float().view(B, M, L, 1, 1)
    td = traj[..., 1].float().view(B, M, L, 1, 1)
    blobs = torch.exp(-((r - tr) ** 2 + (d - td) ** 2) / (2 * sigma ** 2))  # (B, M, L, N, K)
    real = torch.arange(M, device=device)[None] < n_targets.to(device)[:, None]  # (B, M)
    onehot = F.one_hot(cls.to(device).long(), N_CLASSES).float()            # (B, M, 3)
    weight = (onehot * real[..., None]).view(B, M, N_CLASSES, 1, 1, 1)
    per_class = (blobs[:, :, None] * weight).amax(dim=1)                   # (B, 3, L, N, K)
    presence = torch.ones(B, 1, L, n_range, n_doppler, device=device)
    return torch.cat([per_class, presence], dim=1).permute(0, 2, 1, 3, 4).contiguous()


def drop_condition(cond, p, generator=None):
    """Zero the whole condition (all channels, all frames) for a fraction p of sequences."""
    if p <= 0:
        return cond
    keep = torch.rand(cond.shape[0], device=cond.device, generator=generator) >= p
    return cond * keep.view(-1, 1, 1, 1, 1).to(cond.dtype)
```

- [ ] **Step 4: Run to verify pass**

Run: `CUDA_VISIBLE_DEVICES= /truenas/home/arigra/.venv/bin/python -m pytest -q tests/test_trajectory_condition.py`
Expected: `4 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/trajectory_condition.py tests/test_trajectory_condition.py
git commit -m "feat: render requested target trajectories as condition maps

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 2: DiT condition channels and warm start

**Files:**
- Modify: `src/dit.py` (`TemporalDiT.__init__`, `TemporalDiT.forward`; add `load_unconditional_weights`)
- Modify: `src/train.py` (`build_model`)
- Test: `tests/test_dit.py`

**Interfaces:**
- Consumes: nothing from Task 1 (channel count is an int).
- Produces: `TemporalDiT(..., cond_channels=0)`; `TemporalDiT.forward(x, t, cond=None, cond_map=None)` with `cond_map` `(B, L, cond_channels, N, K)` or None (None = zeros); `load_unconditional_weights(model, state_dict) -> None`; `build_model` reads `cfg["model"].get("cond_channels", 0)`.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_dit.py`:

```python


# ---- trajectory condition channels ----------------------------------------

def test_warm_started_conditional_model_matches_the_unconditional_one_exactly():
    from src.dit import load_unconditional_weights
    torch.manual_seed(0)
    base = TemporalDiT(seq_len=4, N=16, K=16, patch=8, stride=8, dim=32, depth=2,
                       heads=4, attn_mode="factorized")
    for p in base.parameters():
        torch.nn.init.normal_(p, std=0.02)
    cond = TemporalDiT(seq_len=4, N=16, K=16, patch=8, stride=8, dim=32, depth=2,
                       heads=4, attn_mode="factorized", cond_channels=4)
    load_unconditional_weights(cond, base.state_dict())
    base.eval()
    cond.eval()
    x, t = torch.randn(2, 4, 16, 16), torch.tensor([10, 700])
    with torch.no_grad():
        ref = base(x, t)
        assert torch.allclose(cond(x, t, cond_map=torch.rand(2, 4, 4, 16, 16)), ref, atol=1e-6)
        assert torch.allclose(cond(x, t), ref, atol=1e-6)


def test_condition_channels_reach_the_output_with_nonzero_weights():
    torch.manual_seed(0)
    m = TemporalDiT(seq_len=4, N=16, K=16, patch=8, stride=8, dim=32, depth=1,
                    heads=4, attn_mode="factorized", cond_channels=4)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.02)
    m.eval()
    x, t = torch.randn(1, 4, 16, 16), torch.tensor([100])
    with torch.no_grad():
        a = m(x, t, cond_map=torch.zeros(1, 4, 4, 16, 16))
        b = m(x, t, cond_map=torch.ones(1, 4, 4, 16, 16))
    assert (a - b).abs().max() > 1e-6


def test_unconditional_model_rejects_a_condition_map():
    import pytest
    m = TemporalDiT(seq_len=4, N=16, K=16, patch=8, stride=8, dim=32, depth=1, heads=4)
    with pytest.raises(ValueError):
        m(torch.randn(1, 4, 16, 16), torch.tensor([1]), cond_map=torch.zeros(1, 4, 4, 16, 16))
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES= /truenas/home/arigra/.venv/bin/python -m pytest -q tests/test_dit.py -k "warm_started or condition_channels or rejects_a_condition"`
Expected: 3 FAIL (`ImportError: cannot import name 'load_unconditional_weights'` / `TypeError: ... unexpected keyword argument 'cond_channels'`).

- [ ] **Step 3: Implement** in `src/dit.py`.

Change the constructor signature and projection:

```python
class TemporalDiT(nn.Module):
    def __init__(self, seq_len=16, N=64, K=64, patch=8, stride=4,
                 dim=256, depth=8, heads=8, attn_mode="temporal",
                 patch_reduction="mean", cond_channels=0):
        super().__init__()
        self.N, self.K, self.p, self.s = N, K, patch, stride
        self.cond_channels = cond_channels
        pr, pc = num_patches(N, K, patch, stride)
        P = pr * pc
        # each token: the noisy map's patch, then one patch per condition channel
        self.proj = nn.Linear(patch * patch * (1 + cond_channels), dim)
```

(the rest of `__init__` is unchanged). Replace the start of `forward`:

```python
    def forward(self, x, t, cond=None, cond_map=None):
        tokens = patchify(x, self.p, self.s)                  # (B, L, P, p*p)
        if self.cond_channels:
            B, L, N, K = x.shape
            if cond_map is None:
                cond_map = x.new_zeros(B, L, self.cond_channels, N, K)
            extra = [patchify(cond_map[:, :, c], self.p, self.s)
                     for c in range(self.cond_channels)]
            tokens = torch.cat([tokens] + extra, dim=-1)
        elif cond_map is not None:
            raise ValueError("cond_map given to a model built with cond_channels=0")
        z = self.proj(tokens) + self.spatial_pos + self.temporal_pos
```

(the rest of `forward` is unchanged). Append at the end of `src/dit.py`:

```python


def load_unconditional_weights(model, state_dict):
    """Warm-start a conditional model from an unconditional checkpoint.

    Every weight is copied; the input projection's map columns come from the
    checkpoint and its condition columns are zero, so the loaded model's
    output equals the unconditional model's for any condition map.
    """
    state = dict(state_dict)
    source = state["proj.weight"]
    target = model.proj.weight
    if source.shape != target.shape:
        expanded = torch.zeros_like(target)
        expanded[:, :source.shape[1]] = source.to(target.device)
        state["proj.weight"] = expanded
    model.load_state_dict(state)
```

In `src/train.py` `build_model`, pass the new argument:

```python
        model = TemporalDiT(
            seq_len=cfg["data"]["seq_len"], patch=m["patch"],
            stride=m["stride"], dim=m["dim"], depth=m["depth"],
            heads=m["heads"], attn_mode=m.get("attn_mode", "temporal"),
            patch_reduction=m.get("patch_reduction", "mean"),
            cond_channels=m.get("cond_channels", 0))
```

- [ ] **Step 4: Run to verify pass**

Run: `CUDA_VISIBLE_DEVICES= /truenas/home/arigra/.venv/bin/python -m pytest -q tests/test_dit.py tests/test_train_smoke.py`
Expected: all pass (existing tests included).

- [ ] **Step 5: Commit**

```bash
git add src/dit.py src/train.py tests/test_dit.py
git commit -m "feat(dit): condition channels with an exact warm start from unconditional weights

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 3: Training with trajectory conditioning

**Files:**
- Modify: `src/train.py` (imports, `_predict`, new `_trajectory_condition`, `_loss_components`, training loop, `init_from`)
- Test: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: `render_condition`, `drop_condition` (Task 1); `load_unconditional_weights`, `cond_channels` (Task 2).
- Produces: config keys `model.cond_channels`, `train.cond_dropout`, `train.init_from`; checkpoints whose `config` records them.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_train_smoke.py`:

```python


def test_trajectory_condition_map_is_built_only_for_conditional_models():
    from src.train import _trajectory_condition, build_model
    from src.simulator import generate_sequences
    cfg = yaml.safe_load(open("configs/base.yaml"))
    cfg["model"].update(dim=32, depth=1, heads=4, patch=8, stride=8, attn_mode="factorized")
    batch = generate_sequences(n=2, seed=0)
    plain = build_model(cfg, torch.device("cpu"))
    assert _trajectory_condition(plain, batch, torch.device("cpu"), 0.0) is None
    cfg["model"]["cond_channels"] = 4
    conditional = build_model(cfg, torch.device("cpu"))
    cond_map = _trajectory_condition(conditional, batch, torch.device("cpu"), 0.0)
    assert cond_map.shape == (2, 16, 4, 64, 64)


def test_init_from_warm_starts_a_conditional_model_exactly(tmp_path):
    torch.manual_seed(0)
    base_cfg = _tiny_config(tmp_path / "base")
    base_cfg["model"].update(patch=8, stride=8, attn_mode="factorized")
    generate_cache(base_cfg["data"]["cache_dir"], 4, 2, seq_len=16, seed=7, shard_size=4)
    train(base_cfg, device=torch.device("cpu"), max_steps=2)
    base = torch.load(tmp_path / "base" / "ckpt" / "last.pt", map_location="cpu")["model"]

    cfg = _tiny_config(tmp_path / "cond")
    cfg["data"]["cache_dir"] = base_cfg["data"]["cache_dir"]
    cfg["model"].update(patch=8, stride=8, attn_mode="factorized", cond_channels=4)
    # lr 0: weights cannot move, so the checkpoint shows exactly what init_from loaded
    cfg["train"].update(init_from=str(tmp_path / "base" / "ckpt" / "last.pt"),
                        cond_dropout=0.5, lr=0.0)
    losses = train(cfg, device=torch.device("cpu"), max_steps=2, _record_losses=True)
    assert all(torch.isfinite(torch.tensor(losses)))
    state = torch.load(tmp_path / "cond" / "ckpt" / "last.pt", map_location="cpu")["model"]
    assert state["proj.weight"].shape == (64, 64 * 5)
    assert torch.equal(state["proj.weight"][:, :64], base["proj.weight"])
    assert torch.count_nonzero(state["proj.weight"][:, 64:]) == 0
    assert all(torch.equal(state[k], base[k]) for k in base if k != "proj.weight")
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES= /truenas/home/arigra/.venv/bin/python -m pytest -q tests/test_train_smoke.py -k "condition_map_is_built or warm_starts"`
Expected: 2 FAIL — `ImportError: cannot import name '_trajectory_condition'`, and `init_from` is ignored so the loaded weights differ from the base checkpoint.

- [ ] **Step 3: Implement** in `src/train.py`.

Imports (replace the `src.dit` import line and add one):

```python
from src.dit import TemporalDiT, load_unconditional_weights
from src.ema import make_ema, update_ema
from src.losses import diffusion_loss, smooth_loss
from src.trajectory_condition import drop_condition, render_condition
```

Replace `_predict` and add `_trajectory_condition` right after it:

```python
def _predict(model, xt, t, cond, tr, device, cond_map=None):
    """Network forward, optionally under bf16 autocast (train.amp: bf16)."""
    amp = tr.get("amp", "off")
    if amp not in ("off", "bf16"):
        raise ValueError(f"unknown train.amp {amp!r}")
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp == "bf16"):
        if cond_map is None:
            prediction = model(xt, t, cond)
        else:
            prediction = model(xt, t, cond, cond_map=cond_map)
    return prediction.float()


def _trajectory_condition(model, batch, device, dropout_p):
    """Condition map for trajectory-conditioned models (cond_channels > 0), else None."""
    if not getattr(model, "cond_channels", 0):
        return None
    cond_map = render_condition(batch["traj"].to(device), batch["n_targets"].to(device),
                                batch["cls"].to(device))
    return drop_condition(cond_map, dropout_p)
```

In `_loss_components`, replace the prediction line:

```python
    cond_map = _trajectory_condition(model, batch, device, dropout_p)
    prediction = _predict(model, xt, t, cond, tr, device, cond_map)
```

In the training loop in `train`, replace the prediction line:

```python
            cond_map = _trajectory_condition(model, batch, device,
                                             tr.get("cond_dropout", 0.1))
            prediction = _predict(model, xt, t, cond, tr, device, cond_map)
```

In `train`, directly after the `if resume:` block (before `ema_decay = tr.get("ema_decay")`):

```python
    if tr.get("init_from") and resume_state is None:
        # warm start from an unconditional checkpoint; EMA weights if it has them
        source = torch.load(tr["init_from"], map_location="cpu")
        load_unconditional_weights(model, source.get("ema_model") or source["model"])
```

- [ ] **Step 4: Run to verify pass**

Run: `CUDA_VISIBLE_DEVICES= /truenas/home/arigra/.venv/bin/python -m pytest -q tests/test_train_smoke.py tests/test_sample.py tests/test_unet.py tests/test_ema.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/train.py tests/test_train_smoke.py
git commit -m "feat(train): trajectory conditioning with condition dropout and init_from warm start

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 4: Guided sampling from requested trajectories

**Files:**
- Modify: `src/sample.py` (add `generate_trajectory_conditioned` after `generate`)
- Test: `tests/test_sample.py`

**Interfaces:**
- Consumes: `render_condition` (Task 1); conditional checkpoints (Tasks 2-3); `diffusion_from_config`, `resolve_cache_dir`, `select_checkpoint_state` (existing).
- Produces: `generate_trajectory_conditioned(ckpt_path, labels, device, steps=30, guidance=2.0, weights="ema", seed=None) -> Tensor (B, L, 64, 64)` dB; `labels` = dict with `traj (B, M, L, 2)`, `n_targets (B,)`, `cls (B, M)`.

- [ ] **Step 1: Write the failing test** — append to `tests/test_sample.py`:

```python


def test_trajectory_conditioned_sampling_with_zero_guidance_is_unconditional(tmp_path):
    from src.sample import generate_trajectory_conditioned
    from src.simulator import generate_sequences
    torch.manual_seed(0)
    with open("configs/base.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["data"].update(cache_dir=str(tmp_path), n_train=4, n_val=2, shard_size=4)
    cfg["model"].update(dim=64, depth=2, heads=4, patch=8, stride=8,
                        attn_mode="factorized", cond_channels=4)
    cfg["train"].update(batch_size=2, epochs=1, ckpt_dir=str(tmp_path / "ckpt"),
                        log_file=str(tmp_path / "train.log"))
    generate_cache(str(tmp_path), 4, 2, seq_len=16, seed=7, shard_size=4)
    train(cfg, device=torch.device("cpu"), max_steps=2)
    ckpt = str(tmp_path / "ckpt" / "last.pt")
    labels = generate_sequences(n=2, seed=3)
    guided = generate_trajectory_conditioned(ckpt, labels, torch.device("cpu"), steps=3,
                                             guidance=0.0, weights="raw", seed=5)
    plain = generate(ckpt, n_seq=2, device=torch.device("cpu"), steps=3, weights="raw", seed=5)
    assert guided.shape == (2, 16, 64, 64) and torch.isfinite(guided).all()
    assert torch.allclose(guided, plain, atol=1e-3)
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES= /truenas/home/arigra/.venv/bin/python -m pytest -q tests/test_sample.py -k zero_guidance`
Expected: FAIL, `ImportError: cannot import name 'generate_trajectory_conditioned'`.

- [ ] **Step 3: Implement** — in `src/sample.py`, directly after `generate`:

```python
def generate_trajectory_conditioned(ckpt_path, labels, device, steps=30, guidance=2.0,
                                    weights="ema", seed=None):
    """Sample sequences that contain the requested targets.

    labels: dict with traj (B, M, L, 2) bins, n_targets (B,), cls (B, M).
    Classifier-free guidance on the network output:
    uncond + guidance * (cond - uncond), where uncond sees an all-zero
    condition map. Returns dB maps (B, L, 64, 64).
    """
    from src.trajectory_condition import render_condition

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model = build_model(cfg, device)
    model.load_state_dict(select_checkpoint_state(ckpt, weights=weights))
    model.eval()
    if not getattr(model, "cond_channels", 0):
        raise ValueError(f"{ckpt_path} is not a trajectory-conditioned checkpoint")
    diff = diffusion_from_config(cfg["diffusion"])
    cond_map = render_condition(labels["traj"].to(device), labels["n_targets"].to(device),
                                labels["cls"].to(device))
    null = torch.zeros_like(cond_map)

    def guided(xt, t, _cond=None):
        out = model(torch.cat([xt, xt]), torch.cat([t, t]), None,
                    cond_map=torch.cat([cond_map, null]))
        with_cond, without = out.chunk(2)
        return without + guidance * (with_cond - without)

    B, L = labels["traj"].shape[0], cfg["data"]["seq_len"]
    if seed is not None:
        torch.manual_seed(seed)
    x = diff.ddim_sample(guided, (B, L, 64, 64), device, steps=steps)
    stats = torch.load(resolve_cache_dir(cfg["data"]["cache_dir"]) / "stats.pt",
                       map_location="cpu")
    return denormalize(x.float().cpu(), stats)
```

- [ ] **Step 4: Run to verify pass**

Run: `CUDA_VISIBLE_DEVICES= /truenas/home/arigra/.venv/bin/python -m pytest -q tests/test_sample.py`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/sample.py tests/test_sample.py
git commit -m "feat(sample): classifier-free guided sampling from requested trajectories

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 5: Trajectory adherence metric

**Files:**
- Create: `src/eval/adherence.py`
- Test: `tests/test_trajectory_adherence.py` (note: `tests/test_adherence.py` already exists for another metric)

**Interfaces:**
- Consumes: `detect_peaks`, `link_tracks`, `filter_tracks` from `src/eval/metrics.py`.
- Produces: `trajectory_adherence(x_db, traj, n_targets, radius=2.0, max_peaks=5, min_track_len=8) -> {"hit_rate": float, "unrequested_lasting_tracks_per_seq": float}`.

Measured before planning: simulator sequences score hit rate 0.962 with own labels and 0.017 with labels shifted by 10 bins; single noisy target: 0.25 unrequested lasting tracks/seq with own labels, 1.25 with labels shifted by 20 (noise-free E0 is unusable here: sidelobes form 2 lasting tracks).

- [ ] **Step 1: Write the failing tests** — `tests/test_trajectory_adherence.py`:

```python
from src.eval.adherence import trajectory_adherence
from src.simulator import generate_sequences


def test_simulator_sequences_follow_their_own_labels():
    d = generate_sequences(n=8, seed=0)
    own = trajectory_adherence(d["x"], d["traj"], d["n_targets"])
    shifted = trajectory_adherence(d["x"], d["traj"] + 10.0, d["n_targets"])
    assert own["hit_rate"] > 0.8
    assert shifted["hit_rate"] < 0.2


def test_lasting_tracks_away_from_requests_count_as_unrequested():
    d = generate_sequences(n=8, n_targets=1, target_class="steady", snr_db=20,
                           clutter=False, noise=True, seed=0)
    own = trajectory_adherence(d["x"], d["traj"], d["n_targets"])
    far = trajectory_adherence(d["x"], d["traj"] + 20.0, d["n_targets"])
    assert own["unrequested_lasting_tracks_per_seq"] <= 0.5
    assert far["unrequested_lasting_tracks_per_seq"] >= 1.0
```

- [ ] **Step 2: Run to verify failure**

Run: `CUDA_VISIBLE_DEVICES= /truenas/home/arigra/.venv/bin/python -m pytest -q tests/test_trajectory_adherence.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'src.eval.adherence'`.

- [ ] **Step 3: Implement** — `src/eval/adherence.py`:

```python
"""Does a generated sequence contain the targets it was asked for?

Uses the evaluator's own detector and linker (src/eval/metrics): peaks at
least 12 dB above the frame median, the 5 strongest per frame, linked across
frames within 3 bins.
"""
import torch

from src.eval.metrics import detect_peaks, filter_tracks, link_tracks


def trajectory_adherence(x_db, traj, n_targets, radius=2.0, max_peaks=5, min_track_len=8):
    """x_db (B, L, N, K) dB maps; traj (B, M, L, 2) requested bins; n_targets (B,).

    hit_rate: fraction of requested (target, frame) pairs with a detected peak
    within `radius` bins.
    unrequested_lasting_tracks_per_seq: linked tracks of >= min_track_len frames
    whose peaks are within `radius` of a requested target on fewer than half
    of their frames.
    """
    hits = requested = unrequested = 0
    B, L = x_db.shape[:2]
    for i in range(B):
        targets = traj[i, :int(n_targets[i])].float()                    # (m, L, 2)
        peaks = [detect_peaks(x_db[i, l], max_peaks=max_peaks) for l in range(L)]
        for l in range(L):
            for m in range(len(targets)):
                requested += 1
                if len(peaks[l]) and float((peaks[l] - targets[m, l]).norm(dim=1).min()) <= radius:
                    hits += 1
        for track in filter_tracks(link_tracks(peaks), min_track_len):
            near = sum(1 for l, pos in track if len(targets) and
                       float((targets[:, l] - torch.tensor(pos)).norm(dim=1).min()) <= radius)
            unrequested += int(near < len(track) / 2)
    return {"hit_rate": hits / max(requested, 1),
            "unrequested_lasting_tracks_per_seq": unrequested / B}
```

- [ ] **Step 4: Run to verify pass**

Run: `CUDA_VISIBLE_DEVICES= /truenas/home/arigra/.venv/bin/python -m pytest -q tests/test_trajectory_adherence.py`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/eval/adherence.py tests/test_trajectory_adherence.py
git commit -m "feat(eval): trajectory hit rate and unrequested lasting tracks

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 6: Scorer, config, resumable pipeline, pre-run note, smoke test

**Files:**
- Create: `experiments/dit_64/score_conditional.py`, `experiments/dit_64/configs/cond_traj.yaml`, `experiments/dit_64/run_cond_traj.sh`, `experiments/dit_64/cond_traj.sbatch`, `docs/notes/2026-09-17-trajectory-conditioning.md`

**Interfaces:**
- Consumes: `generate_trajectory_conditioned` (Task 4), `trajectory_adherence` (Task 5), `generate`, `resolve_cache_dir` (existing), `checks` from `experiments/dit_64/score_hard_standard.py`, `evaluate_sequences`.
- Produces: `experiments/dit_64/results/cond_traj_scores.json`, `experiments/dit_64/results/cond_traj.png`, verdict appended to the note.

- [ ] **Step 1: Write `experiments/dit_64/score_conditional.py`**

```python
"""Score a trajectory-conditioned DiT against the pre-set rule; append the verdict.

Rule (docs/notes/2026-09-17-trajectory-conditioning.md): passes if hit rate >=
simulator hit rate - 0.05, unrequested lasting tracks/seq <= simulator's, and
Ari's four checks pass; the unconditional model on the same requests must hit
at least 0.20 less. Guidance is swept on val 0..3n-1 with seeds 1-3; the
verdict uses val 3n..9n-1 with seeds 4-9.
"""
import argparse
import json
from pathlib import Path

import torch

from score_hard_standard import checks
from src.dataset import RadarSequenceDataset
from src.eval.adherence import trajectory_adherence
from src.eval.metrics import evaluate_sequences
from src.sample import generate, generate_trajectory_conditioned, resolve_cache_dir


def labels_and_maps(items, offset, n):
    chunk = items[offset:offset + n]
    labels = {key: torch.stack([it[key] for it in chunk]) for key in ("traj", "n_targets", "cls")}
    return labels, torch.stack([it["x"] for it in chunk]).float()


def run_requests(ckpt, items, plan, n, device, guidance):
    """plan: list of (val offset, seed); returns concatenated labels, real maps, generated maps."""
    labels, real, gen = [], [], []
    for offset, seed in plan:
        lab, x = labels_and_maps(items, offset, n)
        labels.append(lab)
        real.append(x)
        gen.append(generate_trajectory_conditioned(ckpt, lab, device, steps=30,
                                                   guidance=guidance, weights="ema", seed=seed))
    labels = {key: torch.cat([lab[key] for lab in labels]) for key in labels[0]}
    return labels, torch.cat(real), torch.cat(gen)


def score(gen, real, labels, real_ref, real_metrics, stats):
    adh = trajectory_adherence(gen, labels["traj"], labels["n_targets"])
    sim = trajectory_adherence(real, labels["traj"], labels["n_targets"])
    m = evaluate_sequences(gen, real_ref)
    m["std"] = float(((gen - stats["mean"]) / stats["std"]).std())
    rule = {"hit_rate": adh["hit_rate"] >= sim["hit_rate"] - 0.05,
            "unrequested_tracks": adh["unrequested_lasting_tracks_per_seq"]
                                  <= sim["unrequested_lasting_tracks_per_seq"],
            **checks(m, real_metrics)}
    return {"adherence": adh, "simulator": sim, "metrics": m, "rule": rule,
            "passes": all(rule.values())}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--uncond-ckpt", required=True)
    ap.add_argument("--guidance", default="1,2,3")
    ap.add_argument("--n", type=int, default=32, help="sequences per seed")
    ap.add_argument("--out", default="experiments/dit_64/results/cond_traj_scores.json")
    ap.add_argument("--note")
    ap.add_argument("--png")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = args.n
    cfg = torch.load(args.ckpt, map_location="cpu", weights_only=False)["config"]
    cache = resolve_cache_dir(cfg["data"]["cache_dir"])
    items = RadarSequenceDataset(cache, "val").items
    stats = torch.load(cache / "stats.pt", map_location="cpu")
    _, ref_a = labels_and_maps(items, 0, 32)
    _, ref_b = labels_and_maps(items, 32, 32)
    real_metrics = evaluate_sequences(ref_a, ref_b)
    real_metrics["std"] = float(((ref_b - stats["mean"]) / stats["std"]).std())
    report = {"checkpoint": args.ckpt, "n_per_seed": n, "real_vs_real": real_metrics, "sweep": {}}

    sweep_plan = [((s - 1) * n, s) for s in (1, 2, 3)]
    best = None
    for w in (float(v) for v in args.guidance.split(",")):
        labels, real, gen = run_requests(args.ckpt, items, sweep_plan, n, device, w)
        result = score(gen, real, labels, ref_b, real_metrics, stats)
        report["sweep"][str(w)] = result
        print(f"w={w}", json.dumps({k: result[k] for k in ("adherence", "simulator", "rule", "passes")}),
              flush=True)
        key = (result["passes"], result["adherence"]["hit_rate"])
        if best is None or key > best[0]:
            best = (key, w)
    w = best[1]

    final_plan = [(3 * n + (s - 4) * n, s) for s in range(4, 10)]
    labels, real, gen = run_requests(args.ckpt, items, final_plan, n, device, w)
    final = score(gen, real, labels, ref_b, real_metrics, stats)
    null_gen = torch.cat([generate(args.uncond_ckpt, n, device, steps=30, weights="ema", seed=s)
                          for _, s in final_plan])
    null = trajectory_adherence(null_gen, labels["traj"], labels["n_targets"])
    final.update(guidance=w, null_unconditional=null,
                 null_check_ok=null["hit_rate"] <= final["adherence"]["hit_rate"] - 0.20)
    report["final"] = final
    print("FINAL", json.dumps({k: final[k] for k in ("guidance", "adherence", "simulator", "rule",
                                                     "passes", "null_unconditional", "null_check_ok")}),
          flush=True)
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))

    if args.png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.viz import show_rows
        show_rows([{"x": real, **labels}, {"x": gen, **labels}],
                  ["simulator (requested scene)", f"conditional DiT, w={w}"],
                  title="Requested targets (red circles): simulator vs conditional DiT")
        plt.savefig(args.png, dpi=70)

    if args.note:
        ok = lambda b: "ok" if b else "FAIL"
        s, a, m, rr, rule = (final["simulator"], final["adherence"], final["metrics"],
                             real_metrics, final["rule"])
        sweep = ", ".join(f"w={k}: hit {v['adherence']['hit_rate']:.3f}, passes {v['passes']}"
                          for k, v in report["sweep"].items())
        lines = [
            "", f"## Result (`{args.out}`)", "",
            f"Checkpoint `{args.ckpt}`. Guidance w = {w}, chosen on val 0-{3 * n - 1} / seeds 1-3 "
            f"({sweep}). Verdict on val {3 * n}-{9 * n - 1}, seeds 4-9, {6 * n} sequences.",
            "", "| check | simulator / real | conditional DiT | |", "|---|---:|---:|---|",
            f"| hit rate | {s['hit_rate']:.3f} | {a['hit_rate']:.3f} | {ok(rule['hit_rate'])} |",
            f"| unrequested lasting tracks / seq | {s['unrequested_lasting_tracks_per_seq']:.2f} | "
            f"{a['unrequested_lasting_tracks_per_seq']:.2f} | {ok(rule['unrequested_tracks'])} |",
            f"| std | {rr['std']:.3f} | {m['std']:.3f} | {ok(rule['std'])} |",
            f"| marginal L1 | {rr['marginal_l1']:.3f} | {m['marginal_l1']:.3f} | {ok(rule['marginal_l1'])} |",
            f"| target tracks / seq | {rr['n_target_tracks_per_seq']:.2f} | "
            f"{m['n_target_tracks_per_seq']:.2f} | {ok(rule['target_tracks'])} |",
            f"| persistence | {rr['persistence']:.3f} | {m['persistence']:.3f} | {ok(rule['persistence'])} |",
            "", f"Null check (unconditional e3_long, same requests and seeds): hit rate "
            f"{final['null_unconditional']['hit_rate']:.3f} -> "
            + ("ok" if final["null_check_ok"] else "FAILED: the test is too easy; do not report the result"),
            "", "**Verdict (pre-set rule): "
            + ("the conditional DiT follows requested trajectories.**"
               if final["passes"] and final["null_check_ok"] else
               "the conditional DiT does NOT yet pass; see the failing checks.**"),
        ]
        with open(args.note, "a") as fh:
            fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `experiments/dit_64/configs/cond_traj.yaml`**

```yaml
# Trajectory-conditioned DiT (research plan step 1). Warm-starts from the passing
# unconditional full-data model. Design: docs/design/specs/2026-09-17-trajectory-conditioning-design.md
# Rule: docs/notes/2026-09-17-trajectory-conditioning.md. Run by experiments/dit_64/run_cond_traj.sh.
data:
  n_train: 20000
  n_val: 2000
  seq_len: 16
  frame_interval: 0.5
  cache_dir: data/cache
  shard_size: 1000
  seed: 1234
model:
  patch: 8
  stride: 8
  dim: 384
  depth: 12
  heads: 6
  attn_mode: factorized
  cond_channels: 4           # 3 class blob channels + presence
diffusion:
  timesteps: 1000
  parameterization: v
  schedule_shift: 4.0
  x0_clamp: "off"
  terminal_x0: model
train:
  seed: 2026
  batch_size: 32
  lr: 1.0e-4
  weight_decay: 0.0
  epochs: 40                 # 25,000 steps
  lambda_smooth: 0.0
  lambda_traj: 0.01
  lambda_doppler: 0.01
  amp: bf16
  ema_decay: 0.9999
  phase: 1
  init_from: checkpoints/e3_long_bs32/last.pt
  cond_dropout: 0.1
  ckpt_dir: checkpoints/cond_traj
  log_file: experiments/dit_64/logs/cond_traj.log
  log_every_steps: 100
  save_every_steps: 1000
  snapshot_every_epochs: 10
  val_every_epochs: 5
  val_batch_size: 32
  val_seed: 4321
  wandb: false
sample:
  ddim_steps: 30
  weights: ema
```

- [ ] **Step 3: Write `experiments/dit_64/run_cond_traj.sh` and `experiments/dit_64/cond_traj.sbatch`**

`experiments/dit_64/run_cond_traj.sh`:

```bash
#!/usr/bin/env bash
# Unattended trajectory-conditioned DiT run (research plan step 1).
# Safe to rerun: training resumes from the newest checkpoint (last.pt, written
# atomically every 1000 steps and at each epoch end); a finished run skips to scoring.
# Progress: experiments/dit_64/logs/cond_traj_pipeline.log   Rule/verdict: docs/notes/2026-09-17-trajectory-conditioning.md
set -u
cd "$(dirname "$0")/.."
export PYTHONPATH=.:scripts
PY=/truenas/home/arigra/.venv/bin/python
LOG=experiments/dit_64/logs/cond_traj_pipeline.log
NOTE=docs/notes/2026-09-17-trajectory-conditioning.md
say() { echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

if pgrep -f "src.train --config experiments/dit_64/configs/cond_traj_bs" > /dev/null; then
  say "REFUSED: a cond_traj training process is already running"; exit 1
fi
say "start"
if ! CUDA_VISIBLE_DEVICES= $PY -m pytest -q tests/test_trajectory_condition.py tests/test_dit.py \
     tests/test_trajectory_adherence.py >> "$LOG" 2>&1; then
  say "FAILED: unit tests"; exit 1
fi

CKPT=""
for BS in 32 16 8; do
  CFG=experiments/dit_64/configs/cond_traj_bs$BS.yaml
  sed -e "s/^  batch_size: .*/  batch_size: $BS/" \
      -e "s#checkpoints/cond_traj\$#checkpoints/cond_traj_bs$BS#" \
      -e "s#experiments/dit_64/logs/cond_traj.log#experiments/dit_64/logs/cond_traj_bs$BS.log#" \
      experiments/dit_64/configs/cond_traj.yaml > "$CFG"
  RESUME=()
  if [ -f "checkpoints/cond_traj_bs$BS/last.pt" ]; then
    RESUME=(--resume "checkpoints/cond_traj_bs$BS/last.pt")
  fi
  say "training batch=$BS ($CFG) ${RESUME[*]}"
  if $PY -m src.train --config "$CFG" "${RESUME[@]}" >> "experiments/dit_64/logs/cond_traj_bs$BS.console.log" 2>&1; then
    CKPT=checkpoints/cond_traj_bs$BS/last.pt; break
  fi
  if tail -50 "experiments/dit_64/logs/cond_traj_bs$BS.console.log" | grep -qiE "out of memory|OutOfMemoryError"; then
    say "OOM at batch=$BS, retrying smaller"; continue
  fi
  say "FAILED: training crashed, see experiments/dit_64/logs/cond_traj_bs$BS.console.log (rerun this script to resume)"; exit 1
done
[ -z "$CKPT" ] && { say "FAILED: OOM at every batch size"; exit 1; }

say "scoring $CKPT"
if ! $PY experiments/dit_64/score_conditional.py --ckpt "$CKPT" --uncond-ckpt checkpoints/e3_long_bs32/last.pt \
     --out experiments/dit_64/results/cond_traj_scores.json --note "$NOTE" --png experiments/dit_64/results/cond_traj.png >> "$LOG" 2>&1; then
  say "FAILED: scoring (rerun this script; training will be skipped)"; exit 1
fi
say "DONE: verdict appended to $NOTE"
```

`experiments/dit_64/cond_traj.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=radseq_cond_traj
#SBATCH --partition=shared_3090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=56000M
#SBATCH --time=10:00:00
#SBATCH --output=/truenas/home/arigra/jobs/radseq_cond_traj_%j.out
#SBATCH --error=/truenas/home/arigra/jobs/radseq_cond_traj_%j.err
# Continue (or start) the trajectory-conditioned run from its last checkpoint.
# Submit from the login host ece-hpc:  sbatch experiments/dit_64/cond_traj.sbatch
cd /truenas/home/arigra/permuter/ariGranevich/radSeq
bash experiments/dit_64/run_cond_traj.sh
```

Then: `chmod +x experiments/dit_64/run_cond_traj.sh experiments/dit_64/cond_traj.sbatch && bash -n experiments/dit_64/run_cond_traj.sh && bash -n experiments/dit_64/cond_traj.sbatch`. Expected: no output.

- [ ] **Step 4: Write the pre-run note `docs/notes/2026-09-17-trajectory-conditioning.md`**

```markdown
# Trajectory-conditioned DiT (2026-09-17)

Written before the run. Research plan step 1: the DiT generates the targets it
is asked for, so synthetic sequences come with exact labels.
Design: `docs/design/specs/2026-09-17-trajectory-conditioning-design.md`.

## Setup

`experiments/dit_64/configs/cond_traj.yaml`: the passing full-data recipe (`e3_long`) plus
`model.cond_channels: 4` (3 class blob channels + presence), warm-started from
`checkpoints/e3_long_bs32/last.pt` (EMA weights; new input weights zero, so
training starts from exactly the unconditional model), condition dropout 0.1,
40 epochs (25,000 steps).

## Decision rule (Ari: trajectories as the control, pass bar relative to the simulator)

On requests taken from held-out validation labels, EMA weights, DDIM 30:

- hit rate (requested target-frames with a detected peak within 2 bins)
  >= simulator hit rate on the same labels - 0.05;
- unrequested lasting tracks per sequence (tracks of >= 8 frames not near a
  requested target on most frames) <= the simulator's;
- Ari's four checks pass (std within 0.1, marginal L1 <= 0.15, target
  tracks/seq within 15%, persistence within 25% of real);
- null check: the unconditional e3_long model on the same requests and seeds
  hits at least 0.20 less, else the test is too easy and no result is reported.

Guidance w in {1, 2, 3} is chosen on val 0-95 / seeds 1-3; the verdict uses
val 96-287 / seeds 4-9 (192 sequences).

Known before the run: the simulator hits 0.96 of its own labelled
target-frames, so the bar is about 0.91. It has ~0.4 unrequested lasting
tracks/seq (8-sequence estimate), so with 192 sequences that comparison has a
sampling error of roughly +/-0.05; a miss by less than that should be read as
a tie, not a failure of the method.

Not measured: whether generated targets look like their requested class.

## If the session closes

Rerun `setsid nohup bash experiments/dit_64/run_cond_traj.sh > experiments/dit_64/logs/cond_traj_pipeline.console.log 2>&1 < /dev/null &`
in a new session, or `sbatch experiments/dit_64/cond_traj.sbatch` from `ece-hpc`. Both
resume from `checkpoints/cond_traj_bs32/last.pt`.
```

- [ ] **Step 5: Smoke test (train, resume, score) in the scratchpad**

```bash
S=/tmp/claude-343917844/-truenas-home-arigra/062bb7b8-daf4-412b-9547-91d7dbb4b6ea/scratchpad/cond_smoke
rm -rf $S && mkdir -p $S && export PYTHONPATH=.:scripts && PY=/truenas/home/arigra/.venv/bin/python
$PY -c "
import yaml, torch
from src.train import train
cfg = yaml.safe_load(open('experiments/dit_64/configs/cond_traj.yaml'))
cfg['train'].update(ckpt_dir='$S', log_file='$S/train.log', log_every_steps=20, save_every_steps=20, val_every_epochs=None)
train(cfg, device=torch.device('cuda'), max_steps=40)
train(cfg, device=torch.device('cuda'), max_steps=60, resume='$S/last.pt')
print('step', torch.load('$S/last.pt', map_location='cpu', weights_only=False)['step'], 'peak GB', round(torch.cuda.max_memory_allocated()/1e9, 2))
"
$PY experiments/dit_64/score_conditional.py --ckpt $S/last.pt --uncond-ckpt checkpoints/e3_long_bs32/last.pt \
    --n 2 --guidance 1 --out $S/scores.json --note $S/note.md --png $S/p.png && tail -4 $S/note.md
```

Expected: `step 60`, peak memory below ~13 GB, the scorer prints a `w=1.0` line and a `FINAL` line, and the note ends with a verdict line.

- [ ] **Step 6: Commit**

```bash
git add experiments/dit_64/score_conditional.py experiments/dit_64/configs/cond_traj.yaml experiments/dit_64/run_cond_traj.sh experiments/dit_64/cond_traj.sbatch docs/notes/2026-09-17-trajectory-conditioning.md
git commit -m "exp: scorer, config and resumable pipeline for the trajectory-conditioned DiT

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>"
```

---

### Task 7: Launch and report

- [ ] **Step 1: Check the session and GPU, then launch detached**

```bash
date; date -d @$SLURM_JOB_END_TIME; nvidia-smi --query-gpu=memory.free --format=csv,noheader
setsid nohup bash experiments/dit_64/run_cond_traj.sh > experiments/dit_64/logs/cond_traj_pipeline.console.log 2>&1 < /dev/null & disown
sleep 90; tail -3 experiments/dit_64/logs/cond_traj_pipeline.log; tail -2 experiments/dit_64/logs/cond_traj_bs32.log
```

Expected: at least ~3 h left in the session (else use `sbatch`), `training batch=32`, a `start device=cuda` line with `epoch=0 step=0`, and step logs.

- [ ] **Step 2: When the pipeline log ends in DONE or FAILED, report the verdict table from the note to Ari.**
