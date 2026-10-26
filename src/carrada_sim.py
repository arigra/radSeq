"""CARRADA-style range-Doppler scenes from radar physics (engineer's variant).

Spec: docs/design/2026-10-26-carrada-simulator-design.md (approved by Ari,
2026-10-26). Built from published parameters and the dataset paper's scene
description; nothing is fitted to the recordings.

- Radar: 77 GHz FMCW, TDM MIMO 2 Tx x 4 Rx = 8 virtual channels, static on a
  parked car. Grid as the dataset: 256 range bins of 0.2 m (row 0 near), 64
  Doppler bins of 0.42 m/s, zero at bin 32, approaching objects above it;
  velocities past +/-13.4 m/s wrap. RD map = power summed over the 8 channels.
- Link budget: Tx 12 dBm, noise figure 14.5 dB (AWR1843 datasheet), antennas
  8 dBi each (TI estimator default, flagged), observation time from the
  published v_max and 64 chirps.
- Scene: flat test track (diffuse ground, a few poles), 0-2 moving objects on
  straight paths. Car = the self-shadowing outline of the RADIal simulator.
  Pedestrian = torso + four swinging limbs (Deep et al., arXiv 1910.13706:
  -10..+5 dBsm, limbs to ~3.3x body speed). Cyclist = rider (pedestrian model)
  + frame + two rolling wheels (rim 0..2x bike speed); its RCS is an assumption.
Assumptions not in any source, all flagged in the spec: radar height 1 m,
elevation beam 40 deg, Hamming windows, 3 dB system loss, no receiver filters.
"""
import dataclasses
import math

import torch

from src.radar_physics import window

C_LIGHT = 299_792_458.0
K_T0_DBW_HZ = -204.0


@dataclasses.dataclass(frozen=True)
class CarradaSpec:
    n_range: int = 256
    n_doppler: int = 64
    range_res_m: float = 0.2
    velocity_res_mps: float = 0.42
    zero_doppler: int = 32
    n_virtual: int = 8
    fc_hz: float = 77e9
    frame_interval_s: float = 0.1
    v_max_mps: float = 13.43
    tx_dbm: float = 12.0
    antenna_gain_dbi: float = 8.0
    noise_figure_db: float = 14.5
    system_loss_db: float = 3.0
    adc_duty: float = 0.8

    @property
    def wavelength(self):
        return C_LIGHT / self.fc_hz

    @property
    def max_range_m(self):
        return self.n_range * self.range_res_m

    @property
    def obs_time_s(self):
        """Per virtual channel: 64 chirps of one Tx; the loop of 2 chirps lasts
        lambda / (4 v_max) (TDM with 2 Tx)."""
        return self.n_doppler * (self.wavelength / (4 * self.v_max_mps) / 2) * self.adc_duty


@dataclasses.dataclass(frozen=True)
class TrackScenario:
    radar_height_m: float = 1.0
    elevation_beam_deg: float = 40.0
    azimuth_pattern_power: float = 2.0       # two-way cos^2 element pattern
    ground_scatterers: int = 2500
    ground_rcs_dbsm: tuple = (-25.0, 6.0)
    poles_mean: float = 2.0
    pole_rcs_dbsm: tuple = (5.0, 4.0)
    p_empty: float = 1 - 7193 / 12666        # paper Table II: frames with instances
    speeds_mps: tuple = ((0.5, 2.0), (3.0, 7.0), (3.0, 10.0))   # pedestrian, cyclist, car
    car_rcs_dbsm: float = 10.0
    person_rcs_dbsm: tuple = (-10.0, 5.0)    # Deep et al., over a walking stride
    gait_hz: float = 1.8
    limb_swing: float = 2.3                  # limb speed reaches (1 + 2.3) x body
    wheel_radius_m: float = 0.35
    wheel_rcs_dbsm: float = -15.0            # assumption, flagged
    gear_ratio: float = 1.5                  # wheel turns per pedal turn


def _power_db(rcs_dbsm, range_m, spec):
    """Radar equation, SNR per RD cell and virtual channel (noise = 0 dB)."""
    r = range_m.clamp(min=spec.range_res_m)
    lam = spec.wavelength
    return (spec.tx_dbm - 30 + 2 * spec.antenna_gain_dbi + 20 * math.log10(lam) + rcs_dbsm
            + 10 * math.log10(spec.obs_time_s) - 30 * math.log10(4 * math.pi)
            - 40 * torch.log10(r) - K_T0_DBW_HZ - spec.noise_figure_db - spec.system_loss_db)


class CarradaSimulator:
    def __init__(self, spec=CarradaSpec(), scenario=TrackScenario(), seq_len=8,
                 device=None, generator=None):
        self.spec, self.sc, self.L = spec, scenario, seq_len
        self.device = device or torch.device("cpu")
        self.g = generator
        self.win_r = window(spec.n_range).to(self.device)
        self.win_d = window(spec.n_doppler).to(self.device)
        self.n_idx = torch.arange(spec.n_range, device=self.device, dtype=torch.float)
        self.k_idx = torch.arange(spec.n_doppler, device=self.device, dtype=torch.float)

    def _u(self, lo, hi, size=()):
        return torch.rand(size, generator=self.g) * (hi - lo) + lo

    def _n(self, mean, std, size=()):
        return torch.randn(size, generator=self.g) * std + mean

    # ----------------------------------------------------------- objects
    def _object(self, cls, oid):
        """A moving object: class (1 pedestrian, 2 cyclist, 3 car), start, velocity."""
        sc = self.sc
        speed = float(self._u(*sc.speeds_mps[cls - 1]))
        r0, az0 = float(self._u(3.0, 45.0)), math.radians(float(self._u(-60.0, 60.0)))
        x0, y0 = r0 * math.sin(az0), r0 * math.cos(az0)
        kind = int(torch.randint(3, (), generator=self.g))       # approach, recede, cross
        los = math.atan2(x0, y0)
        heading = {0: los + math.pi, 1: los, 2: los + math.pi / 2 * (1 if self._u(0, 1) < 0.5 else -1)}[kind]
        heading += math.radians(float(self._n(0.0, 10.0)))
        o = {"cls": cls, "id": oid, "x0": x0, "y0": y0, "vx": speed * math.sin(heading),
             "vy": speed * math.cos(heading), "speed": speed, "phase": float(self._u(0, 2 * math.pi))}
        if cls == 3:
            o["car"] = self._car_outline(heading)
        return o

    def _car_outline(self, heading, length=4.5, width=1.8):
        """Scatterers along the footprint outline with their faces' outward normals
        (the RADIal model), rotated to the heading (y = forward)."""
        hl, hw = length / 2, width / 2
        faces = ((-hw, -hl, hw, -hl, 0.0, -1.0), (-hw, hl, hw, hl, 0.0, 1.0),
                 (-hw, -hl, -hw, hl, -1.0, 0.0), (hw, -hl, hw, hl, 1.0, 0.0))
        pts = []
        for x0, y0, x1, y1, fx, fy in faces:
            n = max(round(math.hypot(x1 - x0, y1 - y0) * 4.0), 1)
            for s in ((torch.arange(n, dtype=torch.float) + 0.5) / n).tolist():
                pts.append((x0 + (x1 - x0) * s, y0 + (y1 - y0) * s, fx, fy))
        p = torch.tensor(pts)
        c, s = math.cos(heading), math.sin(heading)
        rot = lambda a, b: (a * c + b * s, -a * s + b * c)       # body (x right, y fwd) -> world
        ox, oy = rot(p[:, 0], p[:, 1])
        nx, ny = rot(p[:, 2], p[:, 3])
        return {"ox": ox, "oy": oy, "nx": nx, "ny": ny, "scint": self._n(0.0, 3.0, (len(pts),))}

    def _returns(self, o, t):
        """Scatterers of object o at time t: x, y, velocity along the line of sight
        towards the radar (m/s), RCS (dBsm)."""
        sc = self.sc
        cx, cy = o["x0"] + o["vx"] * t, o["y0"] + o["vy"] * t
        ux, uy = o["vx"] / o["speed"], o["vy"] / o["speed"]
        if o["cls"] == 3:                                          # car: self-shadowing faces
            car = o["car"]
            px, py = cx + car["ox"], cy + car["oy"]
            r = torch.sqrt(px ** 2 + py ** 2).clamp(min=1e-3)
            cos = -(car["nx"] * px + car["ny"] * py) / r
            vis = cos > 0
            if not bool(vis.any()):
                return None
            w = cos[vis] ** 2
            rcs = sc.car_rcs_dbsm + 10 * torch.log10(w / w.sum()) + car["scint"][vis]
            vx, vy = torch.full_like(w, o["vx"]), torch.full_like(w, o["vy"])
            px, py = px[vis], py[vis]
        else:                                                      # person (and rider)
            total = float(self._u(*sc.person_rcs_dbsm))
            f = sc.gait_hz if o["cls"] == 1 else o["speed"] / (2 * math.pi * sc.wheel_radius_m) / sc.gear_ratio
            ph = 2 * math.pi * f * t + o["phase"]
            swing = torch.tensor([0.0, math.sin(ph), -math.sin(ph), -math.sin(ph), math.sin(ph)])
            amp = sc.limb_swing if o["cls"] == 1 else 0.5 * sc.limb_swing   # pedalling legs swing less
            speed = o["speed"] * (1 + amp * swing)
            along = torch.tensor([0.0, 0.4, -0.4, -0.4, 0.4]) * math.sin(ph)  # micro-range ~ +/-0.4 m
            share = torch.tensor([0.7, 0.075, 0.075, 0.075, 0.075])
            px, py = cx + along * ux, cy + along * uy
            vx, vy = speed * ux, speed * uy
            rcs = total + 10 * torch.log10(share)
            if o["cls"] == 2:                                      # bicycle frame + wheels
                fr = torch.tensor([-0.6, 0.0, 0.6])
                th = torch.arange(8, dtype=torch.float) / 8 * 2 * math.pi + o["speed"] * t / sc.wheel_radius_m
                wheel_along = torch.cat([sc.wheel_radius_m * torch.sin(th) + d for d in (-0.55, 0.55)])
                wheel_speed = o["speed"] * (1 - torch.cos(th)).repeat(2)   # rim: 0 at contact, 2v on top
                px = torch.cat([px, cx + fr * ux, cx + wheel_along * ux])
                py = torch.cat([py, cy + fr * uy, cy + wheel_along * uy])
                sp = torch.cat([torch.full((3,), o["speed"]), wheel_speed])
                vx, vy = torch.cat([vx, sp * ux]), torch.cat([vy, sp * uy])
                rcs = torch.cat([rcs, torch.full((3,), total - 10 * math.log10(3)),
                                 torch.full((16,), sc.wheel_rcs_dbsm - 10 * math.log10(16))])
        r = torch.sqrt(px ** 2 + py ** 2).clamp(min=1e-3)
        v_app = -(vx * px + vy * py) / r                         # towards the radar = positive
        return px, py, v_app, rcs

    # ----------------------------------------------------------- rendering
    def _render(self, x, y, v_app, rcs):
        spec, sc = self.spec, self.sc
        rng = torch.sqrt(x ** 2 + y ** 2)
        az = torch.atan2(x, y)
        keep = (rng > 0.1) & (rng < spec.max_range_m) & (az.abs() < math.pi / 2)
        rng, az, v_app, rcs = rng[keep], az[keep], v_app[keep], rcs[keep]
        dep = torch.rad2deg(torch.atan2(torch.tensor(sc.radar_height_m), rng))
        p = (_power_db(rcs, rng, spec) - 24.0 * (dep / sc.elevation_beam_deg) ** 2
             + 10 * sc.azimuth_pattern_power * torch.log10(torch.cos(az).abs().clamp(min=1e-3)))
        rb = (rng / spec.range_res_m).to(self.device)
        db = (v_app / spec.velocity_res_mps).to(self.device)
        amp = (10 ** (p / 20)).to(self.device)
        phase0 = (torch.rand(len(rb), generator=self.g) * 2 * math.pi).to(self.device)
        rt = torch.exp(2j * math.pi * torch.outer(self.n_idx, rb) / spec.n_range)
        dt = torch.exp(2j * math.pi * torch.outer(db, self.k_idx) / spec.n_doppler)
        win = torch.outer(self.win_r, self.win_d)
        norm = float(self.win_r.sum() * self.win_d.sum())
        sin_az = torch.sin(az).to(self.device)
        power = torch.zeros(spec.n_range, spec.n_doppler, device=self.device)
        for m in range(spec.n_virtual):
            a = amp * torch.exp(1j * (phase0 + math.pi * m * sin_az))
            rd = torch.fft.fft2((rt * a.unsqueeze(0)) @ dt * win, dim=(0, 1)) / norm
            power += rd.abs() ** 2
        return torch.roll(power, spec.zero_doppler, dims=1)      # zero velocity -> bin 32

    def _noise(self):
        if not hasattr(self, "_dg"):
            seed = int(torch.randint(2 ** 62, (1,), generator=self.g))
            self._dg = torch.Generator(device=self.device).manual_seed(seed)
        z = torch.randn(self.spec.n_range, self.spec.n_doppler, 2 * self.spec.n_virtual,
                        generator=self._dg, device=self.device)
        return (z ** 2).sum(-1) / 2

    def gen_sequence(self):
        spec, sc = self.spec, self.sc
        # static scene: ground and poles, at zero Doppler (the radar does not move)
        m = sc.ground_scatterers
        gx, gy = self._u(-30.0, 30.0, (m,)), self._u(0.5, spec.max_range_m, (m,))
        grcs = self._n(*sc.ground_rcs_dbsm, (m,))
        n_poles = int(torch.poisson(torch.tensor(sc.poles_mean), generator=self.g))
        gx = torch.cat([gx, self._u(-20.0, 20.0, (n_poles,))])
        gy = torch.cat([gy, self._u(2.0, spec.max_range_m - 2, (n_poles,))])
        grcs = torch.cat([grcs, self._n(*sc.pole_rcs_dbsm, (n_poles,))])
        empty = bool(self._u(0, 1) < sc.p_empty)
        n_obj = 0 if empty else 1 + int(self._u(0, 1) < 0.5)
        objects = [self._object(1 + int(torch.randint(3, (), generator=self.g)), i) for i in range(n_obj)]
        frames, labels = [], []
        for l in range(self.L):
            t = l * spec.frame_interval_s
            xs, ys, vs, rs, fl = [gx], [gy], [torch.zeros_like(gx)], [grcs], []
            for o in objects:
                ret = self._returns(o, t)
                if ret is None:
                    continue
                px, py, v_app, rcs = ret
                xs.append(px); ys.append(py); vs.append(v_app); rs.append(rcs)
                rng = torch.sqrt(px ** 2 + py ** 2)
                az = torch.atan2(px, py)
                if not (0.5 < float(rng.min()) and float(rng.max()) < spec.max_range_m
                        and float(az.abs().max()) < math.radians(80)):
                    continue                                       # out of view: no label
                w = 10 ** (rcs / 10)
                r_c = float((w * rng).sum() / w.sum())
                # body velocity, as CARRADA boxes centre on the main return
                bx, by = o["x0"] + o["vx"] * t, o["y0"] + o["vy"] * t
                v_body = -(o["vx"] * bx + o["vy"] * by) / math.hypot(bx, by)
                fl.append({"id": o["id"], "cls": o["cls"], "range_bin": r_c / spec.range_res_m,
                           "doppler_bin": (spec.zero_doppler + v_body / spec.velocity_res_mps) % spec.n_doppler})
            power = self._render(torch.cat(xs), torch.cat(ys), torch.cat(vs), torch.cat(rs))
            frames.append(10 * torch.log10(power + self._noise() + 1e-12).cpu())
            labels.append(fl)
        return {"x": torch.stack(frames), "labels": labels,
                "classes": [o["cls"] for o in objects]}
