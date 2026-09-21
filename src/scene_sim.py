"""RADIal-style range-Doppler scenes from radar physics and scene geometry.

The earlier simulator (src/simulator.py) draws a few point targets over a
K-distributed clutter ridge on a 64x64 grid. Real RADIal maps look nothing like
that, because they are shaped by mechanisms it does not have:

1. DDMA MIMO. All 12 transmitters fire together, each with its own Doppler
   phase code, so every return appears once per transmitter, spread across the
   Doppler axis in 16-bin slots with a 4-slot gap (valeoai/RADIal
   SignalProcessing/rpl.py: numReducedDoppler=16, Tx offsets 0 and 80..240).
2. A static scene seen from a moving car. Stationary scatterers at azimuth
   theta have Doppler -v_ego*cos(theta), so the scene fills a Doppler band that
   ends sharply at -v_ego.
3. Sixteen receive antennas. Each sees the scene with angle-dependent phases,
   so summing their power averages the speckle.
4. The receiver's filters: an IF high-pass that darkens near range and an
   anti-alias low-pass that rolls off the far range.
5. Vehicles as extended objects with several scatterers.

Every parameter is published (rpl.py, Rebut et al. CVPR 2022 Table 5) or is a
statement about the driving scenario. Nothing is fitted to the recordings.
"""
import dataclasses
import math

import torch

from src.radar_physics import RADIAL_SPEC, C_LIGHT, window

# rpl.py: dividend_constant_arr = arange(0, 16*16, 16); the Tx sequence keeps
# element 0 and elements 5..15, i.e. slots 1-4 are empty.
DDMA_SLOT = 16
DDMA_OFFSETS = (0,) + tuple(range(5 * DDMA_SLOT, 16 * DDMA_SLOT, DDMA_SLOT))


@dataclasses.dataclass(frozen=True)
class Scenario:
    """What is known about the driving scene. Scenario knowledge, not fitted."""

    ego_speed_mps: tuple = (0.0, 15.0)          # urban/suburban driving
    radar_height_m: float = 0.8                 # Table 5
    elevation_beam_deg: float = 12.0            # Table 5, half-power width
    # static scene
    road_half_width_m: float = 6.0
    # Continuous roadside structures. A straight structure at lateral offset x0
    # traces a curve in range-Doppler as the car passes (range ~x0 and Doppler
    # ~0 alongside, Doppler -> -v_ego far ahead); DDMA replicates that curve
    # every 16 bins, which is the herringbone texture of real RADIal maps.
    # Point posts alone only produce dots along it.
    scatterers_per_m: float = 4.0               # finer than the 0.2 m range bin
    guardrail_prob: float = 0.5
    guardrail_offset_m: tuple = (5.0, 9.0)
    guardrail_rcs_per_m_dbsm: float = 0.0       # metal rail at grazing incidence
    curb_rcs_per_m_dbsm: float = -15.0
    walls_per_100m: float = 3.0                 # facades, fences, parked rows
    wall_offset_m: tuple = (8.0, 25.0)
    wall_length_m: tuple = (5.0, 40.0)
    wall_rcs_per_m_dbsm: float = -5.0
    # Azimuth antenna pattern: patch-array element gain ~cos(az) each way, so
    # the published 180-degree FoV is not uniform -- it falls to 0 at +/-90.
    azimuth_pattern_power: float = 2.0          # two-way cos^2
    # Backscatter from an extended surface falls with incidence away from its
    # normal (cos^2 law); a structure parallel to the road is seen near
    # broadside alongside the car and almost edge-on far ahead.
    roadside_objects_per_100m: float = 40.0     # poles, signs, parked cars, walls
    roadside_rcs_dbsm: tuple = (0.0, 8.0)       # mean, std
    ground_scatterers: int = 3000               # diffuse road-surface return
    ground_rcs_dbsm: tuple = (-25.0, 6.0)
    # moving vehicles
    vehicles: tuple = (1, 4)
    vehicle_speed_mps: tuple = (0.0, 20.0)
    # receiver
    highpass_corner_m: float = 4.0              # IF high-pass, 2nd order
    lowpass_edge: float = 0.93                  # anti-alias, fraction of max range
    lowpass_order: int = 6
    # Thermal noise is designed to sit above the ADC's own noise, so the filter
    # cannot suppress a bin below the ADC floor. Typical margin: ~20 dB.
    thermal_over_adc_db: float = 20.0


def _receiver_gain_db(n_range, spec, sc):
    """IF high-pass (near range) and anti-alias low-pass (far range)."""
    r = torch.arange(n_range, dtype=torch.float) * spec.range_res_m
    x = (r / sc.highpass_corner_m) ** 2
    hp = x ** 2 / (1 + x ** 2)                            # 2nd-order, power
    edge = sc.lowpass_edge * spec.max_range_m
    lp = 1.0 / (1.0 + (r / edge) ** (2 * sc.lowpass_order))
    return 10 * torch.log10(hp * lp + 1e-12)


def _elevation_gain_db(range_m, spec, sc):
    """Two-way antenna gain toward a point on the ground at this range.

    The beam points at the horizon; near-range ground sits far below it.
    Gaussian beam with the published half-power width.
    """
    dep = torch.rad2deg(torch.atan2(torch.tensor(sc.radar_height_m), range_m))
    return -2 * 12.0 * (dep / sc.elevation_beam_deg) ** 2


def _power_db(rcs_dbsm, range_m, spec):
    """Radar equation, referenced to the noise floor of one RD cell."""
    r = range_m.clamp(min=spec.range_res_m)
    return (spec.ref_snr_db + (rcs_dbsm - spec.ref_rcs_dbsm)
            - 40 * torch.log10(r / spec.ref_range_m))


class SceneSimulator:
    def __init__(self, spec=RADIAL_SPEC, scenario=Scenario(), seq_len=8,
                 device=None, generator=None):
        self.spec, self.sc, self.L = spec, scenario, seq_len
        self.device = device or torch.device("cpu")
        self.g = generator
        n, k = spec.n_range, spec.n_doppler
        self.win_r = window(n, spec.window).to(self.device)
        self.win_d = window(k, spec.window).to(self.device)
        self.rx_gain_db = _receiver_gain_db(n, spec, scenario).to(self.device)
        self.n_idx = torch.arange(n, device=self.device, dtype=torch.float)
        self.k_idx = torch.arange(k, device=self.device, dtype=torch.float)

    # ------------------------------------------------------------ sampling
    def _u(self, lo, hi, size=()):
        return torch.rand(size, generator=self.g) * (hi - lo) + lo

    def _n(self, mean, std, size=()):
        return torch.randn(size, generator=self.g) * std + mean

    def _static_scene(self, ego_travel_m):
        """World-frame static scatterers (x lateral, y forward) and RCS."""
        sc, rmax = self.sc, self.spec.max_range_m
        ymax = rmax + ego_travel_m + 5
        xs, ys, rcs, surf = [], [], [], []
        def line(x0, y0, y1, rcs_per_m, spread_db=4.0):
            """Dense scatterers along a straight structure parallel to the road."""
            n = max(int((y1 - y0) * sc.scatterers_per_m), 1)
            y = torch.linspace(float(y0), float(y1), n) + self._u(
                0, 1 / sc.scatterers_per_m, (n,))
            per = rcs_per_m - 10 * math.log10(sc.scatterers_per_m)
            xs.append(torch.full((n,), float(x0)) + self._n(0, 0.05, (n,)))
            ys.append(y)
            rcs.append(self._n(per, spread_db, (n,)))
            surf.append(torch.ones(n))

        for side in (-1, 1):
            line(side * sc.road_half_width_m, 0.5, ymax, sc.curb_rcs_per_m_dbsm)
            if torch.rand((), generator=self.g) < sc.guardrail_prob:
                off = float(self._u(*sc.guardrail_offset_m))
                line(side * off, 0.5, ymax, sc.guardrail_rcs_per_m_dbsm)
        n_walls = int(torch.poisson(torch.tensor(sc.walls_per_100m * ymax / 100),
                                    generator=self.g))
        for _ in range(n_walls):
            side = -1 if torch.rand((), generator=self.g) < 0.5 else 1
            y0 = float(self._u(1.0, ymax))
            line(side * float(self._u(*sc.wall_offset_m)), y0,
                 min(y0 + float(self._u(*sc.wall_length_m)), ymax),
                 sc.wall_rcs_per_m_dbsm)
        # roadside objects (poles, signs, parked cars): isotropic
        n_obj = int(sc.roadside_objects_per_100m * ymax / 100)
        side = torch.where(torch.rand(n_obj, generator=self.g) < 0.5, -1.0, 1.0)
        xs.append(side * self._u(sc.road_half_width_m, 30.0, (n_obj,)))
        ys.append(self._u(1.0, ymax, (n_obj,)))
        rcs.append(self._n(*sc.roadside_rcs_dbsm, (n_obj,)))
        surf.append(torch.zeros(n_obj))
        # diffuse ground: rough surface, grazing handled by elevation gain
        m = sc.ground_scatterers
        xs.append(self._u(-30.0, 30.0, (m,)))
        ys.append(self._u(0.5, ymax, (m,)))
        rcs.append(self._n(*sc.ground_rcs_dbsm, (m,)))
        surf.append(torch.zeros(m))
        return torch.cat(xs), torch.cat(ys), torch.cat(rcs), torch.cat(surf)

    def _vehicles(self):
        """Moving vehicles as extended objects: several body scatterers each."""
        sc = self.sc
        n = int(torch.randint(sc.vehicles[0], sc.vehicles[1] + 1, (),
                              generator=self.g))
        out = []
        for _ in range(n):
            kind = int(torch.multinomial(torch.tensor([0.8, 0.15, 0.05]), 1,
                                         generator=self.g))
            length, width, rcs = [(4.5, 1.8, 10.0), (10.0, 2.5, 20.0),
                                  (2.0, 0.8, 0.0)][kind]
            x = self._u(-sc.road_half_width_m + 1, sc.road_half_width_m - 1)
            y = self._u(8.0, self.spec.max_range_m - 10)
            speed = self._u(*sc.vehicle_speed_mps)
            heading = 1.0 if torch.rand((), generator=self.g) < 0.6 else -1.0
            # body scatterers: corners, wheels, centre; total RCS preserved
            ox = torch.tensor([-0.5, 0.5, -0.5, 0.5, 0.0]) * width
            oy = torch.tensor([-0.5, -0.5, 0.5, 0.5, 0.0]) * length
            share = rcs - 10 * math.log10(len(ox)) + self._n(0, 3.0, ox.shape)
            out.append({"x": x + ox, "y": y + oy, "rcs": share,
                        "vy": heading * speed, "kind": kind,
                        "cx": x, "cy": y})
        return out

    # ---------------------------------------------------------- rendering
    def _render(self, rng_m, az, vr, power_db, looks_phase):
        """RD power map (dB) for point scatterers, summed over receive antennas.

        Each scatterer is a windowed 2-D tone; the Rx antennas see it with
        angle-dependent phase (half-wavelength array), and DDMA replicates the
        result into every transmitter's Doppler slot.
        """
        spec = self.spec
        keep = (rng_m > 0) & (rng_m < spec.max_range_m) & (az.abs() < math.pi / 2)
        rng_m, az, vr, power_db = rng_m[keep], az[keep], vr[keep], power_db[keep]
        rb = (rng_m / spec.range_res_m).to(self.device)
        # Doppler bin in unshifted FFT order; negative velocities wrap
        db = (vr / spec.velocity_res_mps).to(self.device)
        amp = (10 ** (power_db / 20)).to(self.device)
        phase0 = (torch.rand(len(rb), generator=self.g) * 2 * math.pi).to(self.device)

        rt = torch.exp(2j * math.pi * torch.outer(self.n_idx, rb) / spec.n_range)
        dt = torch.exp(2j * math.pi * torch.outer(db, self.k_idx) / spec.n_doppler)
        win = torch.outer(self.win_r, self.win_d)
        # normalise so a scatterer at 0 dB lands at 0 dB in its peak cell
        norm = float(self.win_r.sum() * self.win_d.sum())
        sin_az = torch.sin(az).to(self.device)
        power = torch.zeros(spec.n_range, spec.n_doppler, device=self.device)
        for rx in range(spec.n_rx):
            a = amp * torch.exp(1j * (phase0 + math.pi * rx * sin_az))
            iq = (rt * a.unsqueeze(0)) @ dt
            # forward FFT of exp(+j 2 pi k b / N) peaks at bin +b
            rd = torch.fft.fft2(iq * win, dim=(0, 1)) / norm
            power += rd.abs() ** 2
        # DDMA: one replica per transmitter slot
        replicated = sum(torch.roll(power, off, dims=1) for off in DDMA_OFFSETS)
        return replicated

    def gen_sequence(self):
        spec, sc = self.spec, self.sc
        Tf = spec.frame_interval_s
        v_ego = float(self._u(*sc.ego_speed_mps))
        sx, sy, srcs, ssurf = self._static_scene(v_ego * Tf * self.L)
        vehicles = self._vehicles()
        frames, labels = [], []
        for l in range(self.L):
            t = l * Tf
            ego_y = v_ego * t
            xs, ys, rcs, vys = [sx], [sy - ego_y], [srcs], [torch.zeros_like(sx)]
            surfs = [ssurf]
            frame_labels = []
            for v in vehicles:
                dy = v["vy"] * t
                xs.append(v["x"]); ys.append(v["y"] + dy - ego_y)
                rcs.append(v["rcs"]); vys.append(torch.full_like(v["x"], v["vy"]))
                surfs.append(torch.zeros_like(v["x"]))
                cy = v["cy"] + dy - ego_y
                r_c = math.hypot(float(v["cx"]), float(cy))
                az_c = math.atan2(float(v["cx"]), float(cy))
                vr_c = (v["vy"] - v_ego) * math.cos(az_c)
                frame_labels.append((r_c / spec.range_res_m,
                                     (vr_c / spec.velocity_res_mps) % spec.n_doppler,
                                     v["kind"]))
            x, y = torch.cat(xs), torch.cat(ys)
            rcs, vy = torch.cat(rcs), torch.cat(vys)
            rng = torch.sqrt(x ** 2 + y ** 2)
            az = torch.atan2(x, y)
            # radial velocity: relative velocity projected on the line of sight
            vr = (vy - v_ego) * torch.cos(az)
            surf = torch.cat(surfs)
            p = _power_db(rcs, rng, spec) + _elevation_gain_db(rng, spec, sc)
            # two-way azimuth pattern of the antenna
            p = p + 10 * sc.azimuth_pattern_power * torch.log10(
                torch.cos(az).abs().clamp(min=1e-3))
            # surfaces parallel to the road: incidence from their normal is
            # 90deg - |az|, backscatter ~cos^2(incidence) = sin^2(az)
            p = p + surf * 20 * torch.log10(torch.sin(az).abs().clamp(min=1e-3))
            power = self._render(rng, az, vr, p, None)
            # thermal noise: unit power per Rx per cell, summed over the 16 Rx.
            # Not DDMA-replicated -- it is independent from chirp to chirp.
            noise = torch.distributions.Gamma(float(spec.n_rx), 1.0).sample(
                (spec.n_range, spec.n_doppler)).to(self.device)
            # the IF filters shape everything that passes through them (signal
            # and thermal noise); ADC noise is added after them and sets a floor
            filtered = (power + noise) * (10 ** (self.rx_gain_db / 10)).unsqueeze(1)
            adc = torch.distributions.Gamma(float(spec.n_rx), 1.0).sample(
                (spec.n_range, spec.n_doppler)).to(self.device)
            total = filtered + adc * 10 ** (-sc.thermal_over_adc_db / 10)
            frames.append(10 * torch.log10(total + 1e-12).cpu())
            labels.append(frame_labels)
        return {"x": torch.stack(frames), "labels": labels, "ego_speed": v_ego}
