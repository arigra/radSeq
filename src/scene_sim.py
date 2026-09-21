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
class RoadType:
    """Composition of one kind of road. General road knowledge, not fitted.

    Structures are segments with gaps (driveways, junctions, breaks), and
    parked cars and vegetation are objects and clusters: a road lined with one
    unbroken structure is what makes a simulated map show long clean arcs.
    """

    name: str
    ego_speed_mps: tuple                  # uniform range
    p_ego_stopped: float                  # e.g. waiting at a light
    road_half_width_m: tuple
    curb: tuple                           # (coverage 0-1, rcs per m dBsm)
    guardrail: tuple                      # (coverage, rcs per m, lateral range m)
    facades: tuple                        # (coverage, rcs per m, lateral range m)
    parked_car_coverage: float            # share of the kerb occupied by cars
    trees_per_100m: float
    poles_per_100m: float
    vehicle_speed_mps: tuple
    p_oncoming: float


# RADIal README: "sequences are categorized in highway, country-side and city
# driving". The published distribution over them is not machine-readable, so
# the three are drawn equally.
ROAD_TYPES = {
    "city": RoadType(
        "city", ego_speed_mps=(0.0, 14.0), p_ego_stopped=0.2,
        road_half_width_m=(3.5, 7.0), curb=(0.7, -15.0),
        guardrail=(0.0, 0.0, (4.0, 8.0)), facades=(0.6, -5.0, (7.0, 18.0)),
        parked_car_coverage=0.5, trees_per_100m=4.0, poles_per_100m=5.0,
        vehicle_speed_mps=(0.0, 14.0), p_oncoming=0.4),
    "highway": RoadType(
        "highway", ego_speed_mps=(22.0, 36.0), p_ego_stopped=0.0,
        road_half_width_m=(5.5, 9.0), curb=(0.0, -15.0),
        guardrail=(0.9, 0.0, (4.0, 10.0)), facades=(0.0, -5.0, (15.0, 30.0)),
        parked_car_coverage=0.0, trees_per_100m=1.0, poles_per_100m=2.0,
        vehicle_speed_mps=(20.0, 36.0), p_oncoming=0.15),
    "countryside": RoadType(
        "countryside", ego_speed_mps=(12.0, 25.0), p_ego_stopped=0.0,
        road_half_width_m=(3.0, 5.0), curb=(0.2, -15.0),
        guardrail=(0.3, 0.0, (3.5, 6.0)), facades=(0.05, -5.0, (10.0, 30.0)),
        parked_car_coverage=0.0, trees_per_100m=15.0, poles_per_100m=3.0,
        vehicle_speed_mps=(10.0, 25.0), p_oncoming=0.5),
}


@dataclasses.dataclass(frozen=True)
class Scenario:
    """Everything about the scene and sensor installation. Nothing is fitted."""

    road_types: tuple = ("city", "highway", "countryside")
    radar_height_m: float = 0.8                 # Table 5
    elevation_beam_deg: float = 12.0            # Table 5, half-power width
    scatterers_per_m: float = 4.0               # finer than the 0.2 m range bin
    segment_length_m: tuple = (8.0, 40.0)       # continuous stretch of a structure
    gap_length_m: tuple = (3.0, 15.0)           # driveway, junction, break
    car_length_m: float = 4.5
    car_rcs_dbsm: float = 10.0                  # published 77 GHz car RCS
    tree_rcs_dbsm: tuple = (-5.0, 5.0)          # whole tree, mean and std
    pole_rcs_dbsm: tuple = (5.0, 4.0)
    ground_scatterers: int = 3000               # diffuse road-surface return
    ground_rcs_dbsm: tuple = (-25.0, 6.0)
    # RADIal README: 9,550 labelled vehicles over 8,252 labelled frames. The
    # labels cover traffic, not every parked car, so moving vehicles are drawn
    # with this mean and parked cars are scene, not labels.
    vehicles_per_frame: float = 9550 / 8252
    # antenna and surface scattering
    azimuth_pattern_power: float = 2.0          # two-way cos^2 element pattern
    # receiver
    highpass_corner_m: float = 4.0              # IF high-pass, 2nd order
    lowpass_edge: float = 0.93                  # anti-alias, fraction of max range
    lowpass_order: int = 6
    thermal_over_adc_db: float = 20.0           # thermal noise above ADC floor


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

    def _segments(self, y0, y1, coverage):
        """Stretches of a structure covering about `coverage` of [y0, y1]."""
        sc = self.sc
        out, y = [], y0 + float(self._u(0, sc.gap_length_m[1]))
        while y < y1 and coverage > 0:
            length = float(self._u(*sc.segment_length_m))
            if torch.rand((), generator=self.g) < coverage:
                out.append((y, min(y + length, y1)))
            y += length + float(self._u(*sc.gap_length_m))
        return out

    def _static_scene(self, road, half_width, ymax):
        """World-frame static scatterers (x lateral, y forward).

        Returns x, y, rcs, surface flag, plus the parked cars (which are
        vehicles, so they are also reported as labels).
        """
        sc = self.sc
        xs, ys, rcs, surf = [], [], [], []

        def line(x0, y0, y1, rcs_per_m, spread_db=4.0):
            n = max(int((y1 - y0) * sc.scatterers_per_m), 1)
            y = torch.linspace(float(y0), float(y1), n) + self._u(
                0, 1 / sc.scatterers_per_m, (n,))
            per = rcs_per_m - 10 * math.log10(sc.scatterers_per_m)
            xs.append(torch.full((n,), float(x0)) + self._n(0, 0.05, (n,)))
            ys.append(y)
            rcs.append(self._n(per, spread_db, (n,)))
            surf.append(torch.ones(n))

        def points(x, y, r):
            xs.append(x); ys.append(y); rcs.append(r); surf.append(torch.zeros(len(x)))

        parked = []
        for side in (-1.0, 1.0):
            cov, per_m = road.curb
            for a, b in self._segments(0.5, ymax, cov):
                line(side * half_width, a, b, per_m)
            cov, per_m, lat = road.guardrail
            if cov > 0:
                off = float(self._u(*lat))
                for a, b in self._segments(0.5, ymax, cov):
                    line(side * off, a, b, per_m)
            cov, per_m, lat = road.facades
            for a, b in self._segments(0.5, ymax, cov):
                line(side * float(self._u(*lat)), a, b, per_m)
            # parked cars: a row just outside the kerb, with gaps
            for a, b in self._segments(0.5, ymax, road.parked_car_coverage):
                y = a
                while y + sc.car_length_m < b:
                    parked.append(self._car(side * (half_width + 1.2), y + 2.25,
                                            sc.car_rcs_dbsm, vy=0.0))
                    y += sc.car_length_m + float(self._u(0.5, 2.0))
        # trees and bushes: clusters of weak scatterers
        n_trees = int(torch.poisson(torch.tensor(road.trees_per_100m * ymax / 100),
                                    generator=self.g))
        for _ in range(n_trees):
            side = -1.0 if torch.rand((), generator=self.g) < 0.5 else 1.0
            cx = side * float(self._u(half_width + 1.0, half_width + 12.0))
            cy = float(self._u(1.0, ymax))
            k = int(self._u(15, 40))
            total = float(self._n(*sc.tree_rcs_dbsm))
            points(cx + self._n(0, 1.0, (k,)), cy + self._n(0, 1.0, (k,)),
                   self._n(total - 10 * math.log10(k), 5.0, (k,)))
        # poles and signs: isolated strong points
        n_poles = int(torch.poisson(torch.tensor(road.poles_per_100m * ymax / 100),
                                    generator=self.g))
        side = torch.where(torch.rand(n_poles, generator=self.g) < 0.5, -1.0, 1.0)
        points(side * (half_width + self._u(0.5, 3.0, (n_poles,))),
               self._u(1.0, ymax, (n_poles,)), self._n(*sc.pole_rcs_dbsm, (n_poles,)))
        # diffuse ground
        m = sc.ground_scatterers
        points(self._u(-30.0, 30.0, (m,)), self._u(0.5, ymax, (m,)),
               self._n(*sc.ground_rcs_dbsm, (m,)))
        return (torch.cat(xs), torch.cat(ys), torch.cat(rcs), torch.cat(surf),
                parked)

    def _car(self, x, y, rcs_dbsm, vy, kind=0):
        """A vehicle as several body scatterers; total RCS preserved."""
        length, width = [(4.5, 1.8), (10.0, 2.5), (2.0, 0.8)][kind]
        ox = torch.tensor([-0.5, 0.5, -0.5, 0.5, 0.0]) * width
        oy = torch.tensor([-0.5, -0.5, 0.5, 0.5, 0.0]) * length
        share = rcs_dbsm - 10 * math.log10(len(ox)) + self._n(0, 3.0, ox.shape)
        return {"x": x + ox, "y": y + oy, "rcs": share, "vy": vy, "kind": kind,
                "cx": x, "cy": y, "moving": vy != 0.0}

    def _vehicles(self, road, half_width, v_ego):
        n = int(torch.poisson(torch.tensor(self.sc.vehicles_per_frame),
                              generator=self.g))
        out = []
        for _ in range(n):
            kind = int(torch.multinomial(torch.tensor([0.8, 0.15, 0.05]), 1,
                                         generator=self.g))
            rcs = (10.0, 20.0, 0.0)[kind]
            oncoming = bool(torch.rand((), generator=self.g) < road.p_oncoming)
            lane = float(self._u(0.3, 0.9)) * half_width
            x = -lane if oncoming else lane * float(self._u(-1.0, 1.0))
            speed = float(self._u(*road.vehicle_speed_mps))
            out.append(self._car(x, float(self._u(8.0, self.spec.max_range_m - 10)),
                                 rcs, vy=-speed if oncoming else speed, kind=kind))
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
        name = sc.road_types[int(torch.randint(len(sc.road_types), (),
                                               generator=self.g))]
        road = ROAD_TYPES[name]
        stopped = torch.rand((), generator=self.g) < road.p_ego_stopped
        v_ego = 0.0 if stopped else float(self._u(*road.ego_speed_mps))
        half_width = float(self._u(*road.road_half_width_m))
        ymax = spec.max_range_m + v_ego * Tf * self.L + 5
        sx, sy, srcs, ssurf, parked = self._static_scene(road, half_width, ymax)
        vehicles = self._vehicles(road, half_width, v_ego) + parked
        frames, labels = [], []
        for l in range(self.L):
            t = l * Tf
            ego_y = v_ego * t
            xs, ys, rcs = [sx], [sy - ego_y], [srcs]
            vys, surfs = [torch.zeros_like(sx)], [ssurf]
            frame_labels = []
            for v in vehicles:
                dy = v["vy"] * t
                xs.append(v["x"]); ys.append(v["y"] + dy - ego_y)
                rcs.append(v["rcs"]); vys.append(torch.full_like(v["x"], v["vy"]))
                surfs.append(torch.zeros_like(v["x"]))
                cy = v["cy"] + dy - ego_y
                if (not v["moving"] or cy <= 0
                        or not 0 < math.hypot(v["cx"], cy) < spec.max_range_m):
                    continue
                r_c = math.hypot(float(v["cx"]), float(cy))
                az_c = math.atan2(float(v["cx"]), float(cy))
                vr_c = (v["vy"] - v_ego) * math.cos(az_c)
                frame_labels.append({
                    "range_bin": r_c / spec.range_res_m,
                    "doppler_bin": (vr_c / spec.velocity_res_mps) % spec.n_doppler,
                    "kind": v["kind"], "moving": v["moving"]})
            x, y = torch.cat(xs), torch.cat(ys)
            rcs_all, vy, surf = torch.cat(rcs), torch.cat(vys), torch.cat(surfs)
            rng = torch.sqrt(x ** 2 + y ** 2)
            az = torch.atan2(x, y)
            vr = (vy - v_ego) * torch.cos(az)
            p = _power_db(rcs_all, rng, spec) + _elevation_gain_db(rng, spec, sc)
            p = p + 10 * sc.azimuth_pattern_power * torch.log10(
                torch.cos(az).abs().clamp(min=1e-3))
            # surfaces parallel to the road: backscatter ~cos^2(incidence),
            # incidence from their normal = 90deg - |az|
            p = p + surf * 20 * torch.log10(torch.sin(az).abs().clamp(min=1e-3))
            power = self._render(rng, az, vr, p, None)
            # thermal noise: unit power per Rx per cell, summed over 16 Rx;
            # not DDMA-replicated (independent from chirp to chirp)
            noise = torch.distributions.Gamma(float(spec.n_rx), 1.0).sample(
                (spec.n_range, spec.n_doppler)).to(self.device)
            filtered = (power + noise) * (10 ** (self.rx_gain_db / 10)).unsqueeze(1)
            adc = torch.distributions.Gamma(float(spec.n_rx), 1.0).sample(
                (spec.n_range, spec.n_doppler)).to(self.device)
            total = filtered + adc * 10 ** (-sc.thermal_over_adc_db / 10)
            frames.append(10 * torch.log10(total + 1e-12).cpu())
            labels.append(frame_labels)
        return {"x": torch.stack(frames), "labels": labels,
                "ego_speed": v_ego, "road_type": name}
