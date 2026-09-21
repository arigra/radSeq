"""Radar link budget from published parameters, not fitted to measurements.

Every constant here comes from a datasheet, a published specification table, or
a standard propagation model -- never from the recordings the simulator is
compared against. Fitting the simulator to real maps would launder real data
into the "simulator only" baseline and turn the sim-to-real gap into a fitted
quantity instead of a measured one.

RADIal sources:
  - Rebut et al., "Raw High-Definition Radar for Multi-Task Learning",
    CVPR 2022, Table 5 (sensor specification).
  - valeoai/RADIal SignalProcessing/rpl.py (the authors' own processing code).
"""
import dataclasses
import math

import torch

C_LIGHT = 3e8


@dataclasses.dataclass(frozen=True)
class RadarSpec:
    """Published radar parameters. No entry is fitted to recorded data."""

    name: str
    fc_hz: float
    n_range: int
    n_doppler: int
    range_res_m: float
    velocity_res_mps: float
    frame_rate_hz: float
    n_rx: int
    n_tx: int
    window: str = "hamming"
    # Link-budget anchor: the SNR a reference RCS produces at a reference range.
    # Taken from the sensor class's quoted detection performance, not from maps.
    ref_rcs_dbsm: float = 10.0
    ref_range_m: float = 100.0
    ref_snr_db: float = 13.0
    # Surface-clutter reflectivity at grazing incidence, 77 GHz asphalt.
    clutter_sigma0_db: float = -25.0
    ref_cnr_db: float = 20.0

    @property
    def bandwidth_hz(self) -> float:
        return C_LIGHT / (2 * self.range_res_m)

    @property
    def chirp_interval_s(self) -> float:
        return C_LIGHT / (2 * self.fc_hz * self.n_doppler * self.velocity_res_mps)

    @property
    def frame_interval_s(self) -> float:
        return 1.0 / self.frame_rate_hz

    @property
    def max_range_m(self) -> float:
        return (self.n_range - 1) * self.range_res_m

    @property
    def n_looks(self) -> int:
        """Receive channels summed in power when forming the RD magnitude map.

        RADIal's radar_FFT holds one range-Doppler spectrum per receive
        antenna (rpl.py: RD_spectrums is (512, 256, numRxAnt)), so a power sum
        over that axis integrates exactly n_rx looks. This is a documented
        property of the data, not a free parameter.
        """
        return self.n_rx


# Rebut et al. CVPR 2022, Table 5 + valeoai/RADIal SignalProcessing/rpl.py.
RADIAL_SPEC = RadarSpec(
    name="RADIal",
    fc_hz=77e9,
    n_range=512,            # rpl.py numSamplePerChirp
    n_doppler=256,          # rpl.py numChirps
    range_res_m=0.2,        # Table 5
    velocity_res_mps=0.1,   # Table 5
    frame_rate_hz=5.0,      # Table 5
    n_rx=16,                # rpl.py numRxAnt
    n_tx=12,                # rpl.py numTxAnt
    window="hamming",       # rpl.py builds 0.54 - 0.46 cos on both axes
)


# Published automotive RCS at 76-77 GHz. These describe the target population
# on a road, not these particular recordings.
VEHICLE_RCS_DBSM = {
    "car": (10.0, 5.0),
    "truck": (20.0, 5.0),
    "motorcycle": (0.0, 5.0),
}


def target_snr_db(range_m, rcs_dbsm, spec=RADIAL_SPEC):
    """Single-target SNR from the radar equation: SNR ~ sigma / R^4.

    Anchored on the sensor class's quoted reference performance rather than on
    absolute Pt/G, which are not published for this radar.
    """
    range_m = torch.as_tensor(range_m, dtype=torch.float).clamp(min=spec.range_res_m)
    rcs_dbsm = torch.as_tensor(rcs_dbsm, dtype=torch.float)
    return (spec.ref_snr_db + (rcs_dbsm - spec.ref_rcs_dbsm)
            - 40 * torch.log10(range_m / spec.ref_range_m))


def clutter_cnr_db(range_m, spec=RADIAL_SPEC):
    """Surface clutter versus range.

    The illuminated ground patch grows linearly with range while returned power
    falls as 1/R^4, so clutter-to-noise falls as 1/R^3. Standard
    grazing-incidence surface-clutter geometry.
    """
    range_m = torch.as_tensor(range_m, dtype=torch.float).clamp(min=spec.range_res_m)
    return spec.ref_cnr_db - 30 * torch.log10(range_m / spec.ref_range_m)


def sample_rcs(n, generator=None, weights=(0.8, 0.15, 0.05)):
    """Draw RCS values for a road-vehicle population (car / truck / motorcycle)."""
    kinds = list(VEHICLE_RCS_DBSM)
    idx = torch.multinomial(torch.tensor(weights, dtype=torch.float), n,
                            replacement=True, generator=generator)
    mean = torch.tensor([VEHICLE_RCS_DBSM[k][0] for k in kinds])[idx]
    std = torch.tensor([VEHICLE_RCS_DBSM[k][1] for k in kinds])[idx]
    return mean + std * torch.randn(n, generator=generator), idx


def window(n, kind="hamming"):
    """The window RADIal applies before each FFT (rpl.py lines 128-129)."""
    if kind == "none":
        return torch.ones(n)
    if kind != "hamming":
        raise ValueError(f"unknown window {kind!r}")
    return 0.54 - 0.46 * torch.cos(2 * math.pi * torch.arange(n) / (n - 1))
