import math

import torch

from src.carrada_sim import CarradaSimulator, TrackScenario


def sim(seed=0, **kw):
    return CarradaSimulator(scenario=TrackScenario(**kw), generator=torch.Generator().manual_seed(seed))


def test_sequence_shape_and_reproducible_from_its_seed():
    a, b = sim(3).gen_sequence(), sim(3).gen_sequence()
    assert a["x"].shape == (8, 256, 64) and torch.isfinite(a["x"]).all()
    assert torch.equal(a["x"], b["x"])


def test_static_scene_sits_at_zero_doppler_bin_32():
    """The radar does not move, so ground and poles have zero radial velocity:
    the zero-Doppler ridge CARRADA shows at bin 32."""
    x = sim(1, p_empty=1.0).gen_sequence()["x"]
    prof = x.median(dim=1).values.median(dim=0).values
    assert int(prof.argmax()) == 32


def test_an_approaching_car_sits_above_bin_32_on_its_label():
    s = sim(0)
    o = s._object(3, 0)
    o.update(x0=0.0, y0=20.0, vx=0.0, vy=-5.0, speed=5.0)
    o["car"] = s._car_outline(math.pi)
    px, py, v_app, rcs = s._returns(o, 0.0)
    assert bool((v_app > 4.9).all())                 # moving towards the radar at 5 m/s
    p = s._render(px, py, v_app, rcs)
    r, d = divmod(int(p.argmax()), 64)
    assert d == 32 + round(5.0 / 0.42) and abs(r - (20.0 - 2.25) / 0.2) <= 2


def test_a_walker_spreads_in_doppler_and_a_car_does_not():
    """Limbs swing to ~3.3x the body's speed (micro-Doppler); a car's body moves as one."""
    s = sim(0)
    ped = s._object(1, 0)
    ped.update(x0=0.0, y0=15.0, vx=0.0, vy=-1.5, speed=1.5, phase=math.pi / 2)
    _, _, v_ped, _ = s._returns(ped, 0.0)
    car = s._object(3, 1)
    car.update(x0=0.0, y0=15.0, vx=0.0, vy=-5.0, speed=5.0)
    car["car"] = s._car_outline(math.pi)
    _, _, v_car, _ = s._returns(car, 0.0)
    assert float(v_ped.max() - v_ped.min()) > 2 * 1.5
    assert float(v_car.max() - v_car.min()) < 0.2


def test_rolling_wheel_rims_run_from_zero_to_twice_the_bike_speed():
    s = sim(0)
    bike = s._object(2, 0)
    bike.update(x0=0.0, y0=20.0, vx=0.0, vy=-5.0, speed=5.0)
    _, _, v, _ = s._returns(bike, 0.0)
    wheels = v[-16:]
    assert float(wheels.min()) < 1.0 and float(wheels.max()) > 8.5


def test_about_43_percent_of_sequences_hold_no_object():
    """CARRADA, Table II: 7,193 of 12,666 frames carry an object."""
    n = 300
    empty = sum(not sim(1000 + i).gen_sequence()["classes"] for i in range(n))
    assert abs(empty / n - (1 - 7193 / 12666)) < 0.07
