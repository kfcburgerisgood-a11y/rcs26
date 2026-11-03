"""The workflow's steps: they run, and they keep their rules. Add a test for each rule you add."""

import compas
import numpy as np
import pytest
from compas.geometry import Point, Polyline

from compas_vol import nodes
from compas_vol.meshing import is_watertight
from roof_circulation import Brief, Contribution, Drawing, checks, combine, steps

FLAT = Brief(roughness=0.0)             # the roof's top at 600 everywhere: levels are known
ROCK = Brief()
WAVY = Brief(wave=60.0)


def straight(*pins, start=(1000, 2000), end=(13000, 2000)):
    """A 12 m straight path across the stand-in roof, with optional pins."""
    return Drawing([Polyline([[start[0], start[1], 0], [end[0], end[1], 0]])], [Point(*p) for p in pins])


def build(brief, drawing):
    rock = steps.host(brief)
    profiles = steps.grade(brief, steps.ground(brief, rock, drawing))
    return rock, profiles, steps.segments(brief, profiles)


@pytest.fixture(scope="module")
def wavy():
    return build(WAVY, Drawing.sample())


def test_a_brief_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Not in the brief"):
        Brief.from_dict({"colour": 3})


def test_a_brief_never_allows_a_ramp_steeper_than_1_in_12():
    with pytest.raises(ValueError, match="at most 1:12"):
        Brief(ramp_max=1 / 10)
    assert Brief(ramp_max=1 / 20).ramp_max == pytest.approx(0.05)     # gentler is fine


def test_the_ground_is_found_where_the_host_is():
    _, profiles, _ = build(FLAT, straight())
    assert profiles[0].ground == pytest.approx(600.0, abs=0.05)


def test_a_path_off_the_host_is_refused():
    with pytest.raises(ValueError, match="No host under the path"):
        build(FLAT, straight(end=(30000, 2000)))


def test_a_pin_off_every_path_is_refused():
    with pytest.raises(ValueError, match="not on any path"):
        build(FLAT, straight((7000, 5000, 800)))


def test_on_a_flat_roof_with_nothing_held_the_path_lies_on_it():
    _, profiles, runs = build(FLAT, straight())
    assert profiles[0].level == pytest.approx(600.0, abs=0.05)
    assert len(runs[0]) == 1                                            # one level piece


def test_a_pin_raises_the_path_in_a_1_in_12_ramp_each_way():
    _, profiles, runs = build(FLAT, straight((7000, 2000, 800)))
    p = profiles[0]
    assert np.interp(7000 - 1000, p.s, p.level) == pytest.approx(800.0)  # the pin, 6 m along
    assert np.interp(5800 - 1000, p.s, p.level) == pytest.approx(700.0)  # 1.2 m before it, 1:12 lower
    assert np.interp(3000 - 1000, p.s, p.level) == pytest.approx(600.0)  # beyond 2.4 m: back on the roof
    assert checks.steepest(runs) == pytest.approx(1 / 12)


def test_a_pin_too_high_for_its_route_says_where():
    with pytest.raises(ValueError, match="Draw it longer there"):
        build(FLAT, straight((7000, 2000, 1500), end=(10000, 2000)))


def test_the_path_meets_the_rock_at_both_ends(wavy):
    _, profiles, _ = wavy
    p = profiles[0]
    assert p.level[0] == pytest.approx(p.ground[0]) and p.level[-1] == pytest.approx(p.ground[-1])


def test_the_path_is_never_steeper_than_ramp_max(wavy):
    _, profiles, runs = wavy
    p = profiles[0]
    assert np.all(np.abs(np.diff(p.level)) <= WAVY.ramp_max * np.diff(p.s) + 1e-6)
    assert checks.steepest(runs) <= WAVY.ramp_max + 1e-9


def test_the_path_is_level_through_every_turn_and_keeps_its_pin(wavy):
    _, profiles, _ = wavy
    p = profiles[0]
    for a, b, level in p.holds:
        inside = (p.s >= a - 1e-6) & (p.s <= b + 1e-6)
        assert np.ptp(p.level[inside]) == pytest.approx(0.0, abs=1e-6)
        if level is not None:
            assert p.level[inside][0] == pytest.approx(level)


def test_a_wave_rises_and_falls_and_fades_out_at_the_ends():
    _, profiles, _ = build(Brief(roughness=0.0, wave=50.0), straight())
    p = profiles[0]
    middle = (p.s > 3000) & (p.s < 9000)
    assert p.level[middle].max() == pytest.approx(650.0, abs=1)        # the full wave, away from the ends
    assert p.level[middle].min() == pytest.approx(550.0, abs=1)
    assert p.level[0] == pytest.approx(600.0) and p.level[-1] == pytest.approx(600.0)


def test_the_walking_surface_is_where_the_grade_puts_it():
    _, _, runs = build(FLAT, straight((7000, 2000, 800)))
    walk = steps.circulation(FLAT, runs).add
    assert walk.get_distance([2000, 2000, 700]) == pytest.approx(100.0)                  # on the roof, at 600
    assert walk.get_distance([5800, 2000, 800]) == pytest.approx(100 / (1 + 1 / 144) ** 0.5)  # ramp, at 700


def test_a_raised_path_reaches_depth_into_the_host_below_it():
    _, _, runs = build(FLAT, straight((7000, 2000, 800)))
    walk = steps.circulation(FLAT, runs).add
    assert walk.get_distance([7000, 2000, 500]) < 0                    # filled down into the roof
    assert walk.get_distance([7000, 2000, 400]) > 0                    # no further than depth, 150


def test_a_path_below_the_surface_cuts_its_way_through():
    rock, _, runs = build(FLAT, straight((7000, 2000, 400)))
    solid = combine(rock, [steps.circulation(FLAT, runs)])
    assert solid.get_distance([7000, 2000, 500]) > 0                   # over the path: open
    assert solid.get_distance([7000, 4000, 500]) < 0                   # beside it: still roof


def test_cut_and_fill_measure_how_far_the_path_leaves_the_surface():
    _, profiles, _ = build(FLAT, straight())
    assert checks.cut_and_fill(profiles) == pytest.approx({"cut": 0.0, "fill": 0.0}, abs=0.05)
    _, profiles, _ = build(FLAT, straight((7000, 2000, 800)))
    assert checks.cut_and_fill(profiles) == pytest.approx({"cut": 0.0, "fill": 200.0}, abs=0.05)


def test_support_keeps_the_top_of_the_path_only():
    _, _, runs = build(FLAT, straight())
    keep = steps.circulation(FLAT, runs).keep
    assert keep.get_distance([5000, 2000, 600 - 200]) < 0
    assert keep.get_distance([5000, 2000, 600 - 400]) == pytest.approx(100.0)


def test_the_wavy_sample_is_walkable_on_the_rock_with_no_lip(wavy):
    rock, _, runs = wavy
    part = steps.circulation(WAVY, runs)
    for seg in runs[0]:
        for x, y, z in seg.points(250):
            assert part.add.get_distance([x, y, z + 5]) > 0, (x, y, z)
    counts = checks.path_clear(WAVY, runs, combine(rock, [part]))
    assert counts["samples"] > 0 and counts["unsupported"] == 0 and counts["blocked"] == 0


def test_combine_lets_the_path_cut_another_modules_solid():
    rock, _, runs = build(FLAT, straight())
    planter = Contribution(add=nodes.sphere([5000, 2000, 600], 800.0))
    assert checks.path_clear(FLAT, runs, combine(rock, [planter, steps.circulation(FLAT, runs)]))["blocked"] == 0


def test_path_clear_finds_a_solid_added_after_the_cut():
    rock, _, runs = build(FLAT, straight())
    done = combine(rock, [steps.circulation(FLAT, runs)])
    late = nodes.boolean([done, nodes.sphere([5000, 2000, 600], 800.0)], "union", 0)
    assert checks.path_clear(FLAT, runs, late)["blocked"] > 0


def test_export_writes_a_closed_mesh_and_the_field(tmp_path, wavy):
    rock, profiles, runs = wavy
    coarse = Brief(wave=60.0, voxel=150.0)
    solid = combine(rock, [steps.circulation(coarse, runs)])
    vertices, faces = steps.export(coarse, solid, steps.extent(coarse, profiles), str(tmp_path / "roof"))
    assert len(faces) > 0 and is_watertight(vertices, faces)
    again = compas.json_load(str(tmp_path / "roof.json"))
    point = [5000, 2000, 700]
    assert again.get_distance(point) == pytest.approx(solid.get_distance(point))
