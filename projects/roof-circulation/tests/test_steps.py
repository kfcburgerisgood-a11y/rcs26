"""The workflow's steps: they run, and they keep their rules. Add a test for each rule you add."""

import math

import compas
import numpy as np
import pytest
from compas.geometry import Frame, Plane, Point, Polyline

from compas_vol import nodes
from compas_vol.meshing import is_watertight
from compas_vol.primitives import VolPlane
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


SMALL = Brief(length=4000.0, span=3000.0)


def tilted(slope):
    """A roof whose top rises along X at ``slope``: 600 at x = 0."""
    block = nodes.box(Frame([2000, 1500, 1500], [1, 0, 0], [0, 1, 0]), 4000.0, 3000.0, 3000.0, 0)
    return nodes.boolean([block, VolPlane(Plane([0, 0, 600], [-slope, 0, 1]))], "intersection", 0)


@pytest.mark.parametrize("slope", [1 / 20, 1 / 8])
def test_the_slope_map_reads_a_known_slope(slope):
    found = steps.slopes(SMALL, tilted(slope))
    assert found.slope == pytest.approx(slope, abs=1e-4)
    assert found.height[0, 0] == pytest.approx(600 + slope * found.x[0], abs=0.05)
    assert found.walkable(SMALL.ramp_max).all() == (slope <= SMALL.ramp_max)


def test_the_slope_map_marks_a_hole_as_no_roof():
    block = nodes.box(Frame([2000, 1500, 300], [1, 0, 0], [0, 1, 0]), 4000.0, 3000.0, 600.0, 0)
    hole = nodes.cylinder(Frame([2000, 1500, 300], [1, 0, 0], [0, 1, 0]), 600.0, 1000.0)
    found = steps.slopes(SMALL, nodes.boolean([block, hole], "difference", 0))
    i, j = np.searchsorted(found.x, 2000), np.searchsorted(found.y, 1500)
    assert np.isnan(found.height[i, j]) and not found.walkable(SMALL.ramp_max)[i, j]
    assert found.walkable(SMALL.ramp_max)[0, 0]                         # far from it: flat roof


def stops(*points):
    return Drawing(stops=[Polyline([[x, y, 0] for x, y in points])])


def along(path, step=50.0):
    """Plan points every ``step`` mm along a path."""
    points = []
    for p, q in zip(path[:-1], path[1:]):
        n = max(2, int(np.hypot(q[0] - p[0], q[1] - p[1]) / step))
        points += [(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])) for t in np.linspace(0, 1, n)]
    return np.array(points)


def length(path):
    return float(np.hypot(*np.diff(along(path, 1e9), axis=0).T).sum())


def test_a_route_keeps_every_stop_as_a_corner():
    brief = Brief(roughness=0.0)
    found = steps.route(brief, steps.slopes(brief, steps.host(brief)), stops((1500, 2000), (9000, 6000), (3000, 9000)))
    corners = [(p[0], p[1]) for p in found.paths[0]]
    assert (1500, 2000) in corners and (9000, 6000) in corners and (3000, 9000) in corners
    assert not found.stops


def test_a_route_up_a_steep_slope_goes_across_it():
    found = steps.route(SMALL, steps.slopes(SMALL, tilted(1 / 8)), stops((700, 1500), (3300, 1500)))
    path = found.paths[0]
    assert length(path) > 1.2 * 2600                                    # not straight up the 1:8
    assert (3300 - 700) / 8 / length(path) < 1 / 8                     # so it climbs gentler on the way


def test_a_route_goes_round_a_hole_with_room_for_the_path():
    brief = Brief(roughness=0.0, length=8000.0, span=4000.0)
    block = nodes.box(Frame([4000, 2000, 300], [1, 0, 0], [0, 1, 0]), 8000.0, 4000.0, 600.0, 0)
    hole = nodes.cylinder(Frame([4000, 2000, 300], [1, 0, 0], [0, 1, 0]), 600.0, 1000.0)
    roof = nodes.boolean([block, hole], "difference", 0)
    path = steps.route(brief, steps.slopes(brief, roof), stops((1000, 2000), (7000, 2000))).paths[0]
    clear = np.hypot(*(along(path) - [4000, 2000]).T).min()
    assert clear >= 600 + brief.width / 2 - brief.cell                 # the hole's radius and half the path, to a cell


def test_a_stop_without_room_for_the_path_is_refused():
    with pytest.raises(ValueError, match="has no room for the path"):
        steps.route(SMALL, steps.slopes(SMALL, tilted(0.0)), stops((200, 1500), (3300, 1500)))


def test_a_route_with_no_way_through_says_so():
    block = nodes.box(Frame([2000, 1500, 300], [1, 0, 0], [0, 1, 0]), 4000.0, 3000.0, 600.0, 0)
    gap = nodes.box(Frame([2000, 1500, 300], [1, 0, 0], [0, 1, 0]), 400.0, 4000.0, 1000.0, 0)
    roof = nodes.boolean([block, gap], "difference", 0)                 # cut in two, across its whole span
    with pytest.raises(ValueError, match="No way from"):
        steps.route(SMALL, steps.slopes(SMALL, roof), stops((700, 1500), (3300, 1500)))


def test_a_routed_walk_never_runs_over_itself_and_can_be_walked():
    roof = steps.host(WAVY)
    drawing = steps.route(WAVY, steps.slopes(WAVY, roof), Drawing.sample_stops())
    _, profiles, runs = build(WAVY, drawing)
    counts = checks.path_clear(WAVY, runs, combine(roof, [steps.circulation(WAVY, runs)]))
    assert counts["samples"] > 0 and counts["unsupported"] == 0 and counts["blocked"] == 0
    assert checks.steepest(runs) <= WAVY.ramp_max + 1e-9


def test_a_sharp_turn_is_level_as_far_as_its_arms_overlap():
    _, profiles, _ = build(FLAT, Drawing([Polyline([[1000, 2000, 0], [9000, 2000, 0], [2000, 6000, 0]])]))
    turn = math.acos(-7000 / math.hypot(7000, 4000))                    # about 150 degrees: arms overlap for 2.2 m
    reach = FLAT.width / 2 * math.tan(turn / 2)
    held = [h for h in profiles[0].holds if h[0] > 0 and h[1] < profiles[0].length]
    assert held[0][1] - held[0][0] == pytest.approx(2 * reach)


def test_a_path_turning_straight_back_is_refused():
    with pytest.raises(ValueError, match="turns back on itself"):
        build(FLAT, Drawing([Polyline([[1000, 2000, 0], [9000, 2000, 0], [2000, 2600, 0]])]))


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
