"""The workflow's steps: they run, and they keep their rules. Add a test for each rule you add."""

import math

import compas
import numpy as np
import pytest
from compas.geometry import Frame, Plane, Point, Polyline

from compas_vol import nodes
from compas_vol.meshing import is_watertight
from compas_vol.primitives import VolPlane
from roof_circulation import Brief, Contribution, Drawing, RestBrief, checks, combine, steps
from roof_circulation.routing import _tips

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


SMALL = Brief(length=4000.0, span=3000.0, edge_clearance=0.0)     # a small test roof: room only for the path


def tilted(slope, length=4000.0, span=3000.0):
    """A roof whose top rises along X at ``slope``: 600 at x = 0."""
    block = nodes.box(Frame([length / 2, span / 2, 1500], [1, 0, 0], [0, 1, 0]), length, span, 3000.0, 0)
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
    wide = Brief(length=8000.0, span=6000.0, edge_clearance=0.0)       # room to wind on
    found = steps.route(wide, steps.slopes(wide, tilted(1 / 8, 8000.0, 6000.0)), stops((700, 3000), (7300, 3000)))
    path = found.paths[0]
    assert length(path) > 1.2 * 6600                                    # not straight up the 1:8
    assert 6600 / 8 / length(path) < 1 / 8                             # so it climbs gentler on the way


def test_a_route_goes_round_a_hole_with_room_for_the_path():
    brief = Brief(roughness=0.0, length=8000.0, span=4000.0, edge_clearance=0.0)
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


def grow(brief, entrances=((1500, 2000),), roof=None):
    roof = roof if roof is not None else steps.host(brief)
    drawing = Drawing(entrances=[Point(x, y, 0) for x, y in entrances])
    return roof, steps.network(brief, roof, steps.slopes(brief, roof), drawing)


@pytest.mark.parametrize("count", [1, 2, 5])
def test_a_network_grows_as_many_branches_as_asked(count):
    _, grown = grow(Brief(roughness=0.0, branches=count))
    assert len(_tips(grown.paths, grown.entrances)) == count


@pytest.mark.parametrize("brief", [Brief(roughness=0.0, branches=5), Brief(branches=5)], ids=["flat", "rock"])
def test_every_branch_ends_clear_of_the_roofs_edge(brief):
    _, grown = grow(brief)
    for x, y in _tips(grown.paths, grown.entrances):
        assert min(x, y, brief.length - x, brief.span - y) >= brief.width / 2 + brief.edge_clearance - brief.cell


def test_a_network_is_one_walk_from_its_entrance_split_at_every_fork():
    _, grown = grow(Brief(branches=5))
    ends = [(round(p[0], 3), round(p[1], 3)) for path in grown.paths for p in (path[0], path[-1])]
    reached, frontier = set(), {(1500.0, 2000.0)}
    while frontier:                                                      # walk the pieces from the entrance
        point = frontier.pop()
        reached.add(point)
        for path in grown.paths:
            a, b = (round(path[0][0], 3), round(path[0][1], 3)), (round(path[-1][0], 3), round(path[-1][1], 3))
            for here, there in ((a, b), (b, a)):
                if here == point and there not in reached:
                    frontier.add(there)
    assert set(ends) <= reached
    for e in set(ends):                                                 # split at every fork: no piece runs on
        for path in grown.paths:                                        # through another's end
            inner = [(round(p[0], 3), round(p[1], 3)) for p in list(path)[1:-1]]
            assert e not in inner


def test_a_grown_network_can_be_walked_on_the_rock():
    brief = Brief(wave=60.0, branches=5)
    roof, grown = grow(brief)
    _, profiles, runs = build(brief, grown)
    counts = checks.path_clear(brief, runs, combine(roof, [steps.circulation(brief, runs)]))
    assert counts["samples"] > 0 and counts["unsupported"] == 0 and counts["blocked"] == 0
    assert checks.steepest(runs) <= brief.ramp_max + 1e-9


def test_a_network_needs_an_entrance():
    brief = Brief(roughness=0.0)
    roof = steps.host(brief)
    with pytest.raises(ValueError, match="at least one entrance"):
        steps.network(brief, roof, steps.slopes(brief, roof), Drawing())


def test_a_network_on_ground_too_steep_everywhere_says_so():
    with pytest.raises(ValueError, match="Nowhere to end a branch"):
        grow(SMALL, entrances=((700, 1500),), roof=tilted(1 / 8))


def test_paths_meeting_at_a_sharp_angle_are_level_as_far_as_they_overlap():
    trunk, branch = [[1000, 2000, 0], [8000, 2000, 0]], [[8000, 2000, 0], [2000, 5000, 0]]
    _, profiles, _ = build(FLAT, Drawing([Polyline(trunk), Polyline(branch)]))
    between = math.atan2(3000, 6000)                                    # the two leave the fork 26.6 degrees apart
    reach = FLAT.width / 2 / math.tan(between / 2)
    assert profiles[1].holds[0][1] == pytest.approx(reach)              # the branch's end at the fork
    assert profiles[0].holds[1][1] - profiles[0].holds[1][0] == pytest.approx(reach)


def test_paths_leaving_a_point_almost_together_are_refused():
    trunk, branch = [[1000, 2000, 0], [8000, 2000, 0]], [[8000, 2000, 0], [2000, 3000, 0]]
    with pytest.raises(ValueError, match="leave"):
        build(FLAT, Drawing([Polyline(trunk), Polyline(branch)]))


def turning(path):
    """How far a path turns in all, left and right, in degrees."""
    line = np.array([[p[0], p[1]] for p in path])
    heading = np.arctan2(*np.diff(line, axis=0).T[::-1])
    return math.degrees(np.abs((np.diff(heading) + math.pi) % (2 * math.pi) - math.pi).sum())


def crossing(brief):
    roof = steps.host(brief)
    return roof, steps.route(brief, steps.slopes(brief, roof), Drawing.sample_crossing())


def aside(path, a=(1500, 1500), b=(16500, 10500)):
    """How far each point of a path is from the straight line ``a`` to ``b``."""
    length = math.hypot(b[0] - a[0], b[1] - a[1])
    return np.array([abs((p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])) / length for p in path])


def test_with_no_turns_a_route_over_a_flat_roof_is_straight():
    path = crossing(Brief(roughness=0.0, turns=0))[1].paths[0]
    assert aside(path).max() <= Brief().cell                             # to within the grid it is found on


def test_more_turns_wind_more():
    turned = [turning(crossing(Brief(roughness=0.0, turns=n))[1].paths[0]) for n in (0, 2, 4)]
    assert turned[0] < turned[1] < turned[2]


def test_a_route_swings_to_one_side_and_back_but_keeps_its_stops():
    path = crossing(Brief(roughness=0.0, turns=1))[1].paths[0]
    length = math.hypot(15000, 9000)
    assert aside(path).max() > length / 2 / 4                           # it swings out a good part of the way
    assert list(path[0])[:2] == [1500, 1500] and list(path[-1])[:2] == [16500, 10500]


MIDDLE, ACROSS = (9000, 6000), (-9000 / math.hypot(15000, 9000), 15000 / math.hypot(15000, 9000))


def closest(path, point):
    return float(np.hypot(*(along(path) - point).T).min())


def crossing_with(brief, **points):
    roof = steps.host(brief)
    drawing = Drawing.sample_crossing()
    for name, where in points.items():
        setattr(drawing, name, [Point(x, y, 0) for x, y in where])
    return steps.route(brief, steps.slopes(brief, roof), drawing)


def test_an_attractor_draws_a_route_towards_it():
    flat = Brief(roughness=0.0)
    lure = (MIDDLE[0] + 2000 * ACROSS[0], MIDDLE[1] + 2000 * ACROSS[1])     # 2 m off the straight line
    plain = closest(crossing_with(flat).paths[0], lure)
    drawn = crossing_with(flat, attractors=[lure])
    assert closest(drawn.paths[0], lure) < plain - 500
    assert drawn.attractors[0][0] == pytest.approx(lure[0])                   # kept, for the map


def test_a_repeller_keeps_a_route_away():
    flat = Brief(roughness=0.0)
    assert closest(crossing_with(flat).paths[0], MIDDLE) < flat.cell          # straight through it, without
    assert closest(crossing_with(flat, repellers=[MIDDLE]).paths[0], MIDDLE) > 1000


def test_an_attractor_near_a_swing_captures_it_even_on_the_rock():
    rock = Brief(turns=4)
    lure = (6000, 6000)
    plain = closest(crossing_with(rock).paths[0], lure)
    assert closest(crossing_with(rock, attractors=[lure]).paths[0], lure) < min(plain, rock.width)


def test_a_wider_swing_bows_further():
    narrow, wide = (crossing(Brief(roughness=0.0, turns=1, swing=s))[1].paths[0] for s in (0.25, 0.75))
    assert aside(wide).max() > aside(narrow).max() + 1000


def test_a_longer_bend_sweeps_and_a_shorter_one_hugs_the_ground():
    loose, tight = (turning(crossing(Brief(turns=4, bend=b))[1].paths[0]) for b in (4800.0, 1200.0))
    assert loose < tight


def test_attract_is_a_share_cheaper_under_one():
    with pytest.raises(ValueError, match="attract"):
        Brief(attract=1.0)


def test_a_winding_crossing_can_be_walked_on_the_rock():
    brief = Brief(wave=60.0, turns=4)
    roof, drawing = crossing(brief)
    _, profiles, runs = build(brief, drawing)
    counts = checks.path_clear(brief, runs, combine(roof, [steps.circulation(brief, runs)]))
    assert counts["samples"] > 0 and counts["unsupported"] == 0 and counts["blocked"] == 0
    assert checks.steepest(runs) <= brief.ramp_max + 1e-9


def test_a_branch_winds_over_the_rock_rather_than_run_straight():
    _, grown = grow(Brief(branches=1))
    line = np.array([[p[0], p[1]] for p in grown.paths[0]])
    heading = np.arctan2(*np.diff(line, axis=0).T[::-1])
    turning = np.abs((np.diff(heading) + math.pi) % (2 * math.pi) - math.pi).sum()
    assert len(line) > 20 and math.degrees(turning) > 60


@pytest.mark.parametrize("bend, landing", [(6, False), (12, True)])
def test_a_gentle_bend_climbs_on_and_a_sharper_one_is_a_landing(bend, landing):
    turn = math.radians(bend)
    corner = (7000, 2000)
    path = [[1000, 2000, 0], [*corner, 0], [corner[0] + 5000 * math.cos(turn), corner[1] + 5000 * math.sin(turn), 0]]
    _, profiles, _ = build(FLAT, Drawing([Polyline(path)]))
    assert (len(profiles[0].holds) > 2) == landing                      # beyond the two ends


HILL = Drawing([Polyline([[1000, 2000, 0], [17000, 2000, 0]])], [Point(9000, 2000, 1050)])   # 450 up, 450 down


FLAT_HILL = Brief(roughness=0.0)


def rested(landing, drawing=HILL, brief=FLAT_HILL):
    roof = steps.host(brief)
    profiles = steps.rest(brief, landing, roof, steps.grade(brief, steps.ground(brief, roof, drawing)), drawing)
    return roof, profiles, steps.segments(brief, profiles)


def test_a_path_is_at_least_1200_wide():
    with pytest.raises(ValueError, match="at least 1200"):
        Brief(width=1100.0)
    assert Brief(width=1500.0).width == 1500.0


def test_rest_numbers_are_their_own_apart_from_the_brief():
    assert not any(name.startswith("rest") for name in Brief().to_dict())
    with pytest.raises(ValueError, match="Not in the rest brief"):
        RestBrief.from_dict({"width": 1500})


def test_a_long_climb_rests_every_rise_on_a_level_landing():
    landing = RestBrief(rise=300.0)
    _, profiles, _ = rested(landing)
    p = profiles[0]
    assert len(p.rests) == 2                                            # one on the way up, one down
    for a, b, *_ in p.rests:
        assert b - a == pytest.approx(landing.length)
        inside = (p.s >= a - 1e-6) & (p.s <= b + 1e-6)
        assert np.ptp(p.level[inside]) == pytest.approx(0.0, abs=1e-6)
        assert np.interp((a + b) / 2, p.s, p.level) == pytest.approx(825.0)   # halfway up the 450
    assert steps._long_climb(landing, p) is None                        # nothing left climbs too far


def test_a_network_checks_its_branches_with_rests_only_when_given_them():
    brief = Brief(roughness=0.0, branches=2)
    roof = steps.host(brief)
    found = steps.slopes(brief, roof)
    for landing in (None, RestBrief()):
        grown = steps.network(brief, roof, found, Drawing.sample_entrance(), landing)
        assert len(_tips(grown.paths, grown.entrances)) == 2


def test_leaving_the_rest_step_out_leaves_the_path_without_rests():
    roof = steps.host(FLAT_HILL)
    profiles = steps.grade(FLAT_HILL, steps.ground(FLAT_HILL, roof, HILL))
    assert not profiles[0].rests and len(steps.segments(FLAT_HILL, profiles)) == 1


def test_no_rests_when_the_rise_is_zero_or_no_climb_is_long_enough():
    assert not rested(RestBrief(rise=0.0))[1][0].rests
    assert not rested(RestBrief(rise=500.0))[1][0].rests                # 450 up, then down: counted apart


def test_a_rest_goes_by_an_attractor_near_where_it_may():
    landing = RestBrief(rise=300.0)
    plain = rested(landing)[1][0].rests[0][0]
    lured = Drawing(list(HILL.paths), list(HILL.pins), attractors=[Point(5500, 2000, 0)])
    assert rested(landing, lured)[1][0].rests[0][0] > plain + 200


def test_each_rest_has_a_bay_beside_it_to_stand_in():
    landing = RestBrief(rise=300.0)
    roof, profiles, runs = rested(landing)
    bays = runs[1:]
    assert len(bays) == 2 and all(r[0].width == landing.bay for r in bays)
    x, y, level = bays[0][0].start
    assert abs(y - 2000) == pytest.approx((FLAT_HILL.width + landing.bay) / 2 - steps.OVERLAP)
    solid = combine(roof, [steps.circulation(FLAT_HILL, runs)])
    assert solid.get_distance([x, y, level - 50]) < 0                   # floor
    assert solid.get_distance([x, y, level + 1000]) > 0                 # open above
    assert len(rested(RestBrief(rise=300.0, bay=0.0))[2]) == 1


def test_a_climb_too_short_for_its_rests_says_so():
    with pytest.raises(ValueError, match="rest landings"):
        rested(RestBrief(rise=300.0, length=4000.0))


@pytest.mark.parametrize("clearance", [0.0, 800.0])
def test_a_route_keeps_its_edge_clearance_from_the_roofs_edge_all_along(clearance):
    brief = Brief(roughness=0.0, turns=4, swing=1.0, edge_clearance=clearance)
    path = along(crossing(brief)[1].paths[0], 50.0)
    nearest = np.min([path[:, 0], path[:, 1], brief.length - path[:, 0], brief.span - path[:, 1]], axis=0)
    inner = nearest[(np.hypot(*(path - [1500, 1500]).T) > 3000) & (np.hypot(*(path - [16500, 10500]).T) > 3000)]
    assert inner.min() >= brief.width / 2 + clearance - brief.cell        # away from its two stops


def test_lift_raises_the_walking_surface_over_what_it_follows():
    _, profiles, _ = build(Brief(roughness=0.0, lift=10.0), straight())
    assert profiles[0].level == pytest.approx(610.0, abs=0.05)          # the flat roof at 600, and 10 over it
    _, profiles, _ = build(Brief(roughness=0.0, lift=10.0), straight((7000, 2000, 800)))
    assert np.interp(6000, profiles[0].s, profiles[0].level) == pytest.approx(800.0)   # a pin stays as set


def test_a_path_is_followed_all_the_way_to_its_end():
    # 5830.951894845... mm long: rounding the last sample up must not drop it
    _, profiles, _ = build(FLAT, straight(start=(1000, 2000), end=(6000, 5000)))
    p = profiles[0]
    assert (p.x[-1], p.y[-1]) == pytest.approx((6000, 5000))
    assert p.length == pytest.approx(math.hypot(5000, 3000))


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


def test_a_path_cut_deeper_than_depth_still_stands_on_its_own_solid():
    _, _, runs = build(FLAT, straight((7000, 2000, 300)))              # 300 into a roof at 600, depth 150
    walk = steps.circulation(FLAT, runs).add
    assert walk.get_distance([7000, 2000, 300 - 50]) < 0               # solid just under the walking surface
    assert walk.get_distance([7000, 2000, 300 - FLAT.depth - 50]) > 0  # and no deeper than depth under it


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
