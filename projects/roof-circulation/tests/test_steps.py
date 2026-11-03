"""The workflow's steps: they run, and they keep their rules. Add a test for each rule you add."""

import math

import compas
import pytest
from compas.geometry import Polyline

from compas_vol import nodes
from compas_vol.meshing import is_watertight
from roof_circulation import Brief, Contribution, Drawing, checks, combine, steps


def straight(*points):
    return Drawing([Polyline([list(p) for p in points])])


# the tests' own path, so the sample drawing can change: level, a 1:13.3 ramp, a turn, 5 steps up, a terrace
CLIMB = straight((1000, 1500, 600), (4500, 1500, 600), (7500, 1500, 780), (7500, 4500, 1400), (10500, 4500, 1400))
# and one that goes down: a 1:14 ramp to the deck, a turn, 3 steps down to 400 mm below it
DROP = straight((1000, 1500, 800), (5000, 1500, 600), (5000, 4500, 200), (8000, 4500, 200))


@pytest.fixture(scope="module")
def brief():
    return Brief()


@pytest.fixture(scope="module")
def runs(brief):
    return steps.segments(brief, CLIMB)


@pytest.fixture(scope="module")
def part(brief):
    return steps.circulation(brief, CLIMB)


def test_a_brief_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Not in the brief"):
        Brief.from_dict({"colour": 3})


def test_a_run_up_to_ramp_max_is_a_ramp_and_steeper_is_stairs(brief):
    gentle = steps.segments(brief, straight((0, 0, 0), (2600, 0, 100)))[0][0]     # 1:14 between its landings
    steep = steps.segments(brief, straight((0, 0, 0), (3000, 0, 400)))[0][0]
    assert gentle.kind == "ramp"
    assert steep.kind == "stairs" and steep.risers == 3          # 400 / 150, rounded up
    assert steep.going == pytest.approx((3000 - brief.width) / 3)


def test_a_path_straight_up_is_refused(brief):
    with pytest.raises(ValueError, match="rises straight up"):
        steps.segments(brief, straight((0, 0, 0), (0, 0, 500)))


def test_the_walking_surface_is_where_the_drawing_puts_it(part):
    assert part.add.get_distance([2000, 1500, 700]) == pytest.approx(100.0)             # level, at 600
    # the ramp has no landing where it continues the level run, and one 600 long at the turn
    assert part.add.get_distance([6000, 1500, 812.5]) == pytest.approx(100 / (1 + 0.075 ** 2) ** 0.5)  # 712.5
    assert part.add.get_distance([7200, 1500, 880]) == pytest.approx(100.0)            # its landing, at 780
    assert part.add.get_distance([7500, 2640, 1078]) == pytest.approx(50.0)            # second step, 1028


@pytest.mark.parametrize("drawing", [CLIMB, DROP], ids=["climb", "drop"])
def test_nothing_stands_proud_of_a_ramp_where_it_meets_a_vertex(brief, drawing):
    walk = steps.circulation(brief, drawing).add
    for seg in steps.segments(brief, drawing)[0]:
        if seg.kind == "ramp" and seg.rise:
            for x, y, z in seg.points(100):
                assert walk.get_distance([x, y, z + 5]) > 0, (x, y, z)


def test_a_path_going_down_ramps_and_steps_like_one_going_up(brief):
    runs = steps.segments(brief, DROP)[0]
    assert [s.kind for s in runs] == ["ramp", "stairs", "ramp"]
    assert runs[1].risers == 3 and runs[1].going == pytest.approx(600.0)
    walk = steps.circulation(brief, DROP).add
    level = 600 + 200 * (2500 - 600) / 2800                                              # 2500 from the low end
    assert walk.get_distance([2500, 1500, level + 100]) == pytest.approx(100 / (1 + (2 / 28) ** 2) ** 0.5)
    assert walk.get_distance([5000, 3600, 200 + 400 / 3 + 50]) == pytest.approx(50.0)    # lowest step


def test_a_path_below_the_deck_cuts_its_own_trench(brief):
    solid = combine(steps.host(brief), [steps.circulation(brief, DROP)])
    assert solid.get_distance([6500, 4500, 500]) > 0          # over the sunken path: open
    assert solid.get_distance([6500, 3000, 500]) < 0          # beside it: still deck
    counts = checks.path_clear(brief, steps.segments(brief, DROP), solid)
    assert counts["unsupported"] == 0 and counts["blocked"] == 0


WAVY = Brief(wave=80.0, wavelength=6000.0)


def test_no_wave_leaves_the_drawing_as_drawn(brief):
    assert steps.undulate(brief, CLIMB) is CLIMB


def test_a_wave_keeps_every_drawn_level_and_every_flight(brief):
    wavy = steps.undulate(WAVY, Drawing.sample())
    for p in Drawing.sample().paths[0]:
        assert any(abs(q[0] - p[0]) + abs(q[1] - p[1]) + abs(q[2] - p[2]) < 1e-6 for q in wavy.paths[0])
    flights = [(s.start, s.end) for s in steps.segments(WAVY, wavy)[0] if s.risers]
    assert flights == [(s.start, s.end) for s in steps.segments(WAVY, Drawing.sample())[0] if s.risers]


def test_a_wave_never_passes_ramp_max(brief):
    ramps = [s for s in steps.segments(WAVY, steps.undulate(WAVY, Drawing.sample()))[0] if not s.risers]
    assert max(abs(s.rise) / s.run for s in ramps) <= WAVY.ramp_max + 1e-9


def test_a_wave_is_full_where_there_is_room_and_lowered_where_not():
    def crest(brief, drawing):
        return max(p[2] for p in steps.undulate(brief, drawing).paths[0])

    long_level = straight((0, 0, 600), (8000, 0, 600))           # 6800 between landings: one 6.8 m wave
    assert crest(WAVY, long_level) == pytest.approx(680, abs=1)
    short_level = straight((0, 0, 600), (3000, 0, 600))          # 1800: an 80 mm wave would pass 1:12
    assert crest(WAVY, short_level) == pytest.approx(600 + 1800 / 12 / (2 * math.pi), abs=1)
    at_the_limit = straight((0, 0, 0), (3600, 0, 200))           # 1:12 already: no room at all
    assert crest(WAVY, at_the_limit) == pytest.approx(200)


def test_a_wavy_path_is_walkable_with_no_lip(brief):
    wavy = steps.undulate(WAVY, Drawing.sample())
    runs = steps.segments(WAVY, wavy)
    solid = combine(steps.host(WAVY), [steps.circulation(WAVY, wavy)])
    counts = checks.path_clear(WAVY, runs, solid)
    assert counts["unsupported"] == 0 and counts["blocked"] == 0


def test_a_path_is_solid_down_to_depth_below_its_lowest_level(brief, part):
    floor = 600 - brief.depth
    assert part.add.get_distance([7500, 3000, floor - 100]) == pytest.approx(100.0)     # under the stairs


def test_the_space_over_the_path_is_cut_and_the_path_is_not(part):
    assert part.cut.get_distance([2000, 1500, 1600]) < 0
    assert part.cut.get_distance([2000, 1500, 500]) > 0


def test_a_path_clears_its_way_through_a_taller_host(brief, part):
    tall = steps.host(Brief(thickness=2000.0))
    assert combine(tall, [part]).get_distance([2000, 1500, 1000]) > 0


def test_support_keeps_the_top_of_the_path_only(brief, part):
    assert part.keep.get_distance([2000, 1500, 600 - 200]) < 0
    assert part.keep.get_distance([2000, 1500, 600 - 400]) == pytest.approx(100.0)


@pytest.mark.parametrize("drawing", [CLIMB, Drawing.sample()], ids=["climb", "sample"])
def test_a_path_is_walkable_on_the_stand_in_roof(brief, drawing):
    solid = combine(steps.host(brief), [steps.circulation(brief, drawing)])
    counts = checks.path_clear(brief, steps.segments(brief, drawing), solid)
    assert counts["samples"] > 0 and counts["unsupported"] == 0 and counts["blocked"] == 0


def test_combine_lets_the_path_cut_another_modules_solid(brief, runs, part):
    planter = Contribution(add=nodes.sphere([2000, 1500, 600], 800.0))
    solid = combine(steps.host(brief), [planter, part])
    assert checks.path_clear(brief, runs, solid)["blocked"] == 0


def test_path_clear_finds_a_solid_added_after_the_cut(brief, runs, part):
    late = nodes.boolean([combine(steps.host(brief), [part]), nodes.sphere([2000, 1500, 600], 800.0)], "union", 0)
    assert checks.path_clear(brief, runs, late)["blocked"] > 0


def test_stairs_too_short_for_their_rise_are_flagged(brief):
    short = steps.segments(brief, straight((0, 0, 0), (2000, 0, 600)))
    assert checks.steep_stairs(brief, short) == short[0]
    assert checks.steep_stairs(brief, steps.segments(brief, Drawing.sample())) == []


def test_export_writes_a_closed_mesh_and_the_field(tmp_path, brief, part):
    coarse = Brief(voxel=150.0)
    solid = combine(steps.host(coarse), [part])
    vertices, faces = steps.export(coarse, solid, steps.extent(coarse, CLIMB), str(tmp_path / "roof"))
    assert len(faces) > 0 and is_watertight(vertices, faces)
    assert (tmp_path / "roof.stl").stat().st_size > 0
    again = compas.json_load(str(tmp_path / "roof.json"))
    assert again.get_distance([2000, 1500, 700]) == pytest.approx(solid.get_distance([2000, 1500, 700]))
