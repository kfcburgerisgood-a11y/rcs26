"""The workflow's steps: they run, and they keep their rules. Add a test for each rule you add."""

import compas
import pytest
from compas.geometry import Polyline

from compas_vol import nodes
from compas_vol.meshing import is_watertight
from roof_circulation import Brief, Contribution, Drawing, checks, combine, steps


def straight(*points):
    return Drawing([Polyline([list(p) for p in points])])


# the tests' own path, so the sample drawing can change: level, a 1:15 ramp, a turn, 4 steps up, a terrace
CLIMB = straight((1000, 1500, 600), (4500, 1500, 600), (7500, 1500, 800), (7500, 4500, 1400), (10500, 4500, 1400))
# and one that goes down: a 1:15 ramp to the deck, a turn, 3 steps down to 400 mm below it
DROP = straight((1000, 1500, 800), (4000, 1500, 600), (4000, 4500, 200), (7000, 4500, 200))


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
    gentle = steps.segments(brief, straight((0, 0, 0), (1200, 0, 100)))[0][0]
    steep = steps.segments(brief, straight((0, 0, 0), (3000, 0, 400)))[0][0]
    assert gentle.kind == "ramp"
    assert steep.kind == "stairs" and steep.risers == 3          # 400 / 150, rounded up
    assert steep.going == pytest.approx((3000 - brief.width) / 3)


def test_a_path_straight_up_is_refused(brief):
    with pytest.raises(ValueError, match="rises straight up"):
        steps.segments(brief, straight((0, 0, 0), (0, 0, 500)))


def test_the_walking_surface_is_where_the_drawing_puts_it(part):
    assert part.add.get_distance([2000, 1500, 700]) == pytest.approx(100.0)             # level, at 600
    assert part.add.get_distance([6000, 1500, 800]) == pytest.approx(100 / (1 + (1 / 15) ** 2) ** 0.5)  # ramp, 700
    assert part.add.get_distance([7500, 2775, 1150]) == pytest.approx(50.0)            # second step, 1100


def test_a_path_going_down_ramps_and_steps_like_one_going_up(brief):
    runs = steps.segments(brief, DROP)[0]
    assert [s.kind for s in runs] == ["ramp", "stairs", "ramp"]
    assert runs[1].risers == 3 and runs[1].going == pytest.approx(600.0)
    walk = steps.circulation(brief, DROP).add
    assert walk.get_distance([2500, 1500, 800]) == pytest.approx(100 / (1 + (1 / 15) ** 2) ** 0.5)  # ramp, 700
    assert walk.get_distance([4000, 3600, 200 + 400 / 3 + 50]) == pytest.approx(50.0)    # lowest step


def test_a_path_below_the_deck_cuts_its_own_trench(brief):
    solid = combine(steps.host(brief), [steps.circulation(brief, DROP)])
    assert solid.get_distance([5500, 4500, 500]) > 0          # over the sunken path: open
    assert solid.get_distance([5500, 3000, 500]) < 0          # beside it: still deck
    counts = checks.path_clear(brief, steps.segments(brief, DROP), solid)
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
