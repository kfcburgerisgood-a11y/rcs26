# roof-circulation: handoff and working notes

Read the repository's root `AGENTS.md` and `CONTRIBUTING.md` first; everything there applies. This file is
what an agent picking this project up needs on top: what exists, why, what is next, and how to work with the
owner. It was written at a handoff between agents (Claude Code to Codex), on 2026-11-05.

## The project and the people

- RCS26 studio: six students, one module each, built on `compas_vol` as signed distance fields, to be
  integrated into one roof later. Modules: roof + planting, roof + drainage, **roof + circulation (this one)**,
  roof + daylight + openings, wall + daylight + openings, wall + planting.
- Owner: GitHub `kfcburgerisgood-a11y` (git author name the same, email `kfcburgerisgood@gmail.com`, set
  repo-local). The owner writes in Traditional Chinese; answer in Traditional Chinese, code and docs in English.
- The teacher's brief: modules are general, not for one design; decide what is controlled (rules) and what is
  free (design); code is written so steps can become Grasshopper components later (parked: build none unless asked).
- The owner decides design questions. Ask (with a recommendation) before changing a rule or a default they set.

## Git workflow (agreed with the owner)

- Branch: `kfcburgerisgood-a11y/roof-circulation`. Remotes: `origin` = rccn-dev/rcs26 (no write access, fetch
  only), `fork` = kfcburgerisgood-a11y/rcs26 (push here; upstream is set, so `git push` goes to the fork).
- Commit and push **only when the owner asks**. Commit with `uv run git commit ...` (the pre-commit hook needs
  the uv environment); if the hook strips the notebook, `git add` it and commit again. End messages with the
  attribution line the session asks for.
- No pull request is open. When the owner asks for one: from the fork to `rccn-dev/rcs26` `main`; the description
  says AI was used and that `uv.lock` changed only to add this project (needs the maintainer's review).
- State at handoff: commits up to `5b4cabb` are pushed. **The rest-landing work is uncommitted** (brief.py,
  steps.py, routing.py, checks.py, drawing.py, preview.py, tests, notebook, README, this file). 148 tests pass.

## Commands

```bash
uv run ruff check .
uv run python -m pytest projects/roof-circulation          # ~10 s
uv run python -m pytest                                    # everything, ~35 s
uv run python scripts/run_notebooks.py roof-circulation    # the notebook, top to bottom, ~12 s
uv run nbstripout projects/roof-circulation/workflow.ipynb
```

The notebook was generated with `nbformat` from a script that is not in the repo; edit `workflow.ipynb`
directly (or rebuild it with `nbformat`). Cells call steps and show results; they define no functions or
classes (a repo test enforces it). Run it after every change to it.

## The pipeline (one function per step: each a future Grasshopper component)

| Step | Function | File | In -> out |
|---|---|---|---|
| 1 | `Brief` | `brief.py` | every number, mm, defaults; validates `ramp_max <= 1/12`, `width >= 1200`, `0 <= attract < 1` |
| 2 | `host` | `steps.py` | stand-in rocky roof `VolRock` (`rock.py`, fbm noise); any field can replace it (`VolGrid.from_mesh`) |
| 2 | `slopes` | `steps.py` | host -> `SlopeMap` (top and slope every `cell`; holes NaN) |
| 3a | drawn paths | `drawing.py` | `Drawing.paths` (plan polylines, Z ignored) + `pins` (points holding a level) |
| 3b | `route` | `routing.py` | `Drawing.stops` (start, via, end) -> winding paths; `turns`, `swing`, `bend`, attractors/repellers |
| 3c | `network` | `routing.py` | `Drawing.entrances` -> branching paths; `branches`, `edge_clearance` |
| 4 | `ground` | `steps.py` | paths -> `Profile`s: host surface under each path; holds (level landings) at ends, turns, forks, pins |
| 4 | `grade` | `steps.py` | walking level: ground averaged over `smoothing` + `wave`, within `ramp_max`, through holds; raises "Too steep ... Draw it longer there" with the place |
| 4b | `rest` | `steps.py` | rest landings + bays (`rest_rise`, `rest_length`, `rest_bay`); optional |
| 5 | `segments` | `steps.py` | profiles -> straight pieces (`Segment`), plus one list per rest bay |
| 5 | `circulation` | `steps.py` | -> `Contribution(add, cut, keep)` (`contribution.py`); `combine(host, [...])` |
| 6 | checks | `checks.py` | `steepest`, `cut_and_fill`, `path_clear` (floor underfoot, nothing overhead, no lip over 10 mm) |
| 7 | `extent`, `export` | `steps.py` | sampling box; STL + field JSON into `out/` (git-ignored) |
| - | previews | `preview.py` | `slope_map`, `plan`, `profile`, `section` |

All three path sources (3a drawn, 3b route, 3c network) produce `Drawing.paths` and share steps 4 to 7.
**Keep all three**: the owner wants the free-drawn path with its own turning points and pinned heights as an
alternative to the generated ones.

## Rules and why (do not loosen without asking)

- 1:12 is the steepest a path ever is, and **there are no stairs** (accessibility). `ramp_max` may be gentler.
- Every turn sharper than `GENTLE` (8 degrees) and every end and fork is held level as far as the arms overlap
  (`width/2 * max(1, tan(turn/2))`); gentler bends climb as they turn, pieces mitred. 8 degrees keeps the step on a
  bend's inside under ~7 mm (slope * width/2 * sin(bend)); `path_clear` allows 10 mm.
- Routing searches cell + heading (`_graph`): every turn costs `TURN_COST * width * (1 - cos)`. Measured: below
  ~0.4 a route zigzags a cell wide and smoothing leaves it straight up the slope; 1.0 is fixed, not a parameter.
- `_wind` smooths over `bend`, mirrored past the ends (no kink at ends), resampled every `width/6`.
- `route` pulls each stretch taut between swings (`_pull`) using the same cost, including attractors.
- An attractor within `influence` of a swing captures it; otherwise it only makes nearby steps cheaper.
- Rests: counted along one continuous climb (reset at a reversal or a 1:50-or-gentler stretch >= `rest_length`);
  budgeted between fixed levels (ends, pins) in ramp length (landings excluded); as few as divide the climb, at
  evenly divided levels; by an attractor if near, else the flattest ground; bay on the side with less cut/fill.

## Testing standards

- A failing test first for every bug; a test for every rule. Assert known distances at known points, mostly on
  the flat stand-in (`Brief(roughness=0.0)`) where levels are exact.
- Never weaken a test to pass; if a threshold was wrong, say why in the reply (the owner reads them).
- Before saying something works: ruff, the project tests, the notebook; for geometry, render it and look.

## Next tasks (the owner's latest requests, not started)

1. **Make rest landings a separate, optional step with its own numbers.** `rest()` is already its own step, but
   `rest_rise`, `rest_length`, `rest_bay` live in `Brief` beside `width`. The owner wants them apart: `width` is set
   in the first step; the rest step is one component that can be left out. Suggested: a small `RestBrief`
   dataclass (same pattern as `Brief`: defaults, `to_dict`/`from_dict`/validation) passed to `rest()`, rest fields
   removed from `Brief`; update `network` (it calls `rest` in its walkability check: take it as optional),
   tests, notebook (Step 4b), README. Confirm the design with the owner before changing signatures.
2. **Keep the free-drawn path branch visible**: the notebook's Step 3 shows only `route`; add a short alternative
   (drawn `paths` + `pins`, as `Drawing.sample()`), so all three path sources are shown.
3. Known limits, ask before fixing: `turns` with a middle stop (`Drawing.sample_lookout`) crowds swings round the
   stop and runs out of ramp; `route` on `sample_stops` with `turns` can box the way back in; end and turn landings
   are `width/2` long while the code asks 1500 x 1500 (offered, owner has not decided); default `width` is 1200,
   the code asks 1500 where there is no stair (the notebook uses 1500).
4. Later, only when asked: Grasshopper components (follow `projects/roof/docs/development.md`); a PR description.
