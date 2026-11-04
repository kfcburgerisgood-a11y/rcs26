# roof-circulation

Owner: @kfcburgerisgood-a11y

Roof + circulation: a path people walk on, over any roof, flat or as irregular as rock. You draw where it
goes in plan; the module finds the roof's surface under it and lays a walking level on it that is never
steeper than 1:12, with no stairs, so it can be walked and wheeled all the way. It does not design a roof:
it reads whatever roof it is put on, and hands its work back.

```bash
uv run jupyter lab projects/roof-circulation/workflow.ipynb
uv run python -m pytest projects/roof-circulation
```

## Controlled and free

| Controlled: `Brief`, the rules every path keeps | Free: `Drawing` and the tuning numbers |
|---|---|
| `ramp_max`: 1:12, or gentler, never steeper (the brief refuses it) | entrances: a network grows from them; or stops to route through; or paths drawn yourself |
| every branch end keeps `edge_clearance` from the roof's edge and holes | `branches`: how many ends a network grows to |
| no stairs: where 1:12 cannot be kept, `grade` says where to draw the route longer | pins: levels held where drawn, such as a lookout |
| a level landing at every end, fork and turn sharper than 8 degrees: `width / 2`, longer where paths meet sharply; a gentler bend climbs on as it turns | `avoid_slope`: how strongly a route keeps to gentle ground |
| a route never runs over itself, and keeps `width / 2` clear of holes and edges | `turns`: how many times a route swings side to side between stops: 0 straight, more winds more |
| a route turns no more than it gains by: every turn costs walking | `swing`: how far each swing goes aside |
| | `bend`: the length bends are smoothed over: longer sweeps, shorter hugs the ground |
| | attractors and repellers: places a route bends towards or keeps from, within `influence` (`attract`, `repel`); an attractor near a swing captures it |
| | `width` |
| `headroom` kept empty above, `support` kept solid below | `smoothing`: how closely the path follows the roof |
| both ends meet the roof's surface | `wave`, `wavelength`: a wave along the path, of any length |

The wave obeys 1:12 like everything else: it is lowered wherever it would be steeper. On a level stretch
that allows about `wavelength / 75` (80 mm for a 6 m wave).

## Steps

| Function | Makes |
|---|---|
| `host` | a stand-in rocky roof (`VolRock`); any field can replace it, a mesh through `VolGrid.from_mesh` |
| `slopes` | the roof's top and slope every `cell` mm: where a path could follow it as it is; holes marked |
| `route` | each set of stops as a path: the cheapest way, where a step costs more the steeper it climbs in the direction it goes and the further it turns, so it crosses slopes along their contours in long legs; swinging `turns` times between stops through the gentlest ground near each swing; pulled taut between swings, and drawn as one smooth winding line |
| `network` | from entrances alone, a walk branching out over the gentle ground: each end far from what is walked, each branch leaving the walk where cheapest, split at every fork, and kept only if walkable at 1:12 |
| `ground` | the roof's surface under every path, followed every `spacing` mm |
| `grade` | the walking level: the surface averaged over `smoothing`, plus the wave, within 1:12, meeting the roof at both ends, through every pin, level at every turn |
| `segments` | the graded path as straight pieces, each reaching `depth` into the roof below it |
| `circulation` | the path (add), the headroom over it (cut) and the top of the path (keep), as one `Contribution` |
| `extent`, `export` | the box to sample, and the mesh (`.stl`) and field (`.json`) in `out/` |

Checks: `steepest` (never over `ramp_max`), `cut_and_fill` (how far the path leaves the roof's surface), and
`path_clear`, which walks every path on a finished solid: floor underfoot, nothing overhead, not even a 10 mm lip.

## Putting modules together

Every module could hand over the same three fields, in world coordinates:

```python
Contribution(add=..., cut=..., keep=...)   # solid it puts in, space it needs empty, solid no process may remove
solid = combine(roof, [planting, drainage, circulation, ...])   # (roof + every add) - every cut
```

Cuts go after every add, so a planter placed on a path is cut back to the headroom; a process that
erodes constrains itself with the union of every `keep`. This module also reads the roof: give `ground`
the combined roof of the modules that shape the surface, then run `checks.path_clear` on the result.
`Contribution` and `combine` live in `roof_circulation/contribution.py` until the team agrees on them;
then they belong in compas_vol, so no project imports another.
