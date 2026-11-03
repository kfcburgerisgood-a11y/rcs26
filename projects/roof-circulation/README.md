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
| `ramp_max`: 1:12, or gentler, never steeper (the brief refuses it) | path centre lines, in plan |
| no stairs: where 1:12 cannot be kept, `grade` says where to draw the route longer | pins: levels held where drawn, such as a lookout |
| a level landing `width / 2` long at every turn and end | `width` |
| `headroom` kept empty above, `support` kept solid below | `smoothing`: how closely the path follows the roof |
| both ends meet the roof's surface | `wave`, `wavelength`: a wave along the path, of any length |

The wave obeys 1:12 like everything else: it is lowered wherever it would be steeper. On a level stretch
that allows about `wavelength / 75` (80 mm for a 6 m wave).

## Steps

| Function | Makes |
|---|---|
| `host` | a stand-in rocky roof (`VolRock`); any field can replace it, a mesh through `VolGrid.from_mesh` |
| `slopes` | the roof's top and slope every `cell` mm: where a path could follow it as it is; holes marked |
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
