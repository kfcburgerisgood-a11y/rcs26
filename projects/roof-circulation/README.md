# roof-circulation

Owner: @kfcburgerisgood-a11y

Roof + circulation: paths people walk on, over any roof. Drawn centre lines become a walking
surface that ramps where it can and steps where it must, with headroom kept clear above it and the
ground under it protected. It does not design a roof: it hands its work to whatever roof it is put on.

```bash
uv run jupyter lab projects/roof-circulation/workflow.ipynb
uv run python -m pytest projects/roof-circulation
```

## Controlled and free

| Controlled: `Brief`, the rules every path keeps | Free: `Drawing`, the design |
|---|---|
| `width` clear walking width | path centre lines, in plan |
| `ramp_max` steepest ramp; steeper becomes stairs | the walking level at each vertex (its Z) |
| `riser` highest step, `going_min` shallowest | |
| `headroom` kept empty above, `support` kept solid below | |
| `depth` solid below a path's lowest level | |

## Steps

| Function | Makes |
|---|---|
| `segments` | each path as runs: a ramp, or stairs with a landing `width / 2` at each end |
| `walk` | the path as solid, down to `depth` below its lowest level |
| `clearance` | the headroom over the path, less the path |
| `support` | the top `support` mm under the walking surface |
| `circulation` | all three, as one `Contribution` |
| `extent`, `export` | the box to sample, and the mesh (`.stl`) and field (`.json`) in `out/` |

`checks.path_clear` walks every path on a finished solid (floor underfoot, nothing overhead), and
`checks.steep_stairs` lists flights too short for their rise.

## Putting modules together

Every module could hand over the same three fields, in world coordinates:

```python
Contribution(add=..., cut=..., keep=...)   # solid it puts in, space it needs empty, solid no process may remove
solid = combine(roof, [planting, drainage, circulation, ...])   # (roof + every add) - every cut
```

Cuts go after every add, so a planter placed on a path is cut back to the headroom; a process that
erodes constrains itself with the union of every `keep`. Run `checks.path_clear` on the combined solid.
`Contribution` and `combine` live in `roof_circulation/contribution.py` until the team agrees on them;
then they belong in compas_vol, so no project imports another.
