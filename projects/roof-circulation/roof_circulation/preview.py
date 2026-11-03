"""The preview: what a Rhino viewport would show, as plots."""

import numpy as np

__all__ = ["plan", "section"]


def plan(bounds, segments, title="paths"):
    """The paths in plan: ramps in blue, stairs in red with a tick at every riser."""
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(8, 5))
    for runs in segments:
        for seg in runs:
            xs, ys = [seg.start[0], seg.end[0]], [seg.start[1], seg.end[1]]
            ax.plot(xs, ys, color="tab:red" if seg.risers else "tab:blue", lw=2)
            (x, y, _), (ux, uy) = seg.low_end()
            for k in range(seg.risers):
                u = seg.landing + k * seg.going
                ax.plot(x + ux * u, y + uy * u, "|", color="tab:red", ms=12)
    ax.set_xlim(bounds[0], bounds[3])
    ax.set_ylim(bounds[1], bounds[4])
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax


def section(bounds, solid, axis="y", at=0.0, samples=400, title=""):
    """A section at constant X or Y: solid in black."""
    import matplotlib.pyplot as plt

    i = 0 if axis == "y" else 1
    us = np.linspace(bounds[i], bounds[i + 3], samples)
    zs = np.linspace(bounds[2], bounds[5], max(2, int(samples * (bounds[5] - bounds[2]) / (bounds[i + 3] - bounds[i]))))
    fixed = np.full((1, 1, 1), float(at))
    if axis == "y":
        values = solid.get_distance_numpy(us[:, None, None], fixed, zs[None, None, :])[:, 0, :]
    else:
        values = solid.get_distance_numpy(fixed, us[None, :, None], zs[None, None, :])[0, :, :]
    _, ax = plt.subplots(figsize=(10, 3))
    ax.imshow((values < 0).T, origin="lower", cmap="Greys", extent=(us[0], us[-1], zs[0], zs[-1]))
    ax.set_title("{} ({} = {:.0f} mm)".format(title, axis, at).strip())
    return ax
