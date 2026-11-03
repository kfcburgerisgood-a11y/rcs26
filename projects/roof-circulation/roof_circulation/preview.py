"""The preview: what a Rhino viewport would show, as plots."""

import numpy as np

__all__ = ["plan", "slope_map", "profile", "section"]


def slope_map(found, ramp_max, easy=1 / 20, contour=100.0, title="where a path could go"):
    """The host's plan by slope: easy, a ramp, too steep, or no host; with height contours every ``contour`` mm."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    classes = np.where(np.isnan(found.slope), 3, np.digitize(np.nan_to_num(found.slope), [easy, ramp_max]))
    colours = ["#cfe8c4", "#8cc084", "#d97a6c", "#ffffff"]
    _, ax = plt.subplots(figsize=(9, 6))
    extent = (found.x[0], found.x[-1], found.y[0], found.y[-1])
    ax.imshow(classes.T, origin="lower", extent=extent, cmap=ListedColormap(colours), vmin=0, vmax=3)
    levels = np.arange(np.floor(np.nanmin(found.height) / contour), np.ceil(np.nanmax(found.height) / contour) + 1)
    ax.contour(found.x, found.y, found.height.T, levels * contour, colors="0.3", linewidths=0.5)
    labels = ["easy, under 1:{:.0f}".format(1 / easy), "a ramp, to 1:{:.0f}".format(1 / ramp_max),
              "too steep to follow", "no roof"]
    ax.legend(handles=[Patch(color=c, label=t) for c, t in zip(colours, labels)], loc="upper right", fontsize=8)
    share = found.walkable(ramp_max).mean()
    ax.set_title("{}: {:.0%} walkable as it is".format(title, share))
    ax.set_aspect("equal")
    return ax


def plan(bounds, host, profiles, samples=200, title="paths on the host"):
    """The paths in plan over the host's top, shaded by height; pins as dots."""
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(9, 6))
    xs, ys = np.linspace(bounds[0], bounds[3], samples), np.linspace(bounds[1], bounds[4], samples)
    if hasattr(host, "top"):
        heights = host.top(xs[:, None], ys[None, :])
        image = ax.imshow(heights.T, origin="lower", cmap="Greys", extent=(xs[0], xs[-1], ys[0], ys[-1]))
        plt.colorbar(image, ax=ax, label="host top (mm)", shrink=0.7)
    for p in profiles:
        ax.plot(p.x, p.y, color="tab:blue", lw=2)
        for a, b, level in p.holds:
            if level is not None and a == b:
                ax.plot(np.interp(a, p.s, p.x), np.interp(a, p.s, p.y), "o", color="tab:red")
    ax.set_aspect("equal")
    ax.set_title(title)
    return ax


def profile(profiles, title="along each path"):
    """Each path unrolled: the host's surface under it, and the level walked on."""
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(figsize=(10, 3))
    for p in profiles:
        ax.fill_between(p.s, p.ground, min(p.ground.min(), p.level.min()) - 200, color="0.85", label="host")
        ax.plot(p.s, p.level, color="tab:blue", label="walking level")
        for a, b, level in p.holds:
            if level is not None and a == b:
                ax.plot(a, level, "o", color="tab:red")
    ax.set_xlabel("along the path (mm)")
    ax.set_ylabel("level (mm)")
    ax.legend(loc="upper left")
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
    ax.imshow((values < 0).T, origin="lower", cmap="Greys", extent=(us[0], us[-1], zs[0], zs[-1]), aspect="auto")
    ax.set_title("{} ({} = {:.0f} mm)".format(title, axis, at).strip())
    return ax
