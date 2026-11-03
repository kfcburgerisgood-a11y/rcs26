"""How a module hands its work to a host, so modules built apart can be put together.

Every module returns a :class:`Contribution` of world-coordinate fields, and
:func:`combine` puts any number of them on one host in a fixed order: what
any module needs empty wins over what any module adds. Proposed as the shared
contract between the six modules; written here until it moves to compas_vol.
"""

from dataclasses import dataclass

from compas_vol import nodes

__all__ = ["Contribution", "combine"]


@dataclass
class Contribution:
    """What one module does to a host. Any of the three may be ``None``.

    Attributes
    ----------
    add : :class:`compas_vol.base.VolObject`
        Solid the module puts in.
    cut : :class:`compas_vol.base.VolObject`
        Space the module needs empty.
    keep : :class:`compas_vol.base.VolObject`
        Solid no later process (erosion, foam, weathering) may remove.
    """

    add: object = None
    cut: object = None
    keep: object = None


def combine(host, contributions):
    """``(host + every add) - every cut``. *Canvas: Volumetric Boolean x 2.*

    ``keep`` is not applied here: a process that removes material constrains
    itself with the union of every ``keep``.

    Examples
    --------
    >>> a = Contribution(add=nodes.sphere(None, 2.0), cut=nodes.sphere(None, 1.0))
    >>> solid = combine(nodes.box(None, 10.0, None, None, 0), [a])
    >>> round(solid.get_distance([0, 0, 0]), 6)
    1.0
    """
    solid = nodes.boolean([host] + [c.add for c in contributions], "union", 0)
    cuts = [c.cut for c in contributions if c.cut is not None]
    return nodes.boolean([solid] + cuts, "difference", 0) if cuts else solid
