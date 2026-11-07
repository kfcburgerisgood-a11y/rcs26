"""Roof + circulation: walkable paths on any roof, written the way a Grasshopper definition is drawn.

* :class:`Brief` -- the sliders: every number, saved as JSON; :class:`RestBrief`, the rest landings' own.
* :class:`Drawing` -- the referenced curves: path centre lines and pinned levels, saved as JSON.
* :mod:`roof_circulation.steps` -- the components: one function per step.
* :class:`Contribution`, :func:`combine` -- how this module meets the others.
* :mod:`roof_circulation.checks` -- the rules, measured.
* :mod:`roof_circulation.preview` -- the viewport.
* :class:`VolRock` -- a stand-in rocky roof.
"""

from . import checks, preview, steps
from .brief import RAMP_LIMIT, Brief, RestBrief
from .contribution import Contribution, combine
from .drawing import Drawing
from .rock import VolRock

__all__ = ["Brief", "RestBrief", "RAMP_LIMIT", "Drawing", "Contribution", "combine", "VolRock", "checks", "preview",
           "steps"]
