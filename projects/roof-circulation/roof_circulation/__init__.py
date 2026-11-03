"""Roof + circulation: walkable paths on any roof, written the way a Grasshopper definition is drawn.

* :class:`Brief` -- the sliders: every number, saved as JSON.
* :class:`Drawing` -- the referenced curves: path centre lines, saved as JSON.
* :mod:`roof_circulation.steps` -- the components: one function per step.
* :class:`Contribution`, :func:`combine` -- how this module meets the others.
* :mod:`roof_circulation.checks` -- the rules, measured.
* :mod:`roof_circulation.preview` -- the viewport.
"""

from . import checks, preview, steps
from .brief import Brief
from .contribution import Contribution, combine
from .drawing import Drawing

__all__ = ["Brief", "Drawing", "Contribution", "combine", "checks", "preview", "steps"]
