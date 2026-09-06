"""User-facing primitives for the Loupe **Tuner** — live parameter tuning.

This module is deliberately **Qt-free and cheap to import**, like
:mod:`loupe.configs` and :mod:`loupe.series`.  ``from loupe import Param,
tunable`` must work inside a notebook or script before any ``QApplication``
exists; the GUI panel that renders these knobs lives in
:mod:`loupe.tuner_panel` and is only imported when a window is built.

The two primitives a user touches:

- :class:`Param` — a live, tunable scalar with a ``.value`` you can read back
  after tuning.  Subclasses :class:`IntParam`, :class:`BoolParam`,
  :class:`ChoiceParam` pick the appropriate panel widget.
- :func:`tunable` — wraps a *pure* function plus its arguments.  Any argument
  that is a :class:`Param` becomes a tuned dependency; calling the wrapper runs
  the function with each :class:`Param` replaced by its current value.

Drop a :class:`Tunable` (or a bare zero-arg callable closing over
:class:`Param` objects) into an array-bearing config slot — e.g.
:attr:`loupe.configs.TraceConfig.overlay_arrays` or a stacked
:class:`loupe.configs.SampleMarkers` mask — and Loupe re-evaluates it live as
you drag the matching slider::

    from loupe import view, TraceConfig, Param, tunable
    import wisco_slap as wis

    raw_ls = ls.sel(syn_id=example_syns)
    tau = Param(0.15, 0.01, 1.0, name="tau")          # default, min, max
    view(TraceConfig(
        data=raw_ls,
        overlay_arrays=[tunable(wis.scope.pro.ls_to_matched_filter,
                                raw_ls, tau_s=tau)],
    ))
    # Tuner dock opens automatically; drag `tau`, watch the overlay redraw.
    # Afterwards `tau.value` holds your chosen number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "Param",
    "IntParam",
    "BoolParam",
    "ChoiceParam",
    "Tunable",
    "tunable",
    "Binding",
    "collect_params",
]


class Param:
    """A live, tunable scalar.

    Parameters
    ----------
    default :
        Initial value; ``.value`` returns this until the user changes it.
    min, max : optional
        Slider bounds.  When either is ``None`` the panel renders a spin box
        only (no slider) — the value stays editable but unbounded.
    step : optional
        Spin-box increment.  ``None`` lets the panel pick a sensible default.
    name : str, optional
        Display label in the Tuner panel.

    Notes
    -----
    A :class:`Param` is tracked by **identity**, not by name.  Reuse the *same*
    instance across several :func:`tunable` calls and one slider drives them
    all; two distinct :class:`Param` objects with the same ``name`` are two
    separate knobs.
    """

    def __init__(
        self,
        default: Any,
        min: Any = None,
        max: Any = None,
        step: Any = None,
        name: str | None = None,
    ) -> None:
        self.default = default
        self.min = min
        self.max = max
        self.step = step
        self.name = name
        self._value = self._coerce(default)

    def _coerce(self, v: Any) -> Any:
        """Hook for subclasses to constrain/cast an incoming value."""
        return v

    @property
    def value(self) -> Any:
        """The current value (live)."""
        return self._value

    @value.setter
    def value(self, v: Any) -> None:
        self._value = self._coerce(v)

    def reset(self) -> None:
        """Restore :attr:`value` to the original :attr:`default`."""
        self._value = self._coerce(self.default)

    @property
    def label(self) -> str:
        """Human-readable label for the panel (falls back to a generic name)."""
        return self.name if self.name else f"param@{id(self):x}"

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"{type(self).__name__}(name={self.name!r}, value={self._value!r}, "
            f"default={self.default!r}, min={self.min!r}, max={self.max!r})"
        )


class IntParam(Param):
    """A :class:`Param` constrained to integers (rendered with an int spin box)."""

    def __init__(
        self,
        default: int,
        min: int | None = None,
        max: int | None = None,
        step: int = 1,
        name: str | None = None,
    ) -> None:
        super().__init__(int(default), min, max, step, name)

    def _coerce(self, v: Any) -> int:
        return int(round(v))


class BoolParam(Param):
    """A :class:`Param` constrained to a boolean (rendered as a check box)."""

    def __init__(self, default: bool = False, name: str | None = None) -> None:
        super().__init__(bool(default), name=name)

    def _coerce(self, v: Any) -> bool:
        return bool(v)


class ChoiceParam(Param):
    """A :class:`Param` over a fixed set of choices (rendered as a combo box).

    Parameters
    ----------
    choices :
        The allowed values (any hashable/comparable objects).
    default : optional
        Initial choice; defaults to ``choices[0]``.
    name : str, optional
        Display label.
    """

    def __init__(
        self,
        choices: list,
        default: Any = None,
        name: str | None = None,
    ) -> None:
        choices = list(choices)
        if not choices:
            raise ValueError("ChoiceParam requires at least one choice.")
        self.choices = choices
        if default is None:
            default = choices[0]
        elif default not in choices:
            raise ValueError(
                f"ChoiceParam default {default!r} is not in choices {choices!r}."
            )
        super().__init__(default, name=name)

    def _coerce(self, v: Any) -> Any:
        if v not in self.choices:
            raise ValueError(f"{v!r} is not a valid choice (choices={self.choices!r}).")
        return v


def _collect_param(val: Any, seen: dict[int, Param]) -> None:
    """Record ``val`` if it's a :class:`Param`, scanning one level into
    list/tuple containers."""
    if isinstance(val, Param):
        seen.setdefault(id(val), val)
    elif isinstance(val, (list, tuple)):
        for x in val:
            if isinstance(x, Param):
                seen.setdefault(id(x), x)


def _params_in_callable(func: Callable) -> list[Param]:
    """Best-effort discovery of :class:`Param`s a *bare* callable reads.

    Lets a bare ``lambda: f(raw, tau_s=tau.value)`` still be tuned — the panel
    needs to know which knobs the callable depends on.  Two sources:

    1. **Closure free variables** — params captured from an enclosing function.
    2. **Referenced module/notebook globals** — names the callable's bytecode
       actually references (``co_names``) that resolve to a :class:`Param` in
       its ``__globals__``.  This is what makes the lambda path work from a
       notebook cell, where ``tau`` is a module global rather than a closure
       cell.

    Discovery is conservative (only :class:`Param` instances actually
    referenced by the callable), but for reliability prefer the explicit
    :func:`tunable` form.
    """
    seen: dict[int, Param] = {}
    for cell in getattr(func, "__closure__", None) or ():
        try:
            _collect_param(cell.cell_contents, seen)
        except ValueError:  # empty cell (recursive defn not yet bound)
            continue
    code = getattr(func, "__code__", None)
    g = getattr(func, "__globals__", None)
    if code is not None and isinstance(g, dict):
        for name in getattr(code, "co_names", ()):
            if name in g:
                _collect_param(g[name], seen)
    return list(seen.values())


class Tunable:
    """A deferred call ``func(*args, **kwargs)`` whose :class:`Param` arguments
    are read live at call time.

    Any positional or keyword argument that is a :class:`Param` is recorded as a
    tuned dependency.  Calling the :class:`Tunable` substitutes each such
    argument with its current :attr:`Param.value` and invokes ``func``; non-Param
    arguments pass through unchanged and are reused on every call (fixed inputs
    are never recomputed or re-read from disk).

    The wrapped function **must be pure** — it is called once for the initial
    render and again on every dependency change, so it must not mutate its
    inputs.  The wisco-slap processing functions satisfy this.
    """

    def __init__(self, func: Callable, *args: Any, **kwargs: Any) -> None:
        if not callable(func):
            raise TypeError(
                f"tunable() first argument must be callable, got "
                f"{type(func).__name__}."
            )
        self.func = func
        self.args = args
        self.kwargs = kwargs
        # Set True only when this wraps a *bare* slot callable (no explicit
        # Param args), so params are discovered by scanning the callable's
        # closure/globals. Never set for an explicit ``tunable(func, ...)`` —
        # there params come solely from the given args/kwargs.
        self._scan_callable = False

    @property
    def params(self) -> list[Param]:
        """The :class:`Param` dependencies, de-duplicated by identity.

        Order is: positional args, then keyword args, then (only for a wrapped
        bare callable) any closure/global params the callable references.
        """
        seen: dict[int, Param] = {}
        for v in self.args:
            if isinstance(v, Param):
                seen.setdefault(id(v), v)
        for v in self.kwargs.values():
            if isinstance(v, Param):
                seen.setdefault(id(v), v)
        if self._scan_callable:
            for p in _params_in_callable(self.func):
                seen.setdefault(id(p), p)
        return list(seen.values())

    def __call__(self) -> Any:
        args = [v.value if isinstance(v, Param) else v for v in self.args]
        kwargs = {
            k: (v.value if isinstance(v, Param) else v)
            for k, v in self.kwargs.items()
        }
        return self.func(*args, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        fn = getattr(self.func, "__name__", repr(self.func))
        return f"Tunable({fn}, params={[p.name for p in self.params]})"


def tunable(func: Callable, *args: Any, **kwargs: Any) -> Tunable:
    """Wrap a pure function and its arguments into a :class:`Tunable`.

    Pass :class:`Param` instances as the arguments you want to tune::

        tau = Param(0.15, 0.01, 1.0, name="tau")
        mf = tunable(wis.scope.pro.ls_to_matched_filter, raw_ls, tau_s=tau)
        view(TraceConfig(data=raw_ls, overlay_arrays=[mf]))

    Calling ``mf()`` runs the function with ``tau_s=tau.value``.  Loupe calls it
    for you whenever the matching slider moves.
    """
    return Tunable(func, *args, **kwargs)


def _wrap_callable(fn: Callable) -> Tunable:
    """Wrap a *bare* zero-arg slot callable (e.g. a notebook ``lambda``) into a
    :class:`Tunable` whose params are discovered from the callable itself.

    Used by :func:`loupe.view` when a config slot holds a plain callable rather
    than an explicit :func:`tunable` wrapper.
    """
    t = Tunable(fn)
    t._scan_callable = True
    return t


@dataclass
class Binding:
    """Internal: links a :class:`Tunable` in a config slot to the runtime
    containers / plot items it produced.

    Created by :func:`loupe.view`; consumed by :class:`loupe.app.LoupeApp`'s
    live-update path.  ``kind`` selects which update path runs; only the locator
    fields relevant to that kind are populated.  Each config contributes a
    *contiguous* block to each runtime registry, so a slot is addressed by a
    ``slice`` into ``LoupeApp.series`` / ``LoupeApp.overlay_series``.
    """

    kind: str  # "trace_stacked" | "trace_overlay" | "trace_marker" | "raster"
    #            | "interval_labels"
    tunable: Tunable
    cfg: Any  # the originating Config — carries order_by / descending / hue;
    # for "interval_labels" it is the IntervalLabelSchema of the returned frame
    # locators (only the ones for `kind` are set) ------------------------------
    series_slice: slice | None = None  # trace_stacked → app.series / app.curves;
    # raster → app.raster_series
    overlay_host_slice: slice | None = None  # trace_overlay → host-series indices
    overlay_k: int | None = None  # trace_overlay → which overlay_arrays column
    host_data: Any = None  # trace_overlay/trace_marker → host DataArray
    marker_host_slice: slice | None = None  # trace_marker → host-series indices
    marker_k: int | None = None  # trace_marker → which sample-marker set

    @property
    def params(self) -> list[Param]:
        return self.tunable.params


def collect_params(bindings: list[Binding]) -> list[Param]:
    """All :class:`Param`s referenced by ``bindings``, de-duplicated by identity
    in first-seen order — the panel renders one control per returned param."""
    seen: dict[int, Param] = {}
    for b in bindings:
        for p in b.params:
            seen.setdefault(id(p), p)
    return list(seen.values())
