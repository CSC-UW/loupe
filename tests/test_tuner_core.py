"""Unit tests for the Qt-free Tuner core (:mod:`loupe.tuner`).

Covers the :class:`Param` family, the :class:`Tunable` wrapper / :func:`tunable`
factory (live value substitution, identity-based dependency tracking), bare-
callable param discovery via :func:`loupe.tuner._wrap_callable`, and the
:class:`Binding` / :func:`collect_params` helpers used by :func:`loupe.view`.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from loupe.tuner import (
    Binding,
    BoolParam,
    ChoiceParam,
    IntParam,
    Param,
    _wrap_callable,
    collect_params,
    tunable,
)


# --------------------------------------------------------------------------- #
# Param family
# --------------------------------------------------------------------------- #
def test_param_value_get_set_reset():
    p = Param(0.15, 0.01, 1.0, name="tau")
    assert p.value == 0.15
    assert p.default == 0.15 and p.min == 0.01 and p.max == 1.0
    p.value = 0.4
    assert p.value == 0.4
    p.reset()
    assert p.value == 0.15


def test_param_identity_not_value_equality():
    # Two params with identical fields are distinct knobs (identity, not value).
    a = Param(1.0, name="x")
    b = Param(1.0, name="x")
    assert a is not b
    assert a != b


def test_int_param_coerces_to_rounded_int():
    p = IntParam(3, 1, 10, name="n")
    assert isinstance(p.value, int) and p.value == 3
    p.value = 4.7
    assert p.value == 5


def test_bool_param_coerces():
    p = BoolParam(True, name="flag")
    assert p.value is True
    p.value = 0
    assert p.value is False


def test_choice_param_default_and_validation():
    p = ChoiceParam(["peak", "mean"], name="m")
    assert p.value == "peak"  # default = first choice
    p.value = "mean"
    assert p.value == "mean"
    with pytest.raises(ValueError):
        p.value = "nope"
    with pytest.raises(ValueError):
        ChoiceParam([], name="empty")
    with pytest.raises(ValueError):
        ChoiceParam(["a", "b"], default="z")


# --------------------------------------------------------------------------- #
# Tunable / tunable()
# --------------------------------------------------------------------------- #
def _f(x, tau_s=0.0, k=1):
    return (x, tau_s, k)


def test_tunable_substitutes_value_and_holds_non_params():
    tau = Param(0.15, name="tau")
    t = tunable(_f, "RAW", tau_s=tau, k=2)
    assert t.params == [tau]
    assert t() == ("RAW", 0.15, 2)
    tau.value = 0.9  # read live on the next call
    assert t() == ("RAW", 0.9, 2)


def test_tunable_dedups_params_by_identity():
    tau = Param(0.15, name="tau")
    t = tunable(_f, tau, tau_s=tau)
    assert t.params == [tau]


def test_tunable_explicit_form_does_not_scan_globals():
    # An explicit tunable(func, ...) must derive params solely from its args,
    # never from func's module globals.
    assert tunable(_f).params == []


def test_tunable_rejects_non_callable():
    with pytest.raises(TypeError):
        tunable(123)


def test_wrap_callable_discovers_global_param():
    # Bare lambda referencing a module-global Param (the notebook case).
    tau = Param(0.2, name="tau_g")
    t = _wrap_callable(lambda: _f("L", tau_s=tau.value))
    assert t.params == [tau]
    assert t() == ("L", 0.2, 1)


def test_wrap_callable_discovers_closure_param():
    def make():
        local = Param(0.05, name="local")
        return _wrap_callable(lambda: _f("C", tau_s=local.value)), local

    t, local = make()
    assert t.params == [local]


# --------------------------------------------------------------------------- #
# Binding / collect_params
# --------------------------------------------------------------------------- #
def test_binding_params_delegate_to_tunable():
    tau = Param(0.1, name="tau")
    b = Binding(kind="trace_overlay", tunable=tunable(_f, tau_s=tau), cfg=None,
                overlay_k=0)
    assert b.params == [tau]


def test_collect_params_dedups_in_first_seen_order():
    tau = Param(0.1, name="tau")
    other = Param(2.0, name="other")
    b1 = Binding(kind="trace_overlay", tunable=tunable(_f, tau_s=tau), cfg=None)
    b2 = Binding(kind="trace_stacked", tunable=tunable(_f, "R", k=other), cfg=None)
    b3 = Binding(kind="trace_overlay", tunable=tunable(_f, tau_s=tau), cfg=None)
    assert collect_params([b1, b2, b3]) == [tau, other]
