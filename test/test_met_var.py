from __future__ import annotations

from enum import Enum

import pytest  # type: ignore[import-not-found]

import metai.utils.met_var as met_var


def test_met_base_enum_is_str_and_carries_metadata() -> None:
    """
    MetBaseEnum should behave like a string and carry min/max/missing_value metadata.
    """
    x = met_var.MetRadar.CR

    assert isinstance(x, str)
    assert x == "CR"
    assert x.value == "CR"

    assert x.min == 0
    assert x.max == 800
    assert x.missing_value == -32768
    assert x.limits == {"min": 0, "max": 800, "missing_value": -32768}


def test_limits_is_none_when_min_or_max_is_missing() -> None:
    class _T(met_var.MetBaseEnum):
        NO_MIN = ("NO_MIN", None, 1, -9999)
        NO_MAX = ("NO_MAX", 0, None, -9999)

    assert _T.NO_MIN.limits is None
    assert _T.NO_MAX.limits is None


def test_parent_derivation_from_enum_class_name() -> None:
    assert met_var.MetLabel.RA.parent == "LABEL"
    assert met_var.MetRadar.CR.parent == "RADAR"
    assert met_var.MetNwp.CAPE.parent == "NWP"
    assert met_var.MetGis.LAT.parent == "GIS"


def test_metgis_missing_value_defaults_to_none_but_limits_still_present() -> None:
    x = met_var.MetGis.LAT

    assert x.value == "lat"
    assert x.min == 20
    assert x.max == 50
    assert x.missing_value is None
    assert x.limits == {"min": 20, "max": 50, "missing_value": None}


def test_all_enum_members_have_ordered_ranges_when_present() -> None:
    for enum_cls in (met_var.MetLabel, met_var.MetRadar, met_var.MetNwp, met_var.MetGis):
        assert issubclass(enum_cls, Enum)
        for m in enum_cls:
            if m.min is not None and m.max is not None:
                assert m.min <= m.max


def test_metvar_proxy_transparently_exposes_enum_members() -> None:
    # Access through proxy should return the underlying enum member.
    assert met_var.MetVar.RADAR.CR is met_var.MetRadar.CR
    assert met_var.MetVar.NWP.CAPE is met_var.MetNwp.CAPE
    assert met_var.MetVar.GIS.LON is met_var.MetGis.LON
    assert met_var.MetVar.LABEL.RA is met_var.MetLabel.RA


def test_metvar_proxy_repr_eq_hash() -> None:
    p = met_var.MetVar.RADAR
    q = met_var.MetVar._MetVarAttr(met_var.MetRadar, "RADAR")
    r = met_var.MetVar._MetVarAttr(met_var.MetRadar, "RADAR2")

    assert "RADAR" in repr(p)
    assert "MetRadar" in repr(p)

    assert p == q
    assert hash(p) == hash(q)

    assert p != r
    assert p != object()


def test_metvar_proxy_unknown_attribute_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        _ = met_var.MetVar.RADAR.NOT_A_MEMBER


