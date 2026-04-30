"""Helpers for preserving Matplotlib axis view limits across redraws."""

from dataclasses import dataclass


@dataclass(frozen=True)
class AxisViewState:
    """Stored x/y limits for a Matplotlib axis."""

    xlim: tuple[float, float]
    ylim: tuple[float, float]


def capture_axis_view(axis) -> AxisViewState | None:
    """Return current axis limits when there is plotted data to preserve."""
    if axis is None or not axis.has_data():
        return None

    return AxisViewState(xlim=tuple(axis.get_xlim()), ylim=tuple(axis.get_ylim()))


def restore_axis_view(axis, state: AxisViewState | None) -> None:
    """Restore previously captured axis limits."""
    if axis is None or state is None:
        return

    axis.set_xlim(state.xlim)
    axis.set_ylim(state.ylim)
