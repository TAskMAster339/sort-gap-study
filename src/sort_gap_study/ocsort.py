"""Адаптер OC-SORT к общему жизненному циклу треков."""

import sys
from importlib import import_module
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sort_gap_study.config import OCSortConfig, SortLifecycleConfig
from sort_gap_study.tracker import FrameEvents, TrackEstimate, validate_step
from sort_gap_study.upstream import extract_revision

OCSORT_REVISION = "8462e7e729a93ccd3bd995c0a79a890336cb3a0b"
DEFAULT_OCSORT_CHECKOUT = Path("external/OC_SORT")
_UPSTREAM_PATH = "trackers/ocsort_tracker"


class OCSortTracker:
    """Использовать ассоциацию и восстановление закреплённого OC-SORT."""

    def __init__(
        self,
        checkout: Path,
        config: OCSortConfig,
        lifecycle: SortLifecycleConfig,
    ) -> None:
        upstream = _load_upstream(checkout)
        self._engine = upstream.OCSort(
            det_thresh=0.5,
            delta_t=config.delta_t,
            inertia=config.inertia,
            iou_threshold=config.iou_threshold,
            max_age=lifecycle.max_age,
            min_hits=lifecycle.min_hits,
            use_byte=False,
        )
        self._lifecycle = lifecycle
        self._last_frame: int | None = None
        self._last_events = FrameEvents()

    @property
    def last_events(self) -> FrameEvents:
        return self._last_events

    def step(
        self,
        frame_index: int,
        detections: ArrayLike,
        *,
        artificial_gap: bool,
    ) -> tuple[TrackEstimate, ...]:
        frame_index, rows = validate_step(
            frame_index,
            self._last_frame,
            detections,
            artificial_gap=artificial_gap,
        )
        supplied = rows.copy()
        supplied[:, -1] = 1.0
        previous = {track.id for track in self._engine.trackers}
        # update() advances the age by one. Expire tracks before association.
        self._engine.trackers = [
            track
            for track in self._engine.trackers
            if track.time_since_update < self._lifecycle.max_age
        ]
        self._engine.update(supplied, (1, 1), (1, 1))

        alive = set()
        matched = []
        created = []
        estimates = []
        invalid = []
        for track in self._engine.trackers:
            alive.add(track.id)
            track_id = track.id + 1
            detection_index = track.detection_index
            if detection_index is not None:
                events = matched if track.id in previous else created
                events.append((track_id, detection_index))
            # The upstream hits counter excludes creation, then counts real updates.
            if track.hits + 1 >= self._lifecycle.min_hits:
                with np.errstate(invalid="ignore", divide="ignore"):
                    x1, y1, x2, y2 = track.get_state()[0].tolist()
                if not (all(map(isfinite, (x1, y1, x2, y2))) and x1 < x2 and y1 < y2):
                    invalid.append(track_id)
                    continue
                estimates.append(
                    TrackEstimate(
                        track_id,
                        (x1, y1, x2, y2),
                        detection_index is not None,
                    ),
                )
        self._last_events = FrameEvents(
            matched=tuple(matched),
            created=tuple(created),
            removed=tuple(track_id + 1 for track_id in sorted(previous - alive)),
            invalid_estimates=tuple(invalid),
        )
        self._last_frame = frame_index
        return tuple(estimates)


def _load_upstream(checkout: Path) -> ModuleType:
    # Each adapter has its own upstream class counter, including concurrent videos.
    with TemporaryDirectory(prefix="sort-gap-ocsort-") as directory:
        root = Path(directory)
        extract_revision(checkout, OCSORT_REVISION, _UPSTREAM_PATH, root)
        name = "_" + root.name.replace("-", "_")
        package = ModuleType(name)
        package.__path__ = [str(root / _UPSTREAM_PATH)]
        sys.modules[name] = package
        try:
            upstream = import_module(".ocsort", name)
        finally:
            for module in tuple(sys.modules):
                if module == name or module.startswith(name + "."):
                    sys.modules.pop(module, None)

    class TracedTrack(upstream.KalmanBoxTracker):
        def __init__(self, bbox: NDArray[np.float64], *, delta_t: int) -> None:
            super().__init__(bbox, delta_t=delta_t)
            self.detection_index = _row_index(bbox)

        def update(self, bbox: NDArray[np.float64] | None) -> None:
            super().update(bbox)
            self.detection_index = None if bbox is None else _row_index(bbox)

    upstream.KalmanBoxTracker = TracedTrack
    return upstream


def _row_index(bbox: NDArray[np.float64]) -> int:
    # The pinned update() passes row views of its score-filtered detection matrix.
    # With unit scores all rows survive in input order, including identical boxes.
    matrix = bbox.base
    if not isinstance(matrix, np.ndarray) or matrix.shape[1:] != bbox.shape:
        raise RuntimeError("OC-SORT должен передавать представление строки детекций.")
    return (bbox.ctypes.data - matrix.ctypes.data) // matrix.strides[0]
