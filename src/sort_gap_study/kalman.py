"""Модель движения одного трека SORT на исходной шкале кадров."""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from sort_gap_study.config import SortKalmanConfig

_STATE_SIZE = 7
_MEASUREMENT_SIZE = 4
_AREA = 2
_AREA_VELOCITY = 6

_IDENTITY = np.eye(_STATE_SIZE)
_TRANSITION = _IDENTITY.copy()
_TRANSITION[0:3, 4:7] = np.eye(3)
_OBSERVATION = _IDENTITY[:_MEASUREMENT_SIZE]


def bbox_to_measurement(bbox: ArrayLike) -> NDArray[np.float64]:
    """Преобразовать (x1, y1, x2, y2) в (cx, cy, площадь, отношение сторон)."""
    corners = np.asarray(bbox, dtype=np.float64)
    if corners.shape != (_MEASUREMENT_SIZE,) or not np.isfinite(corners).all():
        raise ValueError("Ожидаются четыре конечные координаты bbox.")

    top_left, bottom_right = corners.reshape(2, 2)
    width, height = bottom_right - top_left
    if width <= 0 or height <= 0:
        raise ValueError("Ширина и высота bbox должны быть положительными.")

    cx, cy = (top_left + bottom_right) / 2
    return np.array([cx, cy, width * height, width / height])


class SortKalmanFilter:
    """Прогнозировать положение на один кадр и учитывать доступное наблюдение."""

    def __init__(self, bbox: ArrayLike, config: SortKalmanConfig) -> None:
        self._state = np.zeros(_STATE_SIZE)
        self._state[:_MEASUREMENT_SIZE] = bbox_to_measurement(bbox)
        self._covariance = np.diag(config.initial_covariance)
        self._process_noise = np.diag(config.process_noise)
        self._measurement_noise = np.diag(config.measurement_noise)
        self._bbox_epsilon = config.bbox_epsilon

    @property
    def bbox(self) -> NDArray[np.float64]:
        """Вернуть текущую оценку (x1, y1, x2, y2) отдельным массивом."""
        cx, cy, area, ratio = self._state[:_MEASUREMENT_SIZE]
        area = max(area, self._bbox_epsilon)
        ratio = max(ratio, self._bbox_epsilon)
        width = np.sqrt(area * ratio)
        height = area / width
        return np.array(
            [cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2],
        )

    def predict(self) -> None:
        """Выполнить переход на один исходный кадр."""
        if self._state[_AREA] + self._state[_AREA_VELOCITY] <= 0:
            self._state[_AREA_VELOCITY] = 0

        self._state = _TRANSITION @ self._state
        self._covariance = (
            _TRANSITION @ self._covariance @ _TRANSITION.T + self._process_noise
        )

    def correct(self, bbox: ArrayLike) -> None:
        """Скорректировать оценку на текущем кадре по наблюдаемому bbox."""
        innovation = bbox_to_measurement(bbox) - _OBSERVATION @ self._state
        cross_covariance = self._covariance @ _OBSERVATION.T
        innovation_covariance = (
            _OBSERVATION @ cross_covariance + self._measurement_noise
        )
        gain = np.linalg.solve(innovation_covariance, cross_covariance.T).T

        self._state += gain @ innovation
        residual = _IDENTITY - gain @ _OBSERVATION
        self._covariance = (
            residual @ self._covariance @ residual.T
            + gain @ self._measurement_noise @ gain.T
        )
