import numpy as np

from commentator.core.pitch_contour import PitchContour


def test_pitch_contour_can_be_instantiated():
    contour = PitchContour(
        times=np.array([0.0, 0.1, 0.2]),
        frequencies=np.array([120.0, 121.5, 123.0]),
        voiced_mask=np.array([True, False, True]),
    )

    assert contour.duration == 0.2
    assert contour.voiced_times.tolist() == [0.0, 0.2]
    assert contour.voiced_frequencies.tolist() == [120.0, 123.0]
