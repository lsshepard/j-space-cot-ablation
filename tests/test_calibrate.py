from jspace.calibrate import auto_select_band, select_band


def test_auto_select_rising_mid_band():
    # Early low, mid high, final spike.
    rates = [0.0, 0.0, 0.1, 0.4, 0.5, 0.55, 0.5, 0.2, 0.9, 0.95]
    start, end = auto_select_band(rates, n_layers=len(rates))
    assert start >= 1
    assert end < len(rates) - 2
    assert start <= end


def test_override_wins():
    rates = [0.0] * 12
    band = select_band(rates, 12, override_start=2, override_end=5)
    assert band.band_start == 2
    assert band.band_end == 5
    assert band.auto_selected is False
    assert band.strength_label == "medium_equivalent"


def test_fallback_middle_third():
    rates = [0.0] * 9
    start, end = auto_select_band(rates, 9)
    assert start == 3
    assert end == 6
