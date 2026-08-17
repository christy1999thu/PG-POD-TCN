import numpy as np

from pg_pod_tcn.ood import MahalanobisDetector


def test_mahalanobis_detector_flags_far_sample(tmp_path):
    rng = np.random.default_rng(8)
    train = rng.normal(size=(200, 4))
    detector = MahalanobisDetector(quantile=0.99).fit(train)
    original_threshold = detector.threshold_
    detector.calibrate(rng.normal(loc=0.5, size=(50, 4)))
    assert detector.threshold_ >= original_threshold
    assert not detector.predict(np.zeros((1, 4)))[0]
    assert detector.predict(np.full((1, 4), 20.0))[0]
    path = tmp_path / "detector.npz"
    detector.save(path)
    loaded = MahalanobisDetector.load(path)
    np.testing.assert_allclose(detector.score(train[:5]), loaded.score(train[:5]))
