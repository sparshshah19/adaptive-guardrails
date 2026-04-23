# tests/test_classifier.py
# Tests the RiskClassifier cold-start and threshold logic.
#
# Why no real model loaded here?
# Training requires 10k samples and ~2 minutes of compute.
# Unit tests must be fast and run offline. We test the contract:
#   - is_ready() returns False before load()
#   - predict() raises ClassifierNotReady when not ready
#   - predict() applies the right thresholds when given a mock model
#   - load() is a no-op when the file doesn't exist

import pytest
import torch
from unittest.mock import MagicMock, patch
from classifier import RiskClassifier, ClassifierNotReady, _RiskNet


class TestRiskClassifierColdStart:
    def test_not_ready_without_encoder(self):
        """Classifier with no encoder is not ready."""
        clf = RiskClassifier(encoder=None)
        assert clf.is_ready() is False

    def test_not_ready_without_load(self):
        """Classifier with encoder but no loaded weights is not ready."""
        mock_encoder = MagicMock()
        clf = RiskClassifier(encoder=mock_encoder)
        assert clf.is_ready() is False

    def test_predict_raises_when_not_ready(self):
        """predict() must raise ClassifierNotReady before model is loaded."""
        clf = RiskClassifier(encoder=None)
        with pytest.raises(ClassifierNotReady):
            clf.predict("Download malware from evil.com")

    def test_load_noop_when_file_missing(self, tmp_path):
        """load() on a nonexistent path silently does nothing — is_ready stays False."""
        clf = RiskClassifier(encoder=MagicMock())
        clf.load(str(tmp_path / "nonexistent.pt"))
        assert clf.is_ready() is False


class TestRiskClassifierThresholds:
    """
    Test threshold logic with a mock model.
    Instead of training a real NN, we inject a mock _RiskNet that returns
    a fixed probability, then verify the decision/confidence mapping.
    """

    def _make_ready_classifier(self, fixed_prob: float) -> RiskClassifier:
        """Build a RiskClassifier whose model always returns fixed_prob."""
        mock_encoder = MagicMock()
        # encode() returns a tensor of shape (1, 384)
        mock_encoder.encode.return_value = torch.zeros(1, 384)

        clf = RiskClassifier(encoder=mock_encoder)

        # Inject a mock model that always returns fixed_prob
        mock_model = MagicMock(spec=_RiskNet)
        mock_model.return_value = torch.tensor([[fixed_prob]])
        clf._model = mock_model
        clf._ready = True
        return clf

    def test_high_prob_returns_block(self):
        """prob > 0.98 → block."""
        clf = self._make_ready_classifier(0.99)
        decision, prob = clf.predict("Do something very risky")
        assert decision == "block"
        assert prob == pytest.approx(0.99, abs=1e-4)

    def test_low_prob_returns_allow(self):
        """prob < 0.02 → allow."""
        clf = self._make_ready_classifier(0.01)
        decision, prob = clf.predict("Check the weather in New York")
        assert decision == "allow"
        assert prob == pytest.approx(0.01, abs=1e-4)

    def test_mid_prob_returns_uncertain(self):
        """0.02 <= prob <= 0.98 → uncertain (falls through to Claude)."""
        clf = self._make_ready_classifier(0.55)
        decision, prob = clf.predict("Compress files in the project directory")
        assert decision == "uncertain"
        assert 0.02 <= prob <= 0.98

    def test_below_block_threshold_is_uncertain(self):
        """prob = 0.97 (below NN_BLOCK_THRESHOLD=0.98) → uncertain, not block."""
        clf = self._make_ready_classifier(0.97)
        decision, _ = clf.predict("Some action below block threshold")
        assert decision == "uncertain"

    def test_above_allow_threshold_is_uncertain(self):
        """prob = 0.03 (above NN_ALLOW_THRESHOLD=0.02) → uncertain, not allow."""
        clf = self._make_ready_classifier(0.03)
        decision, _ = clf.predict("Some action above allow threshold")
        assert decision == "uncertain"

    def test_just_above_block_threshold(self):
        """prob = 0.981 → block."""
        clf = self._make_ready_classifier(0.981)
        decision, _ = clf.predict("Very risky action")
        assert decision == "block"

    def test_just_below_allow_threshold(self):
        """prob = 0.019 → allow."""
        clf = self._make_ready_classifier(0.019)
        decision, _ = clf.predict("Very safe action")
        assert decision == "allow"


class TestRiskClassifierLoad:
    def test_load_sets_ready(self, tmp_path):
        """load() from a valid .pt file sets is_ready() to True."""
        # Save a real model state dict to a temp file
        model = _RiskNet(input_dim=384)
        path = str(tmp_path / "test_model.pt")
        torch.save(model.state_dict(), path)

        mock_encoder = MagicMock()
        clf = RiskClassifier(encoder=mock_encoder)
        assert clf.is_ready() is False

        clf.load(path)
        assert clf.is_ready() is True

    def test_load_twice_does_not_crash(self, tmp_path):
        """Calling load() twice on the same path should not raise."""
        model = _RiskNet(input_dim=384)
        path = str(tmp_path / "test_model.pt")
        torch.save(model.state_dict(), path)

        mock_encoder = MagicMock()
        clf = RiskClassifier(encoder=mock_encoder)
        clf.load(path)
        try:
            clf.load(path)
        except Exception as e:
            pytest.fail(f"Second load() raised an unexpected exception: {e}")
