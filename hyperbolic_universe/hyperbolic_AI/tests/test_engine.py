import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine import NoveltyDetectionEngine, Packet
from manifold import geodesic_distance, project_to_disk


def test_geodesic_distance_near_zero_for_identical_points():
    u = torch.tensor([0.2, 0.3])
    assert geodesic_distance(u, u).item() == pytest.approx(0.0, abs=1e-3)

def test_geodesic_distance_symmetric():
    u = torch.tensor([0.2, 0.3])
    v = torch.tensor([-0.1, 0.4])
    assert geodesic_distance(u, v).item() == pytest.approx(geodesic_distance(v, u).item())

def test_project_to_disk_keeps_points_inside():
    x = torch.tensor([[5.0, 5.0], [0.1, 0.1]])
    projected = project_to_disk(x, max_norm=0.99)
    norms = projected.norm(dim=-1)
    assert (norms <= 0.99 + 1e-6).all()

def test_clean_content_settles_near_anchor():
    engine = NoveltyDetectionEngine()
    packet = Packet("test_source", "physics content", (0.12, 0.14), entropy_risk=0.05)
    verdict = engine.evaluate_batch([packet])[0]
    assert verdict.tag == "known"
    assert verdict.radius < engine.firewall_threshold

def test_high_entropy_content_is_quarantined():
    engine = NoveltyDetectionEngine()
    packet = Packet("spam_source", "insurance shoes software", (0.12, 0.14), entropy_risk=0.95)
    verdict = engine.evaluate_batch([packet])[0]
    assert verdict.tag == "anomaly"
    assert verdict.radius >= engine.firewall_threshold

def test_scattered_topic_spam_cannot_hide_at_origin():
    """Regression test for the original bug: spam near an anchor's coordinates
    should still be caught once flagged as high entropy, instead of collapsing
    to the center and being misread as 'neutral'."""
    engine = NoveltyDetectionEngine()
    disguised_spam = Packet("SEO_Spam_Farm_Delta", "insurance shoes software", (0.12, 0.14), entropy_risk=0.95)
    verdict = engine.evaluate_batch([disguised_spam])[0]
    assert verdict.tag == "anomaly"

def test_batch_matches_single_item_evaluation():
    engine = NoveltyDetectionEngine()
    a = Packet("a", "x", (0.1, 0.1), 0.05)
    b = Packet("b", "x", (-0.2, 0.3), 0.05)

    batched = engine.evaluate_batch([a, b])
    single_a = engine.evaluate_batch([a])[0]
    single_b = engine.evaluate_batch([b])[0]

    assert batched[0].radius == pytest.approx(single_a.radius, abs=1e-5)
    assert batched[1].radius == pytest.approx(single_b.radius, abs=1e-5)
