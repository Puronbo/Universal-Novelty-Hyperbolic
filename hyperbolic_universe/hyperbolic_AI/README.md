# Hyperbolic Manifold Novelty Detection (v2)

Rewritten for correctness, speed, and testability. Same math and same
results as the original prototype, different engineering underneath.

## What changed from the prototype

| | Prototype | This version |
|---|---|---|
| Gradients | Manual central-difference (`eps=1e-5`, 2 evals/anchor/epoch) | Exact, via `torch.autograd` |
| Batch processing | Python `for` loop, one packet at a time | Fully vectorized — a whole batch settles in one graph |
| Structure | One file, prints for logging, no tests | `manifold/` (math) + `engine.py` (logic + live ingestion) + `tests/` |

## Layout

```
manifold/poincare.py   # geodesic distance, disk projection, Riemannian scale — all autograd-native
engine.py              # NoveltyDetectionEngine: batched settle/quarantine logic + JSON export + live ingestion
tests/test_engine.py   # 7 tests: geometry correctness + the original spam-collapse bug as a regression test
config.json            # Source definitions, thresholds, anchor coordinates
index.html             # Chart.js dashboard with auto-refresh and localStorage persistence
requirements.txt
```

## Run it

```bash
pip install -r requirements.txt

# Run the engine (prints verdicts, writes web_data.json)
python3 engine.py --demo

# Live ingestion mode (Hacker News, RSS — press Ctrl+C to stop)
python3 engine.py

# Run the tests
python3 -m pytest tests/ -v

# Serve the dashboard
python3 -m http.server 8000
# open http://localhost:8000
```

## Using it programmatically

```python
from engine import NoveltyDetectionEngine, Packet

engine = NoveltyDetectionEngine()
packets = [
    Packet(source="my_feed", content="...", vector=(0.1, 0.2), entropy_risk=0.1),
]
verdicts = engine.evaluate_batch(packets)
engine.export_manifest(verdicts)
```

## Configuration

Edit `config.json`:

| Key | Description |
|-----|-------------|
| `sources` | Array of `hackernews`, `reddit`, `rss` sources with intervals |
| `thresholds.firewall` | Radius above which content is quarantined (default: 0.85) |
| `thresholds.entropy` | Entropy score above which content skips gradient descent (default: 0.70) |
| `anchors` | Topic anchor coordinates on the Poincare disk |
| `main_interval_seconds` | How often to save `web_data.json` (default: 600) |
