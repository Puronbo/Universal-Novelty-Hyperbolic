# Hyperbolic Manifold Novelty Detection

A toy system that embeds content into a 2D Poincare disk, using topic
anchors and entropy-based anomaly detection to separate legitimate
content from spam/noise.

## Files

- **`hyperbolic_engine.py`** — Core engine. Embeds sample "packets" via
  hyperbolic gradient descent, quarantines high-entropy (scattered-topic)
  content to the disk boundary, and exports `web_data.json`.
- **`index.html`** — Live dashboard (Chart.js) that plots the disk,
  loads `web_data.json`, and persists state across refreshes via
  `localStorage`.
- **`train.py`** — Direct PyTorch version. Learns node positions as
  model weights instead of running the JSON pipeline, and checkpoints
  them to `manifold_model.pth`.

## Run it

```bash
# 1. Install dependencies
pip install numpy torch

# 2. Run the engine to generate web_data.json
python3 hyperbolic_engine.py

# 3. Serve the dashboard (must be served, not opened as file://,
#    or the browser blocks fetch() and localStorage)
python3 -m http.server 8000
# then open http://localhost:8000 and click "Load web_data.json"

# 4. (Optional) Train the manifold directly with PyTorch instead
python3 train.py
```

## How the anomaly detection works

Each anchor (`physics_core`, `tech_infra`, `human_culture`) pulls nearby
content toward it via hyperbolic gradient descent. Content whose
`entropy_risk` score is high (its topics are scattered/incoherent, like
SEO spam mixing "insurance," "sports shoes," and "software") skips the
gradient descent step entirely and gets locked to the outer edge of the
disk (`radius >= 0.85`) — this is what stops scattered spam from
mathematically canceling itself out and slipping into the center as
"neutral" content, which was the original bug in this design.
