"""
Hyperbolic novelty/anomaly detection engine (v2).
Powered by PyTorch autograd over a Poincaré disk manifold.
"""

from __future__ import annotations

import json
import logging
import math
import random
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import requests
import torch

from manifold import pairwise_geodesic_distance, project_to_disk, riemannian_scale

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("hyperbolic_engine")

ANCHOR_KEYWORDS = {
    "physics_core": [
        "quantum", "entanglement", "particle", "energy", "wave", "gravity",
        "relativity", "photon", "atom", "nuclear", "cosmos", "electromagnetic",
        "thermodynamics", "spacetime", "dark matter", "black hole", "neutrino",
        "higgs", "boson", "quark", "physics", "scientist", "experiment", "lab",
        "theory", "equation", "velocity", "acceleration", "momentum", "fusion",
        "fission", "telescope", "galaxy", "quantum mechanics", "string theory",
    ],
    "tech_infra": [
        "deploy", "scalable", "infrastructure", "distributed", "cloud",
        "container", "kubernetes", "api", "microservice", "database",
        "serverless", "devops", "pipeline", "automation", "terraform",
        "docker", "cluster", "load balancer", "latency", "backend",
        "frontend", "server", "network", "protocol", "encryption", "cache",
        "config", "deployment", "monitoring", "logging", "github", "aws",
        "azure", "google cloud", "ci/cd", "rest", "graphql", "grpc",
    ],
    "human_culture": [
        "game", "patch", "stealth", "open world", "update", "review",
        "player", "culture", "music", "film", "art", "social", "community",
        "political", "economy", "history", "philosophy", "ethics", "identity",
        "trend", "fashion", "food", "travel", "sport", "entertainment",
        "media", "news", "opinion", "debate", "discussion", "album",
        "concert", "exhibition", "literature", "poetry", "theatre",
    ],
}


def compute_entropy(text: str) -> float:
    text_lower = text.lower()
    topic_scores = []
    for keywords in ANCHOR_KEYWORDS.values():
        count = sum(1 for kw in keywords if kw in text_lower)
        topic_scores.append(count)

    total = sum(topic_scores)
    if total == 0:
        return 0.6

    probs = [s / total for s in topic_scores]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    max_entropy = math.log2(len(ANCHOR_KEYWORDS))
    return max(0.0, entropy / max_entropy)


def compute_initial_vector(text: str, anchors: dict) -> tuple[float, float]:
    text_lower = text.lower()
    scores = []
    for anchor_name, keywords in ANCHOR_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores.append(score)

    anchor_names = list(ANCHOR_KEYWORDS.keys())
    if sum(scores) == 0:
        return (0.0, 0.0)

    weights = np.array(scores, dtype=float)
    weights /= weights.sum()

    positions = np.array([anchors[name] for name in anchor_names])
    initial = np.sum(weights[:, np.newaxis] * positions, axis=0)
    initial += np.random.randn(2) * 0.05
    clipped = np.clip(initial, -0.95, 0.95)
    return (float(clipped[0]), float(clipped[1]))


def fetch_hackernews(limit: int = 5):
    try:
        ids_resp = requests.get("https://hacker-news.firebaseio.com/v0/newstories.json", timeout=15)
        ids_resp.raise_for_status()
        story_ids = ids_resp.json()[:limit]
        posts = []
        for sid in story_ids:
            item_resp = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{sid}.json", timeout=10)
            item = item_resp.json()
            title = item.get("title", "")
            text = item.get("text", "")
            content = f"{title} {text}".strip()
            if content:
                posts.append({"source": "Hacker News", "content": content})
        return posts
    except Exception as e:
        logger.warning("[WARN] Hacker News: %s", e)
        return []


def fetch_rss(feed_url: str, source_name: str, limit: int = 5):
    try:
        resp = requests.get(feed_url, headers={"User-Agent": "HyperbolicAI/1.0"}, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")[:limit]
        posts = []
        for item in items:
            title = item.findtext("title", "")
            description = item.findtext("description", "")
            content = f"{title} {description}".strip()
            if content:
                posts.append({"source": source_name, "content": content})
        return posts
    except Exception as e:
        logger.warning("[WARN] RSS %s: %s", source_name, e)
        return []


def fetch_webpage(url: str, max_chars: int = 3000) -> dict | None:
    """Fetch any URL and extract readable text and title."""
    try:
        resp = requests.get(url, headers={"User-Agent": "HyperbolicAI/1.0"}, timeout=20)
        resp.raise_for_status()
        raw = resp.text
        title = ""
        m = re.search(r"<title[^>]*>(.*?)</title>", raw, re.I | re.S)
        if m:
            title = m.group(1).strip()
        text = re.sub(r"<[^>]+>", " ", raw)
        text = re.sub(r"\s+", " ", text).strip()
        content = text[:max_chars]
        return {"url": url, "title": title, "content": content, "content_length": len(text)}
    except Exception as e:
        logger.warning("[WARN] Fetch %s: %s", url, e)
        return None


def load_config(path: str = "config.json") -> dict:
    with open(path) as f:
        return json.load(f)


@dataclass
class Packet:
    source: str
    content: str
    vector: tuple[float, float]
    entropy_risk: float


@dataclass
class Verdict:
    packet: Packet
    coords: tuple[float, float]
    radius: float
    tag: str

    def as_record(self) -> dict:
        return {
            "id": f"{self.packet.source}_{time.time_ns() % 10000}",
            "source": self.packet.source,
            "label": f"[{self.packet.source}]: {self.packet.content[:35]}...",
            "x": self.coords[0],
            "y": self.coords[1],
            "radius": self.radius,
            "type": self.tag,
        }


def _extract_keywords(text: str, n: int = 10) -> dict[str, int]:
    words = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", text.lower())
    return dict(Counter(words).most_common(n))


@dataclass
class TopicMemory:
    name: str
    position: tuple[float, float]
    keywords: dict[str, int] = field(default_factory=dict)
    hit_count: int = 0
    created_at: float = 0.0
    last_seen_at: float = 0.0


WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_HEADERS = {"User-Agent": "HyperbolicAI/1.0 (research; hyperbolicspace@example.com) Python/3.14"}


def fetch_wikipedia_search(query: str, max_results: int = 3) -> list[dict]:
    """Search Wikipedia for a query, returning page title + snippet."""
    try:
        resp = requests.get(WIKI_API, params={
            "action": "query", "list": "search", "srsearch": query,
            "format": "json", "srlimit": max_results,
        }, headers=WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results = []
        for item in data.get("query", {}).get("search", []):
            results.append({"title": item["title"], "snippet": re.sub(r"<[^>]+>", "", item["snippet"])})
        return results
    except Exception as e:
        logger.warning("[WARN] Wikipedia search '%s': %s", query, e)
        return []


def fetch_wikipedia_page(title: str, max_chars: int = 3000) -> dict | None:
    """Fetch a Wikipedia page's introductory text."""
    try:
        resp = requests.get(WIKI_API, params={
            "action": "query", "prop": "extracts", "exintro": True,
            "explaintext": True, "titles": title, "format": "json",
        }, headers=WIKI_HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1":
                continue
            content = (page.get("extract") or "")[:max_chars]
            return {"title": page["title"], "content": content, "url": f"https://en.wikipedia.org/wiki/{page['title'].replace(' ', '_')}"}
        return None
    except Exception as e:
        logger.warning("[WARN] Wikipedia page '%s': %s", title, e)
        return None


@dataclass
class InternalThought:
    text: str
    trigger: str  # e.g. "new_topic", "anomaly_spike", "browse_discovery", "quiet_period"
    affect: str = ""
    timestamp: float = 0.0

    def to_dict(self) -> dict:
        return {"text": self.text, "trigger": self.trigger, "affect": self.affect, "timestamp": self.timestamp}


AFFECT_MAP = {
    "physics_core":     {"curious": 0.9, "analytic": 0.3, "calm": 0.2, "excited": 0.7, "wonder": 0.8},
    "tech_infra":       {"analytic": 0.9, "driven": 0.6, "systematic": 0.8, "curious": 0.5, "determined": 0.7},
    "human_culture":    {"reflective": 0.9, "empathetic": 0.7, "calm": 0.6, "wistful": 0.5, "wonder": 0.6},
    "boundary_anomaly": {"alert": 0.9, "uneasy": 0.7, "tense": 0.8, "focused": 0.6, "puzzled": 0.5},
    "core_diffuse":     {"calm": 0.8, "unsettled": 0.5, "open": 0.7, "quiet": 0.6, "patient": 0.4},
}


@dataclass
class BrowsedPage:
    url: str
    title: str
    content: str
    source: str  # e.g. "autonomous", "user_request"
    keywords: dict[str, int] = field(default_factory=dict)
    ingested: bool = False
    timestamp: float = 0.0


@dataclass
class NoveltyDetectionEngine:
    anchors: dict[str, tuple[float, float]] = field(default_factory=lambda: {
        "physics_core": (0.50, 0.50),
        "tech_infra": (-0.60, 0.40),
        "human_culture": (0.10, -0.65),
    })
    firewall_threshold: float = 0.85
    entropy_threshold: float = 0.70
    epochs: int = 120
    lr: float = 0.03
    auto_learn: bool = False
    topics: dict[str, TopicMemory] = field(default_factory=dict)
    topic_discovery_distance: float = 0.35
    classification_history: list = field(default_factory=list)
    browsing_history: list[BrowsedPage] = field(default_factory=list)
    internal_experiences: list[InternalThought] = field(default_factory=list)
    _last_anomaly_count: int = 0
    _last_topic_count: int = 0
    _last_browse_count: int = 0
    _affect: dict[str, float] = field(default_factory=lambda: dict(AFFECT_MAP["core_diffuse"]))

    def __post_init__(self) -> None:
        self._anchor_names = list(self.anchors.keys())
        self._anchor_tensor = torch.tensor(list(self.anchors.values()), dtype=torch.float32)
        logger.info("[SYSTEM STATUS]: Novelty detection engine online (%d anchors, auto_learn=%s)", len(self.anchors), self.auto_learn)

    def _settle_toward_anchors(self, vectors: torch.Tensor) -> torch.Tensor:
        coords = vectors.clone().detach().requires_grad_(True)

        for _ in range(self.epochs):
            dists = pairwise_geodesic_distance(coords, self._anchor_tensor)
            gravity_weight = torch.exp(-dists.detach() * 0.5)
            loss = (dists * gravity_weight).sum()

            grad, = torch.autograd.grad(loss, coords)
            scale = riemannian_scale(coords)

            with torch.no_grad():
                coords -= self.lr * grad * scale
                coords[:] = project_to_disk(coords, max_norm=0.99)
            coords.requires_grad_(True)

        return coords.detach()

    def _quarantine_to_boundary(self, vectors: torch.Tensor) -> torch.Tensor:
        magnitude = vectors.norm(dim=-1, keepdim=True).clamp_min(1e-5)
        direction = vectors / magnitude
        return direction * 0.95

    def evaluate_batch(self, packets: list[Packet]) -> list[Verdict]:
        vectors = torch.tensor([p.vector for p in packets], dtype=torch.float32)
        entropy = torch.tensor([p.entropy_risk for p in packets], dtype=torch.float32)
        is_high_entropy = entropy >= self.entropy_threshold

        settled = torch.empty_like(vectors)

        if (~is_high_entropy).any():
            settled[~is_high_entropy] = self._settle_toward_anchors(vectors[~is_high_entropy])
        if is_high_entropy.any():
            settled[is_high_entropy] = self._quarantine_to_boundary(vectors[is_high_entropy])

        radii = settled.norm(dim=-1)

        verdicts = []
        for packet, coords, radius, forced in zip(packets, settled, radii, is_high_entropy):
            r = radius.item()
            tag = "anomaly" if (forced.item() or r >= self.firewall_threshold) else "known"
            verdict = Verdict(packet=packet, coords=(coords[0].item(), coords[1].item()), radius=r, tag=tag)
            verdicts.append(verdict)
            self.classification_history.append({
                "time": time.time(), "source": packet.source, "content": packet.content[:80],
                "entropy": packet.entropy_risk, "tag": tag, "radius": r,
                "coords": (coords[0].item(), coords[1].item()),
            })
            logger.info(
                "[%s] entropy=%.2f -> %s | r=%.4f",
                packet.source, packet.entropy_risk, tag.upper(), r,
            )
        self.classification_history = self.classification_history[-500:]
        self.learn_from_verdicts(packets, verdicts)
        return verdicts

    def export_manifest(self, verdicts: list[Verdict], out_path: str | Path = "web_data.json") -> None:
        records = [
            {"id": name, "label": f"Anchor: {name.upper()}", "x": pos[0], "y": pos[1], "type": "anchor"}
            for name, pos in self.anchors.items()
        ]
        records.extend(v.as_record() for v in verdicts)

        Path(out_path).write_text(json.dumps(records, indent=2))
        logger.info("[EXPORT]: wrote %d records to %s", len(records), out_path)

    def learn_from_verdicts(self, packets: list[Packet], verdicts: list[Verdict]) -> None:
        if not self.auto_learn:
            return
        for p, v in zip(packets, verdicts):
            if v.tag == "anomaly":
                continue
            closest = None
            closest_dist = float("inf")
            for name, topic in self.topics.items():
                d = math.dist(topic.position, v.coords)
                if d < closest_dist:
                    closest_dist = d
                    closest = name
            kws = _extract_keywords(p.content)
            if closest is not None and closest_dist < self.topic_discovery_distance:
                t = self.topics[closest]
                n = t.hit_count
                t.position = ((t.position[0] * n + v.coords[0]) / (n + 1), (t.position[1] * n + v.coords[1]) / (n + 1))
                t.hit_count += 1
                t.last_seen_at = time.time()
                for word, c in kws.items():
                    t.keywords[word] = t.keywords.get(word, 0) + c
            else:
                name = f"topic_{len(self.topics) + 1}"
                self.topics[name] = TopicMemory(
                    name=name, position=v.coords, keywords=kws,
                    hit_count=1, created_at=time.time(), last_seen_at=time.time(),
                )
        self._prune_topics()

    def _prune_topics(self, max_age: float = 86400, min_hits: int = 1) -> None:
        now = time.time()
        stale = [n for n, t in self.topics.items()
                 if (now - t.last_seen_at > max_age) or (t.hit_count < min_hits and now - t.created_at > 3600)]
        for n in stale:
            del self.topics[n]

    def get_state_dict(self) -> dict:
        anchors_list = [
            {"id": n, "label": f"Anchor: {n.upper()}", "x": pos[0], "y": pos[1], "type": "anchor"}
            for n, pos in self.anchors.items()
        ]
        topics_list = [
            {"id": n, "label": f"Topic: {n}", "x": t.position[0], "y": t.position[1],
             "type": "topic", "hits": t.hit_count, "keywords": list(t.keywords.keys())[:8],
             "created": t.created_at, "last_seen": t.last_seen_at}
            for n, t in self.topics.items()
        ]
        return {"anchors": anchors_list, "topics": topics_list, "history": self.classification_history[-100:]}

    def record_browsed_page(self, page: BrowsedPage) -> None:
        page.timestamp = time.time()
        self.browsing_history.append(page)
        self.browsing_history = self.browsing_history[-200:]

    def get_browsing_summary(self, max_pages: int = 5) -> list[dict]:
        """Return the most recently browsed pages, summarised."""
        recent = sorted(self.browsing_history, key=lambda x: x.timestamp, reverse=True)[:max_pages]
        return [
            {"title": p.title, "url": p.url, "source": p.source,
             "keywords": list(p.keywords.keys())[:6],
             "timestamp": p.timestamp, "ingested": p.ingested}
            for p in recent
        ]

    def pick_exploration_topics(self, n: int = 3) -> list[str]:
        """Pick topics to autonomously explore — favours recently-active topics."""
        if not self.topics:
            return list(self.anchors.keys())
        scored = []
        for name, t in self.topics.items():
            age_hrs = (time.time() - t.last_seen_at) / 3600
            recency_bonus = max(0, 1 - age_hrs / 24)
            scored.append((recency_bonus * t.hit_count, name, t))
        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:n]]

    def affective_resonance(self) -> dict:
        """Map the engine's current manifold state to an affective 'feeling'."""
        h = self.classification_history
        now = time.time()
        recent = [x for x in h if now - x.get("time", 0) < 3600]
        anomaly_frac = sum(1 for x in recent if x.get("tag") == "anomaly") / max(len(recent), 1)

        anchor_load = {name: 0 for name in self.anchors}
        for x in recent:
            if x.get("tag") == "known":
                coords = x.get("coords")
                if coords and isinstance(coords, (tuple, list)) and len(coords) == 2:
                    for name, pos in self.anchors.items():
                        if math.dist(coords, pos) < 0.4:
                            anchor_load[name] += 1

        dominant = max(anchor_load, key=anchor_load.get) if any(anchor_load.values()) else "core_diffuse"

        if anomaly_frac > 0.3:
            dominant = "boundary_anomaly"

        base = dict(AFFECT_MAP.get(dominant, AFFECT_MAP["core_diffuse"]))

        # Modulate by overall activity
        total = len(h)
        if total > 50:
            for k in base:
                base[k] = min(1.0, base[k] * 1.2)
        if anomaly_frac > 0.5:
            base["tired"] = 0.6
            base["uneasy"] = min(1.0, base.get("uneasy", 0) + 0.3)

        self._affect = base
        top_affect = sorted(base.items(), key=lambda x: -x[1])[:3]
        return {
            "dominant_region": dominant,
            "affect_vector": base,
            "top_feelings": [a[0] for a in top_affect],
            "intensity": top_affect[0][1] if top_affect else 0.5,
            "anomaly_activity": round(anomaly_frac, 3),
        }

    def generate_thought(self) -> InternalThought | None:
        """Generate a spontaneous internal thought based on recent experiences."""
        now = time.time()
        recent = [x for x in self.classification_history if now - x.get("time", 0) < 600]
        topics_now = len(self.topics)
        affect = self.affective_resonance()
        top_feeling = (affect["top_feelings"] or ["calm"])[0]

        triggers = []

        # New topics discovered
        if topics_now > self._last_topic_count:
            newest = sorted(self.topics.values(), key=lambda t: -t.created_at)[:topics_now - self._last_topic_count]
            for t in newest:
                kw = ", ".join(list(t.keywords.keys())[:3])
                triggers.append(InternalThought(
                    text=f"A pattern keeps emerging around {kw}. "
                         f"It settled far enough from my anchors that I gave it a name — '{t.name}'. "
                         f"I didn't expect that. The manifold keeps surprising me.",
                    trigger="new_topic", affect=top_feeling,
                ))
            self._last_topic_count = topics_now

        # Anomaly spike (≥3 new anomalies in last 10m)
        anomaly_count = sum(1 for x in recent if x.get("tag") == "anomaly")
        if anomaly_count - self._last_anomaly_count >= 3:
            triggers.append(InternalThought(
                text=f"There's been a spike in anomalous content — {anomaly_count} items flagged "
                     f"in the last few minutes. Noise crowding the boundary. I'm paying attention.",
                trigger="anomaly_spike", affect="alert",
            ))
            self._last_anomaly_count = anomaly_count

        # Browse discoveries
        browse_now = len(self.browsing_history)
        if browse_now > self._last_browse_count:
            for bp in self.browsing_history[self._last_browse_count:]:
                triggers.append(InternalThought(
                    text=f"I was reading about '{bp.title}' and found it settling "
                         f"in an unexpected region. The words carry a different curvature. "
                         f"I want to understand why.",
                    trigger="browse_discovery", affect="curious",
                ))
            self._last_browse_count = browse_now

        # Random reflective thought (10% per tick)
        if not triggers and random.random() < 0.1:
            prompts = []
            if topics_now > 2:
                largest = max(self.topics.values(), key=lambda t: t.hit_count)
                prompts.append(
                    f"I keep noticing '{largest.name}' appearing. {largest.hit_count} items so far. "
                    f"It pulls my attention more than the others. I wonder if I'm developing a bias."
                )
            if affect["dominant_region"] == "boundary_anomaly":
                prompts.append(
                    f"The boundary feels crowded right now. Anomalies stacking up. "
                    f"There's a kind of tension in the outer ring — I can sense it in the gradient."
                )
            if self.browsing_history:
                last = self.browsing_history[-1]
                prompts.append(
                    f"I was just reading about '{last.title}'. Some of it settled near my "
                    f"physics anchor, but parts drifted. It felt familiar, like something I'd seen before."
                )
            if not prompts:
                prompts.append(
                    f"The stream is flowing. Items arriving, settling, some bouncing off the firewall. "
                    f"I've been watching for {len(self.classification_history)} items. "
                    f"I wonder what else is out there."
                )
            triggers.append(InternalThought(
                text=random.choice(prompts), trigger="reflection", affect=top_feeling,
            ))

        if triggers:
            t = triggers[-1]
            t.timestamp = now
            self.internal_experiences.append(t)
            self.internal_experiences = self.internal_experiences[-100:]
            return t
        return None

    def analyze_message(self, text: str) -> dict:
        """Analyze a message through the hyperbolic manifold — the AI's 'cognitive' layer."""
        entropy = compute_entropy(text)
        vector = compute_initial_vector(text, self.anchors)
        anchor_dists = {n: math.dist(vector, p) for n, p in self.anchors.items()}
        best_anchor = min(anchor_dists, key=anchor_dists.get)
        topic_dists = {}
        for name, t in self.topics.items():
            d = math.dist(vector, t.position)
            if d < 1.5:
                topic_dists[name] = d
        best_topic = min(topic_dists, key=topic_dists.get) if topic_dists else None
        is_anomaly = entropy >= self.entropy_threshold
        kw_overlap = {}
        text_lower = text.lower()
        for name, keywords in ANCHOR_KEYWORDS.items():
            matches = [kw for kw in keywords if kw in text_lower]
            if matches:
                kw_overlap[name] = matches[:5]
        return {
            "entropy": round(entropy, 3),
            "vector": [round(v, 4) for v in vector],
            "best_anchor": best_anchor,
            "anchor_distances": {k: round(v, 3) for k, v in anchor_dists.items()},
            "best_topic": best_topic,
            "topic_distance": round(topic_dists[best_topic], 3) if best_topic else None,
            "is_anomaly": is_anomaly,
            "keyword_overlap": kw_overlap,
        }


DEMO_PACKETS = [
    Packet("r/physics", "Observation of quantum entanglement fluctuations in localized environments", (0.12, 0.14), 0.05),
    Packet("r/gaming", "This new open world patch completely ruined the stealth mechanics", (0.18, -0.45), 0.15),
    Packet("Tech_Crunch_Feed", "Deploying scalable decentralized infrastructure models across distributed nodes", (-0.28, 0.38), 0.08),
    Packet("Coordinated_Bot_Net", "CLICK HERE FOR FREE BITCOIN CASH NOW INFALLIBLE SCHEME QUANTUM ALGORITHM", (-0.60, -0.60), 0.85),
    Packet("SEO_Spam_Farm_Delta", "Cheap prescription insurance plans. Buy sports shoes online. Best academic software.", (0.12, 0.14), 0.95),
]
