"""
Hyperbolic Geometry Novelty/Anomaly Detection Engine
-----------------------------------------------------
Embeds incoming content ("packets") into a 2D Poincare disk. Legitimate,
topically-focused content gets pulled by gradient descent toward the
nearest of a handful of fixed topic anchors. Content whose internal
vocabulary is too scattered (high "entropy") is instead pushed straight
to the boundary of the disk and flagged as an anomaly, so it can't
disguise itself as "neutral" content sitting near the center.
"""

import json
import time
import numpy as np

np.random.seed(42)


class UniversalHyperbolicEngine:
    def __init__(self, firewall_threshold: float = 0.85, entropy_threshold: float = 0.70):
        self.origin = np.array([0.0, 0.0])
        self.firewall_threshold = firewall_threshold
        self.entropy_threshold = entropy_threshold

        # Topic anchors spaced out toward the edges so each topic gets its
        # own "territory" on the disk instead of everything collapsing to
        # one point in the middle.
        self.manifold_anchors = {
            "physics_core": np.array([0.50, 0.50]),
            "tech_infra": np.array([-0.60, 0.40]),
            "human_culture": np.array([0.10, -0.65]),
        }

        self.global_registry = []
        print("=" * 62)
        print("[SYSTEM STATUS]: UNIFIED HYPERBOLIC NETWORK CORE ONLINE")
        print("[PROTECTION STATE]: ENTROPY DEFENSES ARMED & ACTIVE")
        print("=" * 62 + "\n")

    def hyperbolic_dist(self, u: np.ndarray, v: np.ndarray) -> float:
        """Geodesic distance between two points inside the Poincare disk."""
        sq_norm_u = np.sum(u ** 2)
        sq_norm_v = np.sum(v ** 2)
        sq_dist = np.sum((u - v) ** 2)
        denom = max((1.0 - sq_norm_u) * (1.0 - sq_norm_v), 1e-12)
        return np.arccosh(1.0 + 2.0 * sq_dist / denom)

    def process_stream_packet(self, source: str, content: str, raw_vector: np.ndarray, entropy_risk: float):
        print(f"[INGESTING STREAM] Source: {source:<22} | Entropy Risk: {entropy_risk:.2f}")

        # Phase 1: High-entropy content is denied entry to the taxonomy
        # outright and pinned to the boundary, no matter where it initially
        # landed. This is what stops "scattered vocabulary" content from
        # exploiting the fact that opposing pulls can cancel out near the
        # center (the bug the original engine had).
        if entropy_risk >= self.entropy_threshold:
            magnitude = np.linalg.norm(raw_vector)
            direction = raw_vector / (magnitude + 1e-5) if magnitude > 0 else np.array([1.0, 0.0])
            settled_coords = direction * 0.95
            status = "EXHAUSTED_TO_HORIZON_WALL (QUARANTINED)"
            tag_type = "anomaly"

        # Phase 2: Legitimate content settles via gradient descent, pulled
        # by anchors whose influence decays with distance ("localized
        # gravity") so nearby topics don't get dragged into unrelated ones.
        else:
            settled_coords = raw_vector.copy()
            epochs, lr, eps = 120, 0.03, 1e-5

            for _ in range(epochs):
                grad = np.zeros(2)
                for _, anchor_pos in self.manifold_anchors.items():
                    d = self.hyperbolic_dist(settled_coords, anchor_pos)
                    gravity_weight = np.exp(-d * 0.5)

                    c_eps0 = settled_coords.copy(); c_eps0[0] += eps
                    g0 = (self.hyperbolic_dist(c_eps0, anchor_pos) - d) / eps

                    c_eps1 = settled_coords.copy(); c_eps1[1] += eps
                    g1 = (self.hyperbolic_dist(c_eps1, anchor_pos) - d) / eps

                    grad += np.array([g0, g1]) * gravity_weight

                riemannian_scale = ((1.0 - np.sum(settled_coords ** 2)) ** 2) / 4.0
                settled_coords = settled_coords - lr * grad * riemannian_scale

                radius = np.linalg.norm(settled_coords)
                if radius >= 0.99:
                    settled_coords = (settled_coords / radius) * 0.99
                    break

            final_radius = np.linalg.norm(settled_coords)
            if final_radius >= self.firewall_threshold:
                status = "EXHAUSTED_TO_HORIZON_WALL (QUARANTINED)"
                tag_type = "anomaly"
            else:
                status = "CLEAN_VERIFICATION (SECURED)"
                tag_type = "known"

        final_radius = np.linalg.norm(settled_coords)
        rounded = np.round(settled_coords, 4)
        print(f"    -> Result: {status}")
        print(f"    -> Final Radius (r): {final_radius:.4f} | Final Coordinates: {rounded}\n")

        self.global_registry.append({
            "id": f"{source}_{time.time_ns() % 10000}",
            "source": source,
            "label": f"[{source}]: {content[:35]}...",
            "x": settled_coords[0].item(),
            "y": settled_coords[1].item(),
            "radius": final_radius.item(),
            "type": tag_type,
        })
        return tag_type, final_radius

    def deploy_and_compile_manifest(self, out_path: str = "web_data.json"):
        master_output = []
        for name, pos in self.manifold_anchors.items():
            master_output.append({
                "id": name,
                "label": f"Anchor: {name.upper()}",
                "x": pos[0].item(),
                "y": pos[1].item(),
                "type": "anchor",
            })
        master_output.extend(self.global_registry)

        with open(out_path, "w") as f:
            json.dump(master_output, f, indent=4)
        print(f"[MASTER SYSTEM UPDATE]: Recompiled and synced global map database to {out_path}")


if __name__ == "__main__":
    core_system = UniversalHyperbolicEngine()

    firehose_packets = [
        {
            "source": "r/physics",
            "content": "Observation of quantum entanglement fluctuations in localized environments",
            "vector": np.array([0.12, 0.14]),
            "entropy": 0.05,
        },
        {
            "source": "r/gaming",
            "content": "This new open world patch completely ruined the stealth mechanics",
            "vector": np.array([0.18, -0.45]),
            "entropy": 0.15,
        },
        {
            "source": "Tech_Crunch_Feed",
            "content": "Deploying scalable decentralized infrastructure models across distributed nodes",
            "vector": np.array([-0.28, 0.38]),
            "entropy": 0.08,
        },
        {
            "source": "Coordinated_Bot_Net",
            "content": "CLICK HERE FOR FREE BITCOIN CASH NOW INFALLIBLE SCHEME QUANTUM ALGORITHM",
            "vector": np.array([-0.60, -0.60]),
            "entropy": 0.85,
        },
        {
            "source": "SEO_Spam_Farm_Delta",
            "content": "Cheap prescription insurance plans. Buy sports shoes online. Best academic software.",
            "vector": np.array([0.12, 0.14]),  # tries to disguise itself near the physics anchor
            "entropy": 0.95,
        },
    ]

    for packet in firehose_packets:
        core_system.process_stream_packet(
            source=packet["source"],
            content=packet["content"],
            raw_vector=packet["vector"],
            entropy_risk=packet["entropy"],
        )

    core_system.deploy_and_compile_manifest()
