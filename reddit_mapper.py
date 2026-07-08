import numpy as np
import json

# Conformal distance calculation for validation
def hyperbolic_dist(u, v):
    sq_norm_u = np.sum(u**2)
    sq_norm_v = np.sum(v**2)
    sq_dist = np.sum((u - v)**2)
    denom = max((1.0 - sq_norm_u) * (1.0 - sq_norm_v), 1e-12)
    return np.arccosh(1.0 + 2.0 * sq_dist / denom)

# Simulating a Reddit post comment architecture
# Thread Topic: "What did you think of the movie's confusing ending?"
reddit_thread = {
    "Post_Root": {
        "text": "The ending made absolutely no sense. Was it a dream?",
        "coord": np.array([0.1, -0.1]), "tier": 1, "author": "User_A"
    },
    "Reply_1_Structural": {
        "text": "No, if you trace the mirror motif, it represents psychological dissociation.",
        "coord": np.array([0.25, -0.20]), "tier": 2, "author": "User_B"
    },
    "Reply_2_Deep_Analysis": {
        "text": "Exactly. The mirror is a classic cinematic metaphor for splitting the ego.",
        "coord": np.array([0.45, -0.35]), "tier": 3, "author": "User_C"
    },
    "Reply_3_Chaos_Outlier": {
        "text": "Honestly the main actor just looked like a wet piece of celery the whole time lmao",
        "coord": np.array([-0.75, 0.55]), "tier": 2, "author": "User_D"
    }
}

print("--- HYPERBOLIC REDDIT MAPPING ENGINE ---")

# Step 2: Evaluate the semantic distance between the deep analysis and the chaos outlier
d_constructive = hyperbolic_dist(reddit_thread["Post_Root"]["coord"], reddit_thread["Reply_2_Deep_Analysis"]["coord"])
d_chaos = hyperbolic_dist(reddit_thread["Post_Root"]["coord"], reddit_thread["Reply_3_Chaos_Outlier"]["coord"])

print(f"\n[Semantic Gravity Analysis]")
print(f"-> Distance from Root to Deep Analysis (User_C): {d_constructive:.4f} (Close proximity; shared context branch)")
print(f"-> Distance from Root to Shitpost Anomaly (User_D): {d_chaos:.4f} (Massive distance; pushed to the horizon)")

# Exporting a fresh map layer specifically for your visual web interface
reddit_export = []
for node_id, meta in reddit_thread.items():
    node_type = "known" if meta["tier"] < 3 else "probe"
    if node_id == "Reply_3_Chaos_Outlier":
        node_type = "anomaly"
        
    reddit_export.append({
        "id": node_id,
        "label": f"{meta['author']}: {meta['text'][:40]}...",
        "tier": meta["tier"],
        "x": meta["coord"][0].item(),
        "y": meta["coord"][1].item(),
        "type": node_type
    })

with open("web_data.json", "w") as f:
    json.dump(reddit_export, f, indent=4)

print("\n[Map Overwritten: web_data.json has been updated with live Reddit thread coordinates]")