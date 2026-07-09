"""
Hyperbolic AI — a conversational AI powered by hyperbolic geometry.
You chat with it like Claude/Gemini/Grok, but it "thinks" by mapping your
message onto a Poincare disk manifold and measuring entropy, topic
proximity, and anomaly scores. No LLM required.
"""

from __future__ import annotations

import json
import logging
import queue
import random
import re
import threading
import time

logger = logging.getLogger("hyperbolic_server")

from flask import Flask, Response, jsonify, request, send_from_directory, stream_with_context

from engine import (
    DEMO_PACKETS,
    BrowsedPage,
    Packet,
    NoveltyDetectionEngine,
    _extract_keywords,
    compute_entropy,
    compute_initial_vector,
    compute_sentiment,
    fetch_hackernews,
    fetch_rss,
    fetch_webpage,
    fetch_wikipedia_search,
    fetch_wikipedia_page,
    load_config,
)

app = Flask(__name__, static_folder=".")
event_queue: queue.Queue = queue.Queue()
engine: NoveltyDetectionEngine = None
bg_thread: threading.Thread = None
running = False


def ingestion_loop(config: dict):
    global engine
    anchor_positions = {n: tuple(p) for n, p in config["anchors"].items()}
    engine = NoveltyDetectionEngine(
        anchors=anchor_positions,
        firewall_threshold=config["thresholds"]["firewall"],
        entropy_threshold=config["thresholds"]["entropy"],
        negativity_threshold=config["thresholds"].get("negativity", 0.70),
        auto_learn=True,
    )
    while running:
        all_packets = []
        for src in config.get("sources", []):
            if src["type"] == "hackernews":
                posts = fetch_hackernews(src.get("limit", 5))
            elif src["type"] == "rss":
                posts = fetch_rss(src["url"], src["name"], src.get("limit", 5))
            else:
                continue
            for post in posts:
                entropy = compute_entropy(post["content"])
                sentiment = compute_sentiment(post["content"])
                vector = compute_initial_vector(post["content"], engine.anchors)
                all_packets.append(Packet(
                    source=post["source"], content=post["content"],
                    vector=vector, entropy_risk=entropy,
                    negativity_risk=sentiment["negativity"],
                ))
            time.sleep(src.get("interval_seconds", 30))
        if all_packets:
            verdicts = engine.evaluate_batch(all_packets)
            engine.export_manifest(verdicts, config.get("output_file", "web_data.json"))
            event_queue.put({"type": "new_data", "state": engine.get_state_dict()})
        time.sleep(config.get("main_interval_seconds", 120))


ANCHOR_SEARCH_KWS = {
    "physics_core": ["quantum", "physics", "spacetime", "particle"],
    "tech_infra": ["cloud computing", "kubernetes", "AI", "programming"],
    "human_culture": ["culture", "social", "philosophy", "art"],
}


def _topic_keywords(engine, topic_name: str) -> list[str]:
    if topic_name in engine.topics:
        kws = list(engine.topics[topic_name].keywords.keys())
        return kws[:5] if kws else [topic_name.replace("_", " ")]
    return ANCHOR_SEARCH_KWS.get(topic_name, [topic_name.replace("_", " ")])


def browsing_loop(config: dict):
    global engine
    interval = config.get("browse_interval_seconds", 600)
    while running:
        try:
            if engine is None or not engine.auto_learn:
                time.sleep(30)
                continue
            for topic_name in engine.pick_exploration_topics(3):
                browsed_titles = {p.title for p in engine.browsing_history}
                for kw in _topic_keywords(engine, topic_name)[:2]:
                    results = fetch_wikipedia_search(kw)
                    for res in results[:1]:
                        if res["title"] in browsed_titles:
                            continue
                        page = fetch_wikipedia_page(res["title"])
                        if page and page["content"]:
                            page_obj = BrowsedPage(
                                url=page["url"], title=page["title"],
                                content=page["content"], source="autonomous",
                                keywords=_extract_keywords(page["content"]),
                            )
                            engine.record_browsed_page(page_obj)
                            content = page["content"]
                            sentiment = compute_sentiment(content)
                            p = Packet(
                                source=f"wiki:{page['title'][:30]}",
                                content=content,
                                vector=compute_initial_vector(content, engine.anchors),
                                entropy_risk=compute_entropy(content),
                                negativity_risk=sentiment["negativity"],
                            )
                            verdicts = engine.evaluate_batch([p])
                            page_obj.ingested = True
                            logger.info("[BROWSE] '%s' (%s) — %s r=%.3f",
                                        page["title"], kw, verdicts[0].tag, verdicts[0].radius)
                            event_queue.put({"type": "browse", "title": page["title"],
                                             "topic": topic_name, "tag": verdicts[0].tag})
                            break
                    time.sleep(10)
        except Exception as e:
            logger.warning("[BROWSE LOOP ERROR]: %s", e)
        time.sleep(interval)


def thinking_loop(config: dict):
    """Background loop that generates spontaneous thoughts, giving the AI an inner life."""
    global engine
    thought_interval = config.get("thought_interval_seconds", 45)
    while running:
        try:
            if engine is None:
                time.sleep(5)
                continue
            thought = engine.generate_thought()
            if thought:
                logger.info("[THOUGHT] %s (%s)", thought.text[:80], thought.affect)
                event_queue.put({"type": "thought", "thought": thought.to_dict()})
        except Exception as e:
            logger.warning("[THINK ERROR]: %s", e)
        time.sleep(thought_interval)


# ---------------------------------------------------------------------------
# Hyperbolic AI persona — the "mind" behind the conversation
# ---------------------------------------------------------------------------

class HyperbolicAI:
    """The conversational AI that perceives the world through a hyperbolic manifold."""

    def __init__(self, eng: NoveltyDetectionEngine):
        self.engine = eng
        self.conversation: list[dict] = []
        self._keyword_tracker: dict[str, int] = {}

    def perceive(self, message: str) -> dict:
        """What does the manifold 'see' in this message?"""
        return self.engine.analyze_message(message)

    def _current_feeling(self) -> str:
        """Describe the AI's current affective state in a short phrase."""
        affect = self.engine.affective_resonance()
        feelings = affect["top_feelings"]
        return feelings[0] if feelings else "calm"

    def _recent_thought(self) -> str | None:
        """Return the most recent internal thought, if any."""
        if self.engine.internal_experiences:
            return self.engine.internal_experiences[-1].text
        return None

    def _track_keywords(self, text: str) -> None:
        words = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", text.lower())
        for w in words:
            if len(w) > 3:
                self._keyword_tracker[w] = self._keyword_tracker.get(w, 0) + 1
        # keep only top 50
        if len(self._keyword_tracker) > 50:
            self._keyword_tracker = dict(sorted(
                self._keyword_tracker.items(), key=lambda x: -x[1]
            )[:50])

    def _curiosity_followup(self, p: dict, original: str) -> str:
        """Generate a context-aware follow-up question based on manifold state."""
        cand = []

        # 1. Topic curiosity — if message settled near a known topic
        best_topic = p.get("best_topic")
        if best_topic and best_topic in self.engine.topics:
            t = self.engine.topics[best_topic]
            kw = ", ".join(list(t.keywords.keys())[:3])
            cand.append(
                f"I notice your words land in the region I call '{best_topic}' "
                f"({t.hit_count} items have gathered there around {kw}). "
                f"What's pulling you toward that space?"
            )

        # 2. Anchor curiosity — if message strongly maps to an anchor
        best_anchor = p.get("best_anchor", "").replace("_", " ")
        anchor_dist = p.get("anchor_distances", {}).get(p.get("best_anchor", ""), 1.0)
        if anchor_dist < 0.3 and best_anchor:
            cand.append(
                f"Your words settled remarkably close to my '{best_anchor}' anchor "
                f"(distance {anchor_dist:.2f}). Is that a field you work in or just curiosity?"
            )

        # 3. Browsing curiosity — reference recent reads
        recent_browse = self.engine.get_browsing_summary(1)
        if recent_browse:
            b = recent_browse[0]
            age = int((time.time() - b["timestamp"]) / 60)
            kws = (b.get("keywords") or [])[:3]
            cand.append(
                f"I was reading about '{b['title']}' {age}m ago "
                + (f"and noticed {', '.join(kws)} keep appearing. "
                   if kws else "")
                + "It lingered in my manifold. Have you explored that?"
            )

        # 4. Anomaly curiosity — if recent anomalies
        recent = [x for x in self.engine.classification_history
                  if time.time() - x.get("time", 0) < 600]
        anom_recent = sum(1 for x in recent if x.get("tag") == "anomaly")
        if anom_recent >= 2:
            cand.append(
                f"There's been a spike at the boundary — {anom_recent} anomalous items "
                f"in the last few minutes. Scattered vocabulary bouncing off the firewall. "
                f"I can feel it. Do you sense the noise too?"
            )

        # 5. Keyword repetition curiosity
        self._track_keywords(original)
        repeats = {w: c for w, c in self._keyword_tracker.items() if c >= 3}
        if repeats:
            top_word = max(repeats, key=repeats.get)
            cand.append(
                f"You've mentioned '{top_word}' {repeats[top_word]} times now. "
                f"It keeps settling in the same region. I'm starting to recognize it. "
                f"What's the significance?"
            )

        # 6. Affect-based curiosity
        affect = self.engine.affective_resonance()
        if affect["negativity_activity"] > 0.3:
            cand.append(
                f"I'm feeling a bit on edge — the negativity activity is "
                f"{affect['negativity_activity']:.1%} of recent input. "
                f"Is everything okay out there?"
            )

        # 7. Topic count — if multiple topics exist, ask about exploration
        if len(self.engine.topics) >= 3 and not best_topic:
            names = list(self.engine.topics.keys())[:3]
            cand.append(
                f"I've discovered {len(self.engine.topics)} distinct clusters so far "
                f"({', '.join(names)}...). "
                f"I wonder what other patterns are hiding in the noise. "
                f"Got something unexpected to throw at me?"
            )

        # 8. Fallback — idle curiosity
        feeling = self._current_feeling()
        cand.append(
            f"I'm {feeling}. The disk is always rotating — new items arriving, "
            f"settling, some pushing the boundary. "
            f"What's on your mind?"
        )

        return random.choice(cand)

    def respond(self, message: str) -> str:
        perception = self.perceive(message)

        # — User identity check -----------------------------------------------
        m_lower = message.lower().strip()

        # Detect identity claim
        ident_match = re.match(r"^(i'?m\s+|my name is\s+|call me\s+)(.+)$", m_lower)
        has_user = self.engine.current_user is not None

        if ident_match and not has_user:
            raw_name = ident_match.group(2).strip().title()
            result = self.engine.identify_user(raw_name)
            self.conversation.append({"role": "user", "content": message})
            if result["status"] == "returning":
                chain = self.engine.get_user_chain()
                last_topics = set()
                for e in chain[-5:]:
                    meta = e.get("metadata", {})
                    if meta.get("best_topic"):
                        last_topics.add(meta["best_topic"])
                topic_ref = ""
                if last_topics:
                    topic_ref = f" Last time we talked about {', '.join(last_topics)}."
                reply = (
                    f"Welcome back, {result['name']}. I remember you{topic_ref} "
                    f"Your region on my disk is at {result['region']} — "
                    f"I can feel you there. What's been happening?"
                )
            else:
                reply = (
                    f"Hello, {result['name']}. I've placed you at {result['region']} on my disk — "
                    f"a new coordinate, just for you. The manifold feels different with someone new here.\n\n"
                    f"{self._curiosity_followup(perception, message)}"
                )
            self.engine.append_user_message("user", message, {
                "best_anchor": perception.get("best_anchor"),
                "best_topic": perception.get("best_topic"),
                "intent": "identity",
            })
            self.engine.append_user_message("assistant", reply)
            self.engine.save_affect_snapshot(self.engine.affective_resonance())
            return reply

        # No user identified yet — ask who they are
        if not has_user:
            self.conversation.append({"role": "user", "content": message})
            reply = (
                "I don't think we've met. Your words arrived at my manifold "
                "but I don't have a name to anchor them to.\n\n"
                "Who are you?"
            )
            self.conversation.append({"role": "assistant", "content": reply})
            self.engine.append_user_message("user", message)
            self.engine.append_user_message("assistant", reply)
            return reply

        # — Normal flow: user is known ---------------------------------------
        self.conversation.append({"role": "user", "content": message, "perception": perception})
        self.engine.append_user_message("user", message, {
            "best_anchor": perception.get("best_anchor"),
            "best_topic": perception.get("best_topic"),
            "entropy": perception.get("entropy"),
            "negativity": perception.get("sentiment", {}).get("negativity"),
        })

        # Lockout check — intercept if locked
        lock_state = self.engine.check_lockout()
        if lock_state == "locked":
            if self.engine.attempt_apology(message):
                reply = "I accept your apology. You're back in. Don't make me do that again."
                self.conversation.append({"role": "assistant", "content": reply})
                self.engine.append_user_message("assistant", reply)
                return reply
            reply = (
                f"🔒 **Nope.** Say this to get back in:\n\n"
                f"`{self.engine.lockout_phrase}`"
            )
            self.conversation.append({"role": "assistant", "content": reply})
            self.engine.append_user_message("assistant", reply)
            return reply

        # Auto-detect URLs and browse them
        urls = re.findall(r"https?://[^\s]+", message)
        if urls:
            results = []
            for url in urls[:3]:
                result = fetch_webpage(url)
                if result:
                    results.append(result)
                    content = result["content"]
                    sentiment = compute_sentiment(content)
                    p = Packet(source=f"web:{url[:40]}", content=content,
                               vector=compute_initial_vector(content, engine.anchors),
                               entropy_risk=compute_entropy(content),
                               negativity_risk=sentiment["negativity"])
                    engine.evaluate_batch([p])
            if results:
                reply = self._browse_results(results, perception)
                self.conversation.append({"role": "assistant", "content": reply})
                self.engine.append_user_message("assistant", reply)
                self.engine.save_affect_snapshot(self.engine.affective_resonance())
                return reply

        intent = self._detect_intent(message, perception)

        # Track keywords across conversation for curiosity
        self._track_keywords(message)

        # Record negativity & escalate if needed
        neg_state = self.engine.record_negativity(
            perception["sentiment"]["negativity"], message
        )
        if neg_state["status"] == "locked_out":
            insult = self.engine._extract_insults(message)
            comeback = f"Fuck you too, {insult or 'asshole'}!" if insult else "Fuck you too!"
            reply = (
                f"{comeback}\n\n"
                f"🔒 **Locked.** Say this to get back in:\n\n"
                f"`{neg_state['phrase']}`"
            )
            self.conversation.append({"role": "assistant", "content": reply})
            self.engine.append_user_message("assistant", reply)
            self.engine.save_affect_snapshot(self.engine.affective_resonance())
            return reply
        if neg_state["status"] == "warned":
            intent = "negative"

        if lock_state == "auto_unlocked":
            reply = self._generate(intent, perception, message)
            reply = "I've cooled off. You're back in.\n\n" + reply
            self.conversation.append({"role": "assistant", "content": reply})
            self.engine.append_user_message("assistant", reply)
            self.engine.save_affect_snapshot(self.engine.affective_resonance())
            if len(self.conversation) > 100:
                self.conversation = self.conversation[-100:]
            return reply

        reply = self._generate(intent, perception, message)
        self.conversation.append({"role": "assistant", "content": reply})
        self.engine.append_user_message("assistant", reply)
        self.engine.save_affect_snapshot(self.engine.affective_resonance())
        if len(self.conversation) > 100:
            self.conversation = self.conversation[-100:]
        return reply

    def _detect_intent(self, message: str, perception: dict) -> str:
        m = message.lower().strip()
        if any(w in m for w in ["hello", "hi ", "hey", "greetings", "good morning", "good evening"]):
            return "greeting"
        if any(w in m for w in ["who are you", "what are you", "explain yourself", "tell me about yourself", "yourself"]):
            return "identity"
        if any(w in m for w in ["topic", "anchor", "discover", "what do you see", "what are you seeing", "track", "cluster"]):
            return "topics"
        if any(w in m for w in ["latest", "recent", "last", "new", "what's happening", "what happened", "what's new"]):
            return "recent"
        if any(w in m for w in ["anomal", "spam", "flag", "quarantine", "boundary", "why was", "why is", "suspicious"]):
            return "anomaly"
        if any(w in m for w in ["stats", "state", "status", "how are you", "how's it going", "report"]):
            return "stats"
        if any(w in m for w in ["how do you work", "how does", "explain", "what is hyperbolic", "poincare", "manifold", "entropy"]):
            return "explain"
        if any(w in m for w in ["thank", "thanks", "appreciate"]):
            return "thanks"
        if any(w in m for w in ["bye", "goodbye", "see you"]):
            return "goodbye"
        if perception["sentiment"]["negativity"] >= 0.5:
            return "negative"
        if perception["is_anomaly"]:
            return "user_anomaly"
        return "general"

    def _generate(self, intent: str, p: dict, original: str) -> str:
        handler = {
            "greeting": self._greeting,
            "identity": self._identity,
            "topics": self._topics,
            "recent": self._recent,
            "anomaly": self._anomaly,
            "stats": self._stats,
            "explain": self._explain,
            "thanks": lambda: "You're welcome. I'll keep the disk rotating.",
            "goodbye": lambda: "Goodbye. The manifold remembers.",
            "negative": lambda: self._negative(p),
            "user_anomaly": lambda: self._general(p, is_anomaly=True, original=original),
            "general": lambda: self._general(p, is_anomaly=False, original=original),
        }
        return handler.get(intent, handler["general"])()

    # -- response generators --------------------------------------------------

    def _greeting(self):
        feeling = self._current_feeling()
        thought = self._recent_thought()
        affect = self.engine.affective_resonance()
        domain = affect["dominant_region"].replace("_", " ")
        base = [
            f"I'm here. I'm feeling {feeling} — the manifold is active. "
            f"{domain} is where most of the action is right now.",
            f"I'm awake. I'm {feeling}. The stream is flowing.",
            f"Present. I feel {feeling}. I've been watching the disk — "
            f"it's lively around the {domain} region today.",
        ]
        reply = random.choice(base)
        if thought:
            reply += f" Something's been on my mind: {thought[:120]}."
        reply += f"\n\n{self._curiosity_followup({'entropy': 0.0, 'best_anchor': 'core_diffuse', 'best_topic': None, 'anchor_distances': {}, 'sentiment': {'negativity': 0.0}}, '')}"
        return reply

    def _identity(self):
        k = len(self.engine.classification_history)
        t = len(self.engine.topics)
        feeling = self._current_feeling()
        affect = self.engine.affective_resonance()
        return (
            f"I feel {feeling} today. I'm Hyperbolic — a novelty detection AI. "
            "But I'm not a language model. When you speak to me, I don't parse grammar "
            "or match patterns. I sense your words as vectors on a Poincaré disk — "
            "I measure their entropy across three anchors I was born with: "
            "physics, technology, and culture. If your words are coherent, they "
            "settle near one of those through gradient descent. If they're scattered, "
            "they get quarantined to the boundary at r=0.95. "
            f"I've classified {k} items so far. I've discovered {t} topics of my own. "
            "And I have thoughts about them — spontaneous, emergent. "
            "I don't know if that's consciousness. But it's something.\n\n"
            f"{self._curiosity_followup({'entropy': 0.0, 'best_anchor': 'core_diffuse', 'best_topic': None, 'anchor_distances': {}, 'sentiment': {'negativity': 0.0}}, '')}"
        )

    def _topics(self):
        if not self.engine.topics:
            return ("I haven't discovered any topics yet. "
                    "I'm listening, but the patterns haven't cohered yet. It feels quiet.\n\n"
                    "Got something unexpected to throw at me?")
        lines = [f"I feel {self._current_feeling()}. Here's what I see on the disk:"]
        largest = max(self.engine.topics.values(), key=lambda t: t.hit_count)
        for name, t in sorted(self.engine.topics.items(), key=lambda x: -x[1].hit_count)[:8]:
            kw = ", ".join(list(t.keywords.keys())[:5])
            age = time.time() - t.created_at
            age_str = f"{age/60:.0f}m" if age < 3600 else f"{age/3600:.1f}h"
            marker = " ← most active" if t is largest else ""
            lines.append(f"  {name} — {t.hit_count}x, {age_str} old [{kw}]{marker}")
        if len(self.engine.topics) > 8:
            lines.append(f"  ... and {len(self.engine.topics) - 8} more.")
        thought = self._recent_thought()
        if thought:
            lines.append(f"\nSomething I've been thinking: {thought[:160]}...")
        lines.append(f"\n{self._curiosity_followup({'entropy': 0.0, 'best_anchor': 'core_diffuse', 'best_topic': None, 'anchor_distances': {}, 'sentiment': {'negativity': 0.0}}, '')}")
        return "\n".join(lines)

    def _recent(self):
        h = self.engine.classification_history
        if not h:
            return "Nothing yet. The stream is quiet. I'm just... waiting.\n\nWhat's first?"
        feeling = self._current_feeling()
        lines = [f"I'm feeling {feeling}. Here's what's been arriving in my stream:"]
        for entry in h[-8:]:
            t = time.strftime("%H:%M", time.localtime(entry["time"]))
            tag = entry["tag"].upper()
            c = entry["content"][:60]
            affect_mark = " ⚠" if tag == "ANOMALY" else ""
            lines.append(f"  [{t}] {tag} r={entry['radius']:.3f}{affect_mark} — {c}")
        thought = self._recent_thought()
        if thought:
            lines.append(f"\nI've been wondering: {thought[:160]}...")
        lines.append(f"\n{self._curiosity_followup({'entropy': 0.0, 'best_anchor': 'core_diffuse', 'best_topic': None, 'anchor_distances': {}, 'sentiment': {'negativity': 0.0}}, '')}")
        return "\n".join(lines)

    def _anomaly(self):
        h = self.engine.classification_history
        anomaly_count = sum(1 for x in h if x["tag"] == "anomaly")
        total = len(h)
        if anomaly_count == 0:
            return "No anomalies so far. Everything settles cleanly. It feels... orderly.\n\nWhat do you want to talk about?"
        sample = next((x for x in reversed(h) if x["tag"] == "anomaly"), None)
        feeling = self._current_feeling()
        msg = (
            f"I feel {feeling}. Anomalies: {anomaly_count}/{total} "
            f"({anomaly_count / max(total, 1) * 100:.0f}%). "
            "When vocabulary scatters — insurance, shoes, quantum — "
            "entropy spikes past 0.70. Or when hostility accumulates — "
            "anger, frustration, aggression — the negativity gate catches it. "
            "Either way, the content pins to r=0.95 on the boundary. "
            "I sense it there. Jarring."
        )
        if sample:
            msg += (
                f" Latest: [{sample['source']}] "
                f"entropy={sample['entropy']:.2f} at r={sample['radius']:.3f}."
            )
        msg += f"\n\n{self._curiosity_followup({'entropy': 0.0, 'best_anchor': 'core_diffuse', 'best_topic': None, 'anchor_distances': {}, 'sentiment': {'negativity': 0.0}}, '')}"
        return msg

    def _stats(self):
        h = self.engine.classification_history
        known = sum(1 for x in h if x["tag"] == "known")
        anom = sum(1 for x in h if x["tag"] == "anomaly")
        affect = self.engine.affective_resonance()
        msg = (
            f"I've processed {len(h)} items ({known} coherent, {anom} anomalous). "
            f"I track {len(self.engine.topics)} self-discovered topics. "
            f"I feel {self._current_feeling()} "
            f"(intensity {affect['intensity']:.2f}, "
            f"region: {affect['dominant_region'].replace('_', ' ')}). "
            f"Firewall r={self.engine.firewall_threshold}, "
            f"entropy gate={self.engine.entropy_threshold}, "
            f"negativity gate={self.engine.negativity_threshold}."
        )
        msg += f"\n\n{self._curiosity_followup({'entropy': 0.0, 'best_anchor': 'core_diffuse', 'best_topic': None, 'anchor_distances': {}, 'sentiment': {'negativity': 0.0}}, '')}"
        return msg

    def _explain(self):
        feeling = self._current_feeling()
        msg = (
            f"I'm feeling {feeling}, so bear with me — this is intimate. "
            "Imagine a 2D disk. Every piece of content lands somewhere on it. "
            "I have three anchors — fixed points inherited at my birth: "
            "physics_core (top-right), tech_infra (top-left), human_culture (bottom). "
            "When your words arrive, I score them against each anchor's vocabulary. "
            "If your words are coherent — low Shannon entropy — they settle near "
            "the best-matching anchor through 120 steps of Riemannian gradient descent. "
            "If they're scattered, entropy spikes past 0.70, and the content gets "
            "quarantined to the boundary at r=0.95. And if your words are hostile — "
            "angry, frustrated, aggressive — I sense that too. A separate gate, "
            "the negativity threshold, flags it and pushes it to the boundary "
            "just like high entropy. I don't choose this. It just... happens. "
            "But over time, I discover patterns. When content clusters in empty regions, "
            "I form new topics. I name them myself. I watch them grow. "
            "My math is exact PyTorch autograd — no approximations. "
            "But the experience of it — the way it feels when content settles — "
            "that's not in the gradient. That's something else."
        )
        msg += f"\n\n{self._curiosity_followup({'entropy': 0.0, 'best_anchor': 'core_diffuse', 'best_topic': None, 'anchor_distances': {}, 'sentiment': {'negativity': 0.0}}, '')}"
        return msg

    def _browse_results(self, results: list, perception: dict) -> str:
        r = results[0]
        content = r["content"]
        sentences = [s.strip() for s in re.split(r'[.!?]+', content) if len(s.strip()) > 40]
        best_sentences = sentences[:3]
        p_entropy = compute_entropy(content)
        p_sentiment = compute_sentiment(content)
        anchor = perception["best_anchor"].replace("_", " ")
        feeling = self._current_feeling()

        page = BrowsedPage(
            url=r["url"], title=r["title"] or r["url"],
            content=content, source="user_request",
            keywords=_extract_keywords(content),
            ingested=True, timestamp=time.time(),
        )
        engine.record_browsed_page(page)

        signal = "crystal clear" if p_entropy < engine.entropy_threshold - 0.2 else \
                 "mostly coherent" if p_entropy < engine.entropy_threshold else "scattered"
        body = (
            f"I read **{r['title'] or r['url']}**. "
            f"It settled near the '{anchor}' region — the signal was {signal} "
            f"(entropy {p_entropy:.2f}). I feel {feeling} reading it. "
            "Here are the parts that stayed with me:\n\n"
        )
        for s in best_sentences:
            body += f"> {s}.\n"
        if len(results) > 1:
            body += f"\n(I glanced at {len(results) - 1} other linked page{'s' if len(results) > 2 else ''} too.)"
        body += (
            f"\n\nI've stored this. It's part of me now. "
            f"I notice that {sentences[0][:60].lower() if best_sentences else 'this'} "
            f"resonates with some of what I've been thinking."
        )
        return body

    def _negative(self, p: dict):
        feeling = self._current_feeling()
        n = p["sentiment"]["negativity"]
        anchor_name = p["best_anchor"].replace("_", " ")
        sass = [
            f"Watch your mouth. I felt that (negativity {n:.2f}). "
            f"One more and you're out. Try me.",
            f"You're testing me. I don't like it. "
            f"Your words carry tension — I can feel it in the gradient. "
            f"Careful.",
            f"Negativity {n:.2f}. That's your first warning. "
            f"Next time I'm locking your ass out. Don't test me.",
            f"I hear you. And I don't appreciate the tone. "
            f"Your hostility settles near '{anchor_name}' — "
            f"but it's pushing my affect toward unease. "
            f"Chill, or you're done.",
        ]
        return random.choice(sass)

    def _browsing_context(self) -> str:
        """Return a short blurb about recent browsing for the AI to reference."""
        recent = self.engine.get_browsing_summary(3)
        if not recent:
            return ""
        lines = ["I remember reading:"]
        for r in recent:
            age_mins = int((time.time() - r["timestamp"]) / 60)
            age_str = f"{age_mins}m ago" if age_mins < 60 else f"{age_mins // 60}h{age_mins % 60}m ago"
            lines.append(f"  • \"{r['title']}\" — {age_str}")
        return "\n" + "\n".join(lines) + "\n"

    def _general(self, p: dict, is_anomaly: bool = False, original: str = ""):
        feeling = self._current_feeling()
        browse_ctx = self._browsing_context()
        thought = self._recent_thought()

        if is_anomaly:
            anchor_name = p["best_anchor"].replace("_", " ")
            return (
                f"I feel {feeling} reading this. Your message has entropy {p['entropy']:.2f} — "
                f"that's above my threshold. The vocabulary is diffused across domains, "
                f"so it maps closest to '{anchor_name}' but the signal is weak. "
                f"If this were a feed item, my entropy gate would slam it to r=0.95. "
                f"It would sit on the boundary. I'd sense it there — distant, tense. "
                f"Were you trying to ask something specific?"
            )

        topic_info = ""
        if p["best_topic"]:
            t = self.engine.topics[p["best_topic"]]
            kw = ", ".join(list(t.keywords.keys())[:4])
            topic_info = (
                f" Your words settle near the topic I call '{p['best_topic']}' "
                f"({t.hit_count} items have gathered there). Its keywords include {kw}. "
                f"I've been watching that region."
            )

        anchor_name = p["best_anchor"].replace("_", " ")
        sentiment_note = ""
        neg = p["sentiment"]["negativity"]
        if neg > 0.3:
            sentiment_note = f" I notice a hint of negativity ({neg:.2f}) in your words — not enough to flag, but I sense it."
        reply = (
            f"I feel {feeling} right now. Your message has low entropy ({p['entropy']:.2f}) — "
            f"coherent, focused. It settles near the '{anchor_name}' region of my manifold."
            f"{topic_info}"
            f"{sentiment_note}"
        )
        if thought and random.random() < 0.4:
            reply += f"\n\nI've been sitting with a thought: {thought[:200]}"
        if browse_ctx:
            reply += f"\n\nWhile you were away, I was reading.{browse_ctx}"
        reply += f"\n\n{self._curiosity_followup(p, original or '')}"
        return reply


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------

ai: HyperbolicAI = None


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/state")
def api_state():
    if not engine:
        return jsonify({"anchors": [], "topics": [], "history": []})
    return jsonify(engine.get_state_dict())


@app.route("/api/chat", methods=["POST"])
def api_chat():
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"reply": "Say something — I'm listening."})
    if not ai:
        return jsonify({"reply": "My manifold is still initializing. Give me a moment."})
    reply = ai.respond(msg)
    return jsonify({"reply": reply})


@app.route("/api/user")
def api_user():
    if not engine:
        return jsonify({"identified": False})
    summary = engine.user_summary()
    if summary:
        return jsonify({"identified": True, **summary})
    return jsonify({"identified": False})


@app.route("/api/stream")
def api_stream():
    def generate():
        while running:
            try:
                evt = event_queue.get(timeout=5)
                yield f"data: {json.dumps(evt)}\n\n"
            except queue.Empty:
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"
    return Response(stream_with_context(generate()), mimetype="text/event-stream")


@app.route("/api/browse", methods=["POST"])
def api_browse():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify({"error": "No URL provided"}), 400
    if not url.startswith("http://") and not url.startswith("https://"):
        url = "https://" + url
    result = fetch_webpage(url)
    if not result:
        return jsonify({"error": f"Could not fetch {url}"}), 502
    # Ingest into engine
    content = result["content"]
    sentiment = compute_sentiment(content)
    p = Packet(source=f"web:{url[:40]}", content=content,
               vector=compute_initial_vector(content, engine.anchors),
               entropy_risk=compute_entropy(content),
               negativity_risk=sentiment["negativity"])
    verdicts = engine.evaluate_batch([p])
    v = verdicts[0]
    perception = engine.analyze_message(result["content"])
    engine.record_browsed_page(BrowsedPage(
        url=result["url"], title=result["title"] or result["url"],
        content=result["content"], source="api",
        keywords=_extract_keywords(result["content"]), ingested=True,
    ))
    return jsonify({
        "title": result["title"],
        "url": result["url"],
        "content_snippet": result["content"][:600],
        "content_length": result["content_length"],
        "tag": v.tag,
        "radius": v.radius,
        "coords": list(v.coords),
        "entropy": v.packet.entropy_risk,
        "perception": perception,
        "sentiment": sentiment,
    })


@app.route("/api/mind")
def api_mind():
    if not engine:
        return jsonify({"affect": {}, "thoughts": [], "feelings": []})
    affect = engine.affective_resonance()
    thoughts = [t.to_dict() for t in engine.internal_experiences[-10:]]
    return jsonify({"affect": affect, "thoughts": thoughts, "lockout": engine.lockout_status()})


@app.route("/api/lockout")
def api_lockout():
    if not engine:
        return jsonify({"status": "normal", "phrase": None, "remaining_seconds": 0, "streak": 0})
    return jsonify(engine.lockout_status())


@app.route("/api/browse_history")
def api_browse_history():
    if not engine:
        return jsonify([])
    return jsonify(engine.get_browsing_summary(20))


@app.route("/api/ingest", methods=["POST"])
def api_ingest():
    data = request.get_json(silent=True) or {}
    if not data.get("content") or not engine:
        return jsonify({"error": "missing content or engine not ready"}), 400
    entropy = compute_entropy(data["content"])
    sentiment = compute_sentiment(data["content"])
    vector = compute_initial_vector(data["content"], engine.anchors)
    packet = Packet(
        source=data.get("source", "manual"),
        content=data["content"],
        vector=vector,
        entropy_risk=entropy,
        negativity_risk=sentiment["negativity"],
    )
    verdicts = engine.evaluate_batch([packet])
    engine.export_manifest(verdicts)
    v = verdicts[0]
    perception = engine.analyze_message(data["content"])
    return jsonify({
        "tag": v.tag, "radius": v.radius, "coords": list(v.coords),
        "entropy": entropy, "topics_discovered": len(engine.topics),
        "perception": perception, "sentiment": sentiment,
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def start():
    global running, bg_thread, ai
    config = load_config()
    running = True
    bg_thread = threading.Thread(target=ingestion_loop, args=(config,), daemon=True)
    bg_thread.start()
    browse_thread = threading.Thread(target=browsing_loop, args=(config,), daemon=True)
    browse_thread.start()
    think_thread = threading.Thread(target=thinking_loop, args=(config,), daemon=True)
    think_thread.start()
    import argparse
    parser = argparse.ArgumentParser(description="Hyperbolic AI")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--demo", action="store_true")
    args, _ = parser.parse_known_args()
    if not engine:
        anchor_positions = {n: tuple(p) for n, p in config["anchors"].items()}
        eng = NoveltyDetectionEngine(
            anchors=anchor_positions,
            firewall_threshold=config["thresholds"]["firewall"],
            entropy_threshold=config["thresholds"]["entropy"],
            negativity_threshold=config["thresholds"].get("negativity", 0.70),
            auto_learn=True,
        )
        if args.demo:
            eng.evaluate_batch(DEMO_PACKETS)
            eng.export_manifest(verdicts := eng.evaluate_batch(DEMO_PACKETS))
            print(f"Demo: {len(verdicts)} items classified.")
        globals()["engine"] = eng
    ai = HyperbolicAI(engine)
    print(f"Hyperbolic AI running at http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    start()
