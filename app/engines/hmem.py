# Minimal H-MEM: tiered memory store + hybrid retrieval + audit ledger.
# Layers: short_term (this run only), working and long_term (saved to disk).
import json
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LAYER_WEIGHT = {"long_term": 1.0, "working": 0.85, "short_term": 0.7}
PERSISTED = ("working", "long_term")
_STOP = {"the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "are", "was",
         "were", "for", "my", "i", "me", "it", "this", "that", "with", "do", "did",
         "what", "why", "how", "which"}


def _now():
    return datetime.now(timezone.utc).isoformat()


def _tok_list(text):
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in _STOP:
            continue
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]          # crude plural stripping: "screens" -> "screen"
        out.append(w)
    return out


def _tokens(text):
    return set(_tok_list(text))


class HMem:
    def __init__(self, store_path="data/hmem_store.json",
                 ledger_path="logs/hmem_ledger.jsonl", run_id=None):
        self.store_path = Path(store_path)
        self.ledger_path = Path(ledger_path)
        self.run_id = run_id or f"run-{uuid.uuid4().hex[:6]}"
        self.memories = {}
        if self.store_path.exists():                     # warm start
            self.memories = json.loads(self.store_path.read_text())

    # ---- persistence and audit ----
    def _save(self):
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        keep = {k: v for k, v in self.memories.items() if v["layer"] in PERSISTED}
        self.store_path.write_text(json.dumps(keep, indent=2))

    def _log(self, memory_id, action, actor, previous=None, new=None, details=None):
        entry = {"timestamp": _now(), "run_id": self.run_id, "memory_id": memory_id,
                 "action": action, "actor": actor, "previous_state": previous,
                 "new_state": new, "details": details or {}}
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.ledger_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    # ---- write ----
    def add(self, content, type="factual", layer="short_term", tags=None,
            entities=None, source="agent", confidence=0.6, agent_id="system",
            task_id=None, supersedes=None, derived_from=None):
        norm = " ".join(content.lower().split())
        for rec in self.memories.values():               # exact duplicate: just bump
            if (" ".join(rec["content"].lower().split()) == norm
                    and not rec.get("superseded_by")):
                rec["access_count"] += 1
                self._log(rec["id"], "duplicate_bumped", agent_id)
                self._save()
                return rec
        mid = f"mem-{uuid.uuid4().hex[:8]}"
        rec = {"id": mid, "content": content, "type": type, "layer": layer,
               "timestamp": _now(), "last_accessed": _now(), "access_count": 0,
               "tags": tags or [], "entities": entities or [],
               "confidence": confidence, "source": source, "agent_id": agent_id,
               "run_id": self.run_id, "task_id": task_id, "supersedes": supersedes,
               "derived_from": derived_from or [], "superseded_by": None}
        if supersedes in self.memories:
            self.memories[supersedes]["superseded_by"] = mid
            self._log(supersedes, "superseded", agent_id, details={"by": mid})
        self.memories[mid] = rec
        self._log(mid, "create", agent_id, new=rec)
        self._save()
        return rec

    def mine_event(self, event_type, summary, agent_id, tags=None,
                   entities=None, task_id=None):
        """No-model capture: store a concise summary of a runtime event."""
        kind = "procedural" if event_type == "run_outcome" else "factual"
        return self.add(summary[:300], type=kind, layer="short_term",
                        tags=[event_type, *(tags or [])], entities=entities,
                        source=f"event:{event_type}", agent_id=agent_id,
                        task_id=task_id)

    # ---- read ----
    def retrieve(self, query, top_n=5, threshold=0.05, entities=None,
                 agent_id="system"):
        recs = list(self.memories.values())
        if not recs:
            return self._payload(query, [], "no memories stored")
        docs = [" ".join([r["content"], *r["tags"], *r["entities"]]) for r in recs]
        try:
            X = TfidfVectorizer(tokenizer=_tok_list, token_pattern=None,
                                lowercase=False).fit_transform(docs + [query])
            sims = cosine_similarity(X[-1], X[:-1])[0]
        except ValueError:                               # nothing left to compare
            sims = [0.0] * len(recs)
        q_tok = _tokens(query)
        want = {e.lower() for e in (entities or [])}
        now = datetime.now(timezone.utc)
        scored = []
        for rec, sim in zip(recs, sims):
            meta = _tokens(" ".join(rec["tags"]))
            tag = len(q_tok & meta) / max(1, len(meta))
            ent = 1.0 if {e.lower() for e in rec["entities"]} & (q_tok | want) else 0.0
            if sim <= 0 and tag <= 0 and ent <= 0:
                continue                                 # recency alone never qualifies
            age = (now - datetime.fromisoformat(rec["timestamp"])).days
            base = 0.6 * float(sim) + 0.15 * tag + 0.15 * ent + 0.10 * math.exp(-age / 30)
            score = base * LAYER_WEIGHT[rec["layer"]] * (0.5 + 0.5 * rec["confidence"])
            if rec.get("superseded_by"):
                score *= 0.3                             # replaced memories rank low
            scored.append((score, rec))
        scored.sort(key=lambda s: s[0], reverse=True)
        picked = [(s, r) for s, r in scored if s >= threshold][:top_n]
        for s, r in picked:
            r["access_count"] += 1
            r["last_accessed"] = _now()
            self._log(r["id"], "read", agent_id,
                      details={"score": round(s, 3), "query": query[:80]})
        if picked:
            self._save()
        return self._payload(query, picked,
                             f"{len(picked)} of {len(recs)} memories above threshold {threshold}")

    @staticmethod
    def _payload(query, picked, reason):
        return {"query": query, "strategy": "task_focused", "retrieval_reason": reason,
                "memories": [{"id": r["id"], "content": r["content"], "type": r["type"],
                              "layer": r["layer"], "confidence": round(r["confidence"], 2),
                              "score": round(s, 3), "tags": r["tags"]}
                             for s, r in picked]}

    # ---- feedback and lifecycle ----
    def feedback(self, memory_id, verdict, actor="evaluator"):
        rec = self.memories[memory_id]
        prev = {"confidence": rec["confidence"], "access_count": rec["access_count"]}
        if verdict == "helpful":
            rec["access_count"] += 1
            rec["confidence"] = min(1.0, rec["confidence"] + 0.05)
        elif verdict == "not_helpful":
            rec["confidence"] = max(0.0, rec["confidence"] - 0.05)
        elif verdict == "wrong":
            rec["confidence"] = max(0.0, rec["confidence"] - 0.2)
            if "needs-review" not in rec["tags"]:
                rec["tags"].append("needs-review")
        else:
            raise ValueError("verdict must be helpful, not_helpful or wrong")
        self._log(memory_id, f"feedback_{verdict}", actor, previous=prev,
                  new={"confidence": rec["confidence"]})
        self._save()

    def promote(self, memory_id, layer, actor="system"):
        rec = self.memories[memory_id]
        prev = rec["layer"]
        rec["layer"] = layer
        self._log(memory_id, "promote", actor, previous={"layer": prev},
                  new={"layer": layer})
        self._save()

    def end_run(self):
        """Move this run's short-term memories into working memory (saved)."""
        n = 0
        for rec in list(self.memories.values()):
            if rec["layer"] == "short_term":
                self.promote(rec["id"], "working", actor="end_run")
                n += 1
        return n
