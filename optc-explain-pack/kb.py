import json
import re
from dataclasses import dataclass, field
from pathlib import Path

NULLS = {"", "-", "?", "none", "null", "n/a"}
ATTACKER_FIELDS = {"CommandLine", "ParentCommandLine", "TargetFilename", "Details", "PipeName",
                   "QueryName", "QueryResults", "NewName"}


def _norm(v):
    return str(v).strip().lower()


def _basename(v):
    s = _norm(v).replace("/", "\\")
    return s.rsplit("\\", 1)[-1]


def _hex(v):
    try:
        return int(str(v).strip(), 16)
    except (TypeError, ValueError):
        return None


def _present(v):
    return v is not None and _norm(v) not in NULLS


class Condition:
    OPS = {"equals", "in", "endswith", "startswith", "contains", "regex", "exists",
           "hex_in", "hex_mask_all", "not_startswith", "not_contains", "not_regex", "not_endswith"}

    def __init__(self, spec):
        unknown = set(spec) - self.OPS
        if unknown:
            raise ValueError(f"unknown ops {unknown}")
        self.spec = {}
        for op, val in spec.items():
            if op in ("regex", "not_regex"):
                self.spec[op] = re.compile(val, re.IGNORECASE)
            elif op == "hex_in":
                self.spec[op] = {_hex(x) for x in val}
            elif op == "hex_mask_all":
                self.spec[op] = _hex(val)
            elif op == "exists":
                self.spec[op] = bool(val)
            else:
                vals = val if isinstance(val, list) else [val]
                self.spec[op] = [_norm(x) for x in vals]

    def test(self, value):
        if not _present(value):
            return self.spec.get("exists") is False
        v = _norm(value)
        for op, ref in self.spec.items():
            if op == "exists":
                ok = ref
            elif op in ("equals", "in"):
                ok = v in ref
            elif op == "endswith":
                ok = any(v.endswith(r) for r in ref)
            elif op == "startswith":
                ok = any(v.startswith(r) for r in ref)
            elif op == "contains":
                ok = any(r in v for r in ref)
            elif op == "not_startswith":
                ok = not any(v.startswith(r) for r in ref)
            elif op == "not_endswith":
                ok = not any(v.endswith(r) for r in ref)
            elif op == "not_contains":
                ok = not any(r in v for r in ref)
            elif op == "regex":
                ok = ref.search(str(value)) is not None
            elif op == "not_regex":
                ok = ref.search(str(value)) is None
            elif op == "hex_in":
                ok = _hex(value) in ref
            elif op == "hex_mask_all":
                h = _hex(value)
                ok = h is not None and (h & ref) == ref
            else:
                ok = False
            if not ok:
                return False
        return True


@dataclass
class Clause:
    technique: str
    cid: str
    sub: str | None
    eids: set | None
    conds: list
    cmps: list
    source: str = ""

    @property
    def attested(self):
        names = {n for ns, _ in self.conds for n in ns} | {c[k] for c in self.cmps for k in ("a", "b")}
        return not (names & ATTACKER_FIELDS)

    def match(self, fields, eid=None, strict_eid=False):
        if self.eids is not None:
            if eid is None and strict_eid:
                return None
            if eid is not None and eid not in self.eids:
                return None
        used = []
        for names, cond in self.conds:
            hit = None
            for n in names:
                if n in fields and cond.test(fields[n]):
                    hit = n
                    break
            if hit is None:
                return None
            if hit not in used:
                used.append(hit)
        for c in self.cmps:
            a, b = fields.get(c["a"]), fields.get(c["b"])
            if not (_present(a) and _present(b)):
                return None
            if c["op"] == "neq":
                ok = _norm(a) != _norm(b)
            elif c["op"] == "basename_neq":
                ok = _basename(a) != _basename(b)
            elif c["op"] == "eq":
                ok = _norm(a) == _norm(b)
            else:
                raise ValueError(f"unknown cmp {c['op']}")
            if not ok:
                return None
            for n in (c["a"], c["b"]):
                if n not in used:
                    used.append(n)
        return used


@dataclass
class Witness:
    technique: str
    sub: str | None
    clause: str
    event: int
    eid: int | None
    evidence: list = field(default_factory=list)

    def key(self):
        return (self.technique, self.event, tuple(f for f, _ in self.evidence))

    @property
    def attested(self):
        return not ({f for f, _ in self.evidence} & ATTACKER_FIELDS)

    def as_items(self):
        return [{"event": self.event, "eid": self.eid, "field": f, "value": v} for f, v in self.evidence]


class KB:
    def __init__(self, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        self.version = raw.get("version")
        self.enum = dict(raw.get("enum", {}))
        self.names = {}
        self.clauses = []
        for t, spec in raw["techniques"].items():
            self.names[t] = spec.get("name", t)
            self.enum.setdefault(t, self.names[t])
            for c in spec["clauses"]:
                conds = [(tuple(k.split("|")), Condition(v)) for k, v in c.get("all", {}).items()]
                self.clauses.append(Clause(
                    technique=t, cid=c["id"], sub=c.get("sub"),
                    eids=set(c["eid"]) if c.get("eid") else None,
                    conds=conds, cmps=c.get("cmp", []), source=c.get("source", "")))
        self.techniques = sorted(self.names)
        self._by_id = {(c.technique, c.cid): c for c in self.clauses}

    def covers(self, technique):
        return technique[:5] in self.names

    def satisfied_clause(self, technique, items):
        by_event, eids = {}, {}
        for it in items:
            by_event.setdefault(it["event"], {})[it["field"]] = it["value"]
            eids[it["event"]] = it.get("eid")
        for c in self.clauses:
            if c.technique != technique[:5]:
                continue
            for ev, fields in by_event.items():
                if c.match(fields, eids.get(ev), strict_eid=True) is not None:
                    return c
        return None

    def minimize(self, technique, items):
        cur = list(items)
        changed = True
        while changed:
            changed = False
            for i in range(len(cur)):
                trial = cur[:i] + cur[i + 1:]
                if trial and self.entails(technique, trial):
                    cur = trial
                    changed = True
                    break
        return cur

    def witnesses(self, events, minimize=True):
        out, seen = [], set()
        for e in events:
            for c in self.clauses:
                used = c.match(e.get("fields", {}), e.get("eid"))
                if used is None:
                    continue
                w = Witness(c.technique, c.sub, c.cid, e["idx"], e.get("eid"),
                            [(f, e["fields"][f]) for f in used])
                if minimize and len(w.evidence) > 1:
                    items = self.minimize(c.technique, w.as_items())
                    if len(items) < len(w.evidence):
                        c2 = self.satisfied_clause(c.technique, items)
                        w = Witness(c.technique, c2.sub, c2.cid, e["idx"], e.get("eid"),
                                    [(it["field"], it["value"]) for it in items])
                if w.key() not in seen:
                    seen.add(w.key())
                    out.append(w)
        return out

    def candidates(self, events):
        return sorted({w.technique for w in self.witnesses(events)})

    def entails(self, technique, items):
        return self.satisfied_clause(technique, items) is not None

    def minimal(self, technique, items):
        if not self.entails(technique, items):
            return False
        return all(not self.entails(technique, items[:i] + items[i + 1:]) for i in range(len(items)))
