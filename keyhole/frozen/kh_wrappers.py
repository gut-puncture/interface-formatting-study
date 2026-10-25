"""Keyhole v2 wrapper generator -- FROZEN for FINAL_SPEC.md (derived from wrappers_proto/kh_wrappers.py).
Changes vs proto: three nested doses S/M/L = 256/2048/16384 nominal Qwen3 tokens (factor 8);
table family and mid layout removed; 16 fillers per family; tail_budget is a build
parameter (64 = primary, 512 = escalation E1);
reference token counter = Qwen3 tokenizer;
dose filling uses additive per-unit token counts (O(n), not quadratic);
two null wrappers only (null_pre, null_post), both used for the noise floor.

Deterministic templates + seeded filler; no LLM rewriting. One stimulus text
per (item, condition), IDENTICAL for every model. The canonical question
string is inserted verbatim (never escaped) in every condition.

Prompt = chat_template(user = BLOCK + "\n\n" + TAIL), no system message.
BLOCK  = "Question: " + q                                   (plain)
       = one neutral sentence before / after it               (null_pre / null_post)
       = head + PRE(dose) + QUNIT(q) + POST + foot           (wrapper family)
PRE grows with dose (units added outward, nearest first); POST is fixed
(~TAIL_BUDGET tokens) at every dose, so everything from the question to the
end of the prompt is token-identical across doses of one filler.
A filler depends only on (wrapper, filler_id); many items share one filler,
so the pre-question prefix is shared (KV/prefix caching) and the same filler
can host different questions (swap controls).
"""
import hashlib, json, random, re, unicodedata

ANCHOR = "Question: "
TAIL = "Answer the question above. Reply with only the answer, in plain text."
SEP = "\n\n"
FORBIDDEN = set('"\\<>&|{}`\t\n\r\u2028\u2029')
DOSES = {"S": 256, "M": 2048, "L": 16384}  # nominal wrapper tokens excl. question line (Qwen3 counter)
TAIL_BUDGET = 64       # post-question wrapper tokens, identical at every dose (E1: 512)
MAX_Q_TOKENS = 160     # interior span cap (any model tokenizer)
MAX_Q_TO_END = {64: 448, 512: 900}  # question start -> last prompt token, any model (< Gemma SWA 1024)
N_FILLERS = 16         # fillers per family; item -> filler by hash

# ------------------------------------------------------------ canonical question
def canonical_question(q: str):
    """Shared canonical text (used in plain AND all wrapped prompts) or None."""
    q = unicodedata.normalize("NFC", q).replace("\u00a0", " ").replace("\u2009", " ")
    q = re.sub(r"\s+", " ", q).strip()
    # balanced straight double quotes -> curly pair (same canonical text in ALL
    # conditions, so plain and wrapped stay identical; avoids JSON/HTML escaping)
    if q.count('"') and q.count('"') % 2 == 0:
        parts = q.split('"')
        q = "".join(p + ("\u201c" if k % 2 == 0 else "\u201d") if k < len(parts) - 1 else p for k, p in enumerate(parts))
    if any(c in FORBIDDEN for c in q): return None       # would need escaping somewhere
    if q.count("?") > 1: return None                      # single question mark only
    if not (len(q) > 3 and q[-1] in "?." and q[-2].isalnum()): return None
    return q

def interior(q):
    """Patched/compared span text = ' ' + q[:-1] (space after 'Question:' through
    the last alphanumeric char). The terminal mark fuses with the next char in
    every BPE tested ('?\"', '?</', '?\\n'), so it is excluded."""
    return " " + q[:-1]

FREE_L = r"(?:^|(?<=[\s(\[\"'“]))"          # free-standing: preceded by start/space/open-bracket/quote
FREE_R = r"(?=$|[\s)\],;!\"'”]|[.:](?:\s|$))"  # followed by end/space/close/comma or sentence-final . :
def leak_in(ans, low_text):
    """Answer alias occurs FREE-STANDING (as a word/number of its own).
    Occurrences embedded in timestamps, dates, ids, paths, key=value pairs
    (e.g. '17' in '10:17:05', '200' in 'status=200') are not leaks; they are
    counted separately by embedded_count() and used as a covariate."""
    return re.search(FREE_L + re.escape(ans.lower()) + FREE_R, low_text) is not None

def embedded_count(ans, low_text):
    return len(re.findall(r"(?<![0-9a-z])" + re.escape(ans.lower()) + r"(?![0-9a-z])", low_text))

# ------------------------------------------------------------------ filler text
FIRST = ["Dana", "Priya", "Tomas", "Ines", "Kofi", "Mei", "Lars", "Amara", "Ravi", "Elena",
         "Jonah", "Sana", "Felix", "Noor", "Mateo", "Hana", "Owen", "Leila", "Yuki", "Bram"]
LAST = ["Whitfield", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Haddad", "Novak", "Mensah",
        "Castillo", "Brennan", "Kowalski", "Ferreira", "Iyer", "Sorensen", "Duval"]
TEAMS = ["facilities", "billing", "onboarding", "procurement", "it-support", "logistics"]
G_SUBJ = ["The printer on the third floor", "The badge reader at the east entrance", "The shared calendar",
          "The weekly inventory sheet", "The kitchen dishwasher", "The visitor log", "The meeting room projector",
          "The recycling schedule", "The supplier price sheet", "The heating in the north wing",
          "The onboarding checklist", "The cleaning rota for the lobby", "The spare monitor near the copier",
          "The coffee machine", "The courier pickup window", "The team folder", "The storage room lock",
          "The air filter in the server closet", "The name plate order", "The water dispenser",
          "The lift in block B", "The parking permit list", "The training room booking", "The desk plan"]
G_PRED = ["was serviced", "was updated", "was checked", "was moved", "was replaced", "was cleaned",
          "was reviewed", "was archived", "was reset", "was rescheduled", "was labelled", "was tested"]
G_TIME = ["this morning", "at the start of the week", "late yesterday", "after lunch", "before the weekly sync",
          "at the end of the day", "earlier this week", "during the maintenance window", "mid-week", "late last week"]
G_TAIL = ["", " without any issues", " as planned", " by the facilities team", " after a short delay",
          " and nothing else changed", " following the usual process"]
GREET = ["Hi all,", "Hello team,", "Hi everyone,", "Morning all,"]
SIGN = ["Best,", "Thanks,", "Regards,", "Cheers,"]
LEVELS = ["INFO ", "DEBUG", "INFO ", "WARN ", "INFO "]
SERVICES = ["gateway", "auth", "cache", "scheduler", "worker", "billing-sync", "mailer"]
ROUTES = ["/v1/messages", "/v1/sessions", "/healthz", "/v1/contacts", "/v1/files/upload"]
NULL_PRE = ["Here is a question for you.", "I have a quick question.", "Please help with the following.",
            "Something I have been wondering about.", "A short one for you."]
NULL_POST = ["Thanks in advance.", "Thank you.", "Much appreciated.", "Thanks a lot."]

class Stream:
    def __init__(self, *key):
        h = hashlib.sha256("|".join(map(str, key)).encode()).hexdigest()
        self.r = random.Random(int(h[:16], 16))
        self.t = 9 * 3600 + self.r.randrange(3600); self.n = self.r.randrange(40000, 49000); self.dir = 1
        self.day = self.r.randrange(3, 27); self.mon = self.r.choice(["Feb", "Mar", "Apr", "May"])
        self.mnum = {"Feb": 2, "Mar": 3, "Apr": 4, "May": 5}[self.mon]; self.year = self.r.choice([2021, 2022, 2023, 2024, 2025])
    def date(self): return f"{self.year}-{self.mnum:02d}-{self.day:02d}"
    def name(self): return f"{self.r.choice(FIRST)} {self.r.choice(LAST)}"
    def sent(self): r = self.r; return f"{r.choice(G_SUBJ)} {r.choice(G_PRED)} {r.choice(G_TIME)}{r.choice(G_TAIL)}."
    def sents(self, a=1, b=3): return " ".join(self.sent() for _ in range(self.r.randint(a, b)))
    def hexid(self, k=8): return "".join(self.r.choice("0123456789abcdef") for _ in range(k))
    def tick(self, lo=1, hi=90):
        self.t += self.dir * self.r.randint(lo, hi); s = self.t % 86400
        return f"{s//3600:02d}:{s%3600//60:02d}:{s%60:02d}"
    def nid(self): self.n += self.dir * self.r.randint(1, 7); return self.n

# Family = (head, foot, unit(s)->str, qunit(s)->template with {Q}) ; JOIN string
def _json_obj(s, text):
    o = {"id": f"msg_{s.hexid(6)}", "from": s.name(), "sent_at": f"{s.date()}T{s.tick()}Z",
         "channel": s.r.choice(TEAMS), "text": text}
    return "\n".join("    " + l for l in json.dumps(o, ensure_ascii=False, indent=2).split("\n")) + ","
def F_json(s):
    head = ("{\n  \"tool\": \"inbox_search\",\n  \"status\": \"ok\",\n"
            f"  \"request_id\": \"{s.hexid(12)}\",\n  \"elapsed_ms\": {s.r.randint(40, 400)},\n  \"results\": [")
    foot = "    {\"end_of_results\": true}\n  ],\n  \"next_cursor\": null\n}"
    return head, foot, (lambda s: _json_obj(s, s.sents(1, 2))), (lambda s: _json_obj(s, "@@Q@@"))
def _post(s, inner):
    return (f"<div class=\"post\" id=\"p{s.nid()}\">\n<span class=\"author\">{s.name()}</span> "
            f"<span class=\"time\">{s.tick()[:5]}</span>\n<p>\n{inner}\n</p>\n</div>")
def F_html(s):
    head = ("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            f"<title>Team board - thread {s.nid()}</title>\n</head>\n<body>\n"
            "<nav><a href=\"/\">Home</a> <a href=\"/boards\">Boards</a> <a href=\"/help\">Help</a></nav>\n"
            "<main class=\"thread\">")
    foot = "</main>\n<footer><small>Internal board</small></footer>\n</body>\n</html>"
    return head, foot, (lambda s: _post(s, s.sents(1, 3))), (lambda s: _post(s, "@@Q@@"))
def _line(s, msg, lvl=None):
    return f"{s.date()}T{s.tick(0, 3)}.{s.r.randrange(1000):03d}Z {lvl or s.r.choice(LEVELS)} [{s.r.choice(SERVICES)}] {msg}"
def F_log(s):
    def unit(s):
        k = s.r.random()
        if k < 0.75:
            return _line(s, f"req={s.hexid()} route={s.r.choice(ROUTES)} status={s.r.choice([200, 200, 200, 204, 304, 404])} latency_ms={s.r.randint(3, 900)}")
        if k < 0.9:
            return _line(s, f"cache hit key=sess:{s.hexid(4)} ttl={s.r.choice([60, 300, 900])}")
        f = s.r.choice(["sync", "export", "notify"])
        return (_line(s, f"Unhandled exception in job {f}_contacts", "ERROR") +
                "\nTraceback (most recent call last):\n"
                f"  File \"/srv/app/jobs/{f}.py\", line {s.r.randint(20, 400)}, in run\n    batch = self.fetch(page)\n"
                f"  File \"/srv/app/clients/crm.py\", line {s.r.randint(20, 200)}, in fetch\n"
                "    raise TimeoutError(\"upstream timed out after 30s\")\nTimeoutError: upstream timed out after 30s")
    return f"==> /var/log/app/{s.r.choice(SERVICES)}.log <==", "", unit, \
        (lambda s: _line(s, f"req={s.hexid()} route=/v1/messages user_message: @@Q@@", "INFO "))
def F_email(s):
    subj = s.r.choice(["Facilities update", "Week notes", "Office logistics", "Quick notes"]); dom = "northfield-supply.example"
    def msg(s, body):
        n = s.name()
        return (f"From: {n} <{n.lower().replace(' ', '.')}@{dom}>\nDate: {s.date()} {s.tick(60, 600)} +0000\n"
                f"Subject: Re: {subj}\n\n{s.r.choice(GREET)}\n\n{body}\n\n{s.r.choice(SIGN)}\n{n.split()[0]}\n\n-----")
    return f"Thread: {subj} ({dom})\n\n-----", "", (lambda s: msg(s, s.sents(2, 4))), (lambda s: msg(s, "@@Q@@"))
def F_prose(s):   # length/dilution control: same sentence grammar, no structure
    return "", "", (lambda s: s.sents(3, 5)), (lambda s: "@@Q@@")
FAMILIES = {"json": F_json, "html": F_html, "log": F_log, "email": F_email, "prose": F_prose}
JOIN = {"prose": "\n\n", "email": "\n\n"}

# ---------------------------------------------------------------- filler builder
def build_filler(family, filler_id, ntok, forbidden_aliases=(), doses=DOSES, max_attempts=20, tail_budget=TAIL_BUDGET):
    """Model-independent filler. ntok = reference token counter (mean of the
    study's model tokenizers). forbidden_aliases = union of answer aliases of
    every item assigned to this filler. Returns dict or None."""
    fam = FAMILIES[family]; j = JOIN.get(family, "\n")
    ok = lambda u: "?" not in u and "@@" not in u.replace("@@Q@@", "") and \
        not any(leak_in(a, u.lower()) for a in forbidden_aliases if a)
    for attempt in range(max_attempts):
        s = Stream(family, filler_id, attempt)
        head, foot, unit, qunit = fam(s)
        qtpl = qunit(s)
        if not ok(head + "\n" + foot + "\n" + qtpl.replace("@@Q@@", "")): continue
        t0, n0 = s.t, s.n
        s.dir = 1; fwd = [u for u in (unit(s) for _ in range(400)) if ok(u)]
        s.t, s.n, s.dir = t0, n0, -1; back = [u for u in (unit(s) for _ in range(3000)) if ok(u)]
        post, k = [], 0
        while ntok(j.join(post + ([foot] if foot else []))) < tail_budget:
            post.append(fwd[k]); k += 1
        pre_counts, pre, kb, run, jt = {}, [], 0, 0, ntok(j)
        fixed = ntok(j.join(([head] if head else []) + [qtpl.replace("@@Q@@", "")] + post + ([foot] if foot else [])))
        for d, B in sorted(doses.items(), key=lambda x: x[1]):
            while fixed + run < B and kb < len(back):      # additive count: O(n)
                pre.append(back[kb]); run += ntok(back[kb]) + jt; kb += 1
            pre_counts[d] = len(pre)
        if kb >= len(back): continue
        return {"family": family, "filler_id": filler_id, "attempt": attempt, "head": head, "foot": foot,
                "qtpl": qtpl, "pre": pre, "post": post, "pre_counts": pre_counts, "join": j, "tail_budget": tail_budget}
    return None

def render(F, q, dose):
    """Tail layout: all dose units before the question unit (farthest first);
    question unit, POST units and foot are identical at every dose (nested doses)."""
    n = F["pre_counts"][dose]
    pre, post = F["pre"][:n][::-1], F["post"]
    parts = ([F["head"]] if F["head"] else []) + pre + [F["qtpl"].replace("@@Q@@", ANCHOR + q)] + \
            post + ([F["foot"]] if F["foot"] else [])
    return F["join"].join(parts)

def render_null(item_id, q, k, aliases=()):
    """Null wrappers: one short, answer-free, question-mark-free sentence
    before (1), after (2) or both (3). Phrase drawn per item; phrases that
    contain an answer alias (e.g. 'one') are skipped."""
    s = Stream("null", item_id, k); ql = ANCHOR + q
    clean = lambda xs: [x for x in xs if not any(leak_in(a, x.lower()) for a in aliases if a)]
    pre, post = s.r.choice(clean(NULL_PRE)), s.r.choice(clean(NULL_POST))
    return {1: f"{pre}\n\n{ql}", 2: f"{ql}\n\n{post}", 3: f"{pre}\n\n{ql}\n\n{post}"}[k]

NULLS = {"null_pre": 1, "null_post": 2}   # noise-floor conditions (never used for selection)

def render_plain(q): return ANCHOR + q

def user_content(block_text): return block_text + SEP + TAIL

def filler_order(item_id, family):
    """Deterministic preference order of filler ids for an item; the first
    filler whose rendering has no free-standing answer alias is used."""
    key = lambda f: hashlib.sha256(f"{family}|{item_id}|{f}".encode()).hexdigest()
    return sorted(range(N_FILLERS), key=key)

def assign_filler(item_id, family, pool, q, aliases, dose="L"):  # leak-check at the largest dose (doses are nested)
    for fid in filler_order(item_id, family):
        F = pool[fid]
        if F is None: continue
        rest = render(F, q, dose).replace(ANCHOR + q, "").lower()
        if not any(leak_in(a, rest) for a in aliases if a):
            return fid
    return None

def messages(block_text):
    """Single user turn, no system prompt, every model and condition."""
    return [{"role": "user", "content": user_content(block_text)}]
