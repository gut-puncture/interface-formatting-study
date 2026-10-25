"""Fixed settings of the study. Every threshold used anywhere lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MODELS = {
    "q4": {"repo": "Qwen/Qwen3-4B-Instruct-2507", "rev": "cdbee75f17c01a7cc42f958dc650907174af0554",
           "name": "Qwen3-4B-Instruct-2507"},
    "olmo": {"repo": "allenai/Olmo-3-7B-Instruct", "rev": "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc",
             "name": "OLMo-3-7B-Instruct"},
    "q8": {"repo": "Qwen/Qwen3-8B", "rev": "b968826d9c46dd6066d109eabc6255188de91218",
           "name": "Qwen3-8B"},
    "g12": {"repo": "google/gemma-4-12B-it", "rev": "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7",
            "name": "Gemma-4-12B-it"},
}
MODEL_ORDER = ["q4", "olmo", "q8", "g12"]
TEMPLATE_KWARGS = {"enable_thinking": False}  # ignored by templates that do not use it

FAMILIES = ["json", "html", "log", "email", "prose"]
STRUCTURED = ["json", "html", "log", "email"]
DOSES = ["S", "M", "L"]
NULLS = ["null_pre", "null_post"]
WRAPPED = [f"{f}_{d}" for f in FAMILIES for d in DOSES]
CONDITIONS = ["plain"] + NULLS + WRAPPED

ANSWER_CUE = "Answer:"
MAX_NEW_TOKENS = 24      # long enough for a full-sentence reply to finish (local test, see prereg note)
KNOWN_P = 0.5            # plain P(g) threshold for a known item
KNOWN_P_SENS = 0.8       # pre-registered sensitivity threshold
MAX_G_TOKENS = 8
FAIL_GAP = 1.0986122886681098   # ln 3: plain must give gold >= 3x the wrapped probability
GATE_DROP = 0.15
GATE_MIN_KNOWN_DEV = 150
N_PATCH = 240
MAX_PAIRS_PER_ITEM = 2
BOOTSTRAP = 10_000
H1_MIN = 0.25
SEED = 20261025

PREREG_NOTES = [
    "2026-10-25, before any GPU data: a 60-item local pipeline test on Qwen3-4B showed that wrapped prompts "
    "often switch the reply to a full sentence ('The capital city of X is Y') even after the 'Answer:' cue, and "
    "the 10-token cap cut some of these replies before the answer. Changes: generation cap 24 tokens; the "
    "behavioural access measure and the gate use 'retained' = the known answer appears in the reply as a whole "
    "word (format ignored); exact-format accuracy is still recorded. Patching still uses F1/F2 failures only.",
]
