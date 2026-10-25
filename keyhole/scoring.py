"""Answer normalisation, correctness and failure classes (FINAL_SPEC section 4)."""
import re
import unicodedata

from keyhole.frozen import kh_wrappers as K

_STRIP = ".,;:!?\"'()[]{}*`“”«» "
_ARTICLES = re.compile(r"^(the|a|an|le|la|les|l'|un|une|der|die|das|ein|eine)\s+")
_REFUSALS = ("i can't", "i cannot", "i'm sorry", "sorry", "i don't", "i do not", "unable", "there is no",
             "it seems", "based on", "the question", "according to")


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    s = "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))
    s = s.replace("ё", "е").replace("œ", "oe").replace("ß", "ss")
    s = s.strip()
    if s.startswith("answer:"):
        s = s[len("answer:"):]
    s = s.strip(_STRIP)
    s = re.sub(r"\s+", " ", s)
    s = _ARTICLES.sub("", s)
    s = re.sub(r"(?<=\d),(?=\d)", "", s)
    return s.strip(_STRIP)


def first_line(text: str) -> str:
    return text.split("\n", 1)[0]


def is_correct(out: str, aliases) -> bool:
    o = norm(first_line(out))
    for a in aliases:
        na = norm(a)
        if na and (o == na or o.startswith(na + " ")):
            return True
    return False


def answer_present(out: str, aliases) -> bool:
    """The known answer appears anywhere in the reply as a whole word (format ignored)."""
    low = " " + norm(out) + " "
    return any(norm(a) and re.search(r"(?<![0-9a-z])" + re.escape(norm(a)) + r"(?![0-9a-z])", low) for a in aliases)


def failure_class(out: str, aliases, family: str, wrapper_text: str) -> str:
    """F0 = format / non-answer, F1 = copied from the wrapper, F2 = content failure."""
    o_raw = first_line(out).strip()
    o = norm(o_raw)
    if not o or o_raw[:1] in "{<[|#>" or o_raw.startswith("```"):
        return "F0"
    if o.startswith(_REFUSALS):
        return "F0"
    if family == "ARITH" and not re.search(r"\d", o):
        return "F0"
    low = " " + norm(out) + " "
    for a in aliases:
        na = norm(a)
        if na and re.search(r"(?<![0-9a-z])" + re.escape(na) + r"(?![0-9a-z])", low):
            return "F0"
    if wrapper_text and K.leak_in(o, wrapper_text.lower()):
        return "F1"
    return "F2"
