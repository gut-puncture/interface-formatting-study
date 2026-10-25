"""Group trials by shared prefix, compute each prefix cache once, and batch the suffixes."""
import collections
import math

import torch


def kv_bytes(cache):
    return sum(l.keys.nbytes + l.values.nbytes for l in cache.layers if getattr(l, "keys", None) is not None)


def max_rows(eng, cache, positions, cap=64, frac=0.45):
    """Rows per forward that fit in free GPU memory (cache copies + float32 logits)."""
    if eng.device != "cuda":
        return min(cap, 8)
    free = torch.cuda.mem_get_info()[0]
    per_row = kv_bytes(cache) * 1.15 + positions * len(eng.tok) * 6 + 64e6
    return int(max(1, min(cap, math.floor(frac * free / per_row))))


def by_prefix(trials):
    groups = collections.OrderedDict()
    for t in trials:
        groups.setdefault(t.prefix_key, []).append(t)
    return groups


def iter_groups(eng, trials):
    """Yield (cache, group_trials), computing one prefix cache per group and freeing it after."""
    for key, group in by_prefix(trials).items():
        cache = eng.prefix_cache(group[0].prefix_ids)
        yield cache, group
        del cache
        if eng.device == "cuda":
            torch.cuda.empty_cache()
