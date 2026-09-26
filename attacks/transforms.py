"""
Six unguided baseline transformations per Draft §5.1. Each takes (url, rng)
and returns either a candidate URL string or None (meaning "not applicable
right now" -- the caller retries with a different random choice, per the
attack loop in unguided.py). None of these check validity themselves; the
caller always runs the result through attacks.validity.is_valid_url().
"""
import random
import string
from urllib.parse import urlsplit

PRINTABLE = [chr(i) for i in range(32, 127)]
COMMON_TLDS = ["com", "net", "org", "info", "biz", "io", "co", "xyz",
               "top", "site", "online", "club", "shop", "app"]


def random_substitution(url: str, rng: random.Random):
    if len(url) == 0:
        return None
    pos = rng.randrange(len(url))
    ch = rng.choice(PRINTABLE)
    return url[:pos] + ch + url[pos + 1:]


def random_insertion(url: str, rng: random.Random):
    pos = rng.randrange(len(url) + 1)
    ch = rng.choice(PRINTABLE)
    return url[:pos] + ch + url[pos:]


def random_deletion(url: str, rng: random.Random):
    if len(url) <= 12:  # keep a floor so we don't degrade toward an empty string
        return None
    pos = rng.randrange(len(url))
    return url[:pos] + url[pos + 1:]


def adjacent_swap(url: str, rng: random.Random):
    if len(url) < 2:
        return None
    pos = rng.randrange(len(url) - 1)
    chars = list(url)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


def segment_manipulation(url: str, rng: random.Random):
    """Duplicate, shuffle, or reverse one path segment."""
    full = url if "://" in url else "http://" + url
    try:
        parts = urlsplit(full)
    except Exception:
        # e.g. a stray '[' makes newer Python's urlsplit try (and fail) to
        # parse the netloc as a bracketed IPv6 literal -- treat as "not
        # applicable right now" rather than crashing the whole attack run.
        return None
    segments = [s for s in parts.path.split("/") if s]
    if not segments:
        return None
    idx = rng.randrange(len(segments))
    action = rng.choice(["duplicate", "shuffle", "reverse"])
    if action == "duplicate":
        segments.insert(idx, segments[idx])
    elif action == "shuffle" and len(segments[idx]) > 1:
        chars = list(segments[idx])
        rng.shuffle(chars)
        segments[idx] = "".join(chars)
    elif action == "reverse":
        segments[idx] = segments[idx][::-1]
    else:
        return None
    new_path = "/" + "/".join(segments)
    if parts.path:
        return url.replace(parts.path, new_path, 1)
    return url + new_path


def domain_path_tld_transform(url: str, rng: random.Random):
    """Add a random subdomain, swap the TLD, or hyphenate the registered domain label."""
    full = url if "://" in url else "http://" + url
    try:
        parts = urlsplit(full)
    except Exception:
        return None  # see note in segment_manipulation above
    host = parts.netloc
    if not host:
        return None
    action = rng.choice(["add_subdomain", "swap_tld", "hyphenate"])
    if action == "add_subdomain":
        sub = "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(3, 7)))
        new_host = f"{sub}.{host}"
    elif action == "swap_tld":
        labels = host.split(".")
        if len(labels) < 2:
            return None
        labels[-1] = rng.choice(COMMON_TLDS)
        new_host = ".".join(labels)
    elif action == "hyphenate":
        labels = host.split(".")
        if not labels[0] or len(labels[0]) < 2:
            return None
        pos = rng.randrange(1, len(labels[0]))
        labels[0] = labels[0][:pos] + "-" + labels[0][pos:]
        new_host = ".".join(labels)
    else:
        return None
    return url.replace(host, new_host, 1)


METHODS = {
    "random_substitution": random_substitution,
    "random_insertion": random_insertion,
    "random_deletion": random_deletion,
    "adjacent_swap": adjacent_swap,
    "segment_manipulation": segment_manipulation,
    "domain_path_tld_transform": domain_path_tld_transform,
}


# ---------------------------------------------------------------------------
# Targeted (position-specific) primitives, for the attribution-guided attack
# (§5.2). Unlike the six baselines above, these mutate a SPECIFIC position
# rather than a random one -- attacks/guided.py picks the position via
# Integrated Gradients, these functions just execute the edit there.
# ---------------------------------------------------------------------------

def substitution_at(url: str, pos: int, rng: random.Random):
    if pos < 0 or pos >= len(url):
        return None
    ch = rng.choice(PRINTABLE)
    return url[:pos] + ch + url[pos + 1:]


def insertion_at(url: str, pos: int, rng: random.Random):
    if pos < 0 or pos > len(url):
        return None
    ch = rng.choice(PRINTABLE)
    return url[:pos] + ch + url[pos:]


def deletion_at(url: str, pos: int, rng: random.Random = None):
    if len(url) <= 12 or pos < 0 or pos >= len(url):
        return None
    return url[:pos] + url[pos + 1:]


def swap_at(url: str, pos: int, rng: random.Random = None):
    """Swaps the character at pos with the one immediately after it."""
    if pos < 0 or pos >= len(url) - 1:
        return None
    chars = list(url)
    chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    return "".join(chars)


# Length-preserving primitives are the ones safe to use with a FIXED position
# ranking computed once (one-shot guidance, §5.3) -- inserting or deleting a
# character shifts every later index, silently invalidating the rest of a
# ranking computed before the edit. Adaptive guidance recomputes the ranking
# after every edit, so it can safely use the full set including insertion/deletion.
TARGETED_METHODS = {
    "substitution": substitution_at,
    "insertion": insertion_at,
    "deletion": deletion_at,
    "swap": swap_at,
}
LENGTH_PRESERVING_TARGETED_METHODS = ["substitution", "swap"]
