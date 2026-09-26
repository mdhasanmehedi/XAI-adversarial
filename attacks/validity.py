"""
Tier-1 (syntactic) URL-validity predicate, per Draft §5.5.

Every candidate perturbation produced by an attack -- unguided (§5.1) or
attribution-guided (§5.2) -- must pass is_valid_url() before it is accepted.
This is the primary validity check that ALL headline ASR numbers (Table 2-6)
are computed against. The Tier-2 live-resolution check is a separate,
sampled, secondary script -- not implemented here.
"""
import re
from urllib.parse import urlsplit

_PCT_RE = re.compile(r"%[0-9A-Fa-f]{2}")


def has_valid_percent_encoding(s: str) -> bool:
    """Every literal '%' must be followed by exactly two hex digits."""
    i = 0
    while i < len(s):
        if s[i] == "%":
            if not _PCT_RE.match(s[i:i + 3]):
                return False
        i += 1
    return True


def is_valid_tld_pattern(host: str) -> bool:
    """Loose syntactic TLD check: last label must be alphabetic, 2-24 chars.
    Not a real-world TLD whitelist -- deliberately loose so novel-but-plausible
    TLD mutations (an explicit §5.1 transformation class) aren't rejected."""
    if not host:
        return False
    labels = host.split(".")
    if len(labels) < 2:
        return False
    return bool(re.fullmatch(r"[A-Za-z]{2,24}", labels[-1]))


def is_valid_url(url: str, allow_tld_mutation: bool = False, max_len: int = 2048) -> bool:
    """Tier-1 validity: parses under a URL grammar, non-empty well-formed
    scheme/host, valid percent-encoding, TLD-pattern preserved unless TLD
    mutation is the transformation being tested (allow_tld_mutation=True must
    be passed explicitly by the caller -- defaults to False/preserved, per §5.5:
    "Preserves TLD-pattern validity, unless TLD mutation is itself one of the
    declared transformation classes")."""
    if not url or len(url) > max_len or len(url) < 4:
        return False
    if not has_valid_percent_encoding(url):
        return False
    try:
        parts = urlsplit(url if "://" in url else "http://" + url)
    except Exception:
        return False
    if not parts.netloc:
        return False
    host = parts.netloc.split("@")[-1].split(":")[0]
    if not host or " " in host or host.startswith(".") or host.endswith("-"):
        return False
    if not allow_tld_mutation and not is_valid_tld_pattern(host):
        return False
    # host must contain no control characters
    if any(ord(c) < 32 for c in host):
        return False
    return True


if __name__ == "__main__":
    # Quick self-test -- run directly with: python validity.py (or attacks/validity.py)
    cases = [
        ("br-icloud.com.br", False, True),
        ("mp3raid.com/music/krizz_kaliko.html", False, True),
        ("http://example.com/path?x=1", False, True),
        ("not a url with spaces .com", False, False),
        ("example.com/%zz", False, False),          # malformed percent-encoding
        ("http://", False, False),                    # empty host
        ("example.c", False, False),                  # TLD too short, mutation not allowed
        ("example.c", True, True),                    # TLD too short, but mutation IS allowed
        ("example.xyz123notreal", True, True),        # weird TLD, allowed under mutation
    ]
    n_ok = 0
    for url, allow_mut, expected in cases:
        got = is_valid_url(url, allow_tld_mutation=allow_mut)
        status = "OK" if got == expected else "MISMATCH"
        n_ok += (status == "OK")
        print(f"[{status}] is_valid_url({url!r}, allow_tld_mutation={allow_mut}) "
              f"= {got} (expected {expected})")
    print(f"\n{n_ok}/{len(cases)} self-test cases passed.")
