"""
URL-validity predicates (paper §3.3).

Every candidate perturbation produced by an attack must pass is_valid_url() before it
is accepted. Two predicates are implemented:

* syntactic (default): length 4-2,048, well-formed percent-encoding, parseable URL with a
  non-empty host without spaces or control characters, and (unless the TLD transformation
  is being tested) a final host label that is alphabetic and 2-24 characters long;
* functional (set the environment variable URL_VALIDITY_MODE=functional): additionally,
  the scheme, if present, must be http or https, and the host must be a dotted-quad IPv4
  address or consist of DNS letter-digit-hyphen labels ending in a public suffix with a
  non-empty registrable label (offline Public Suffix List snapshot bundled with tldextract).
  An IPv4 host already fails the syntactic final-label check, so the IPv4 branch admits a
  candidate only when that check is skipped (the TLD transformation); in the released
  results no accepted candidate has an IPv4 host.

The main-corpus attack comparisons (results_v5/) and the GramBeddings and BiLSTM
replications (results_v6/) use the functional predicate; the other analyses (results/)
use the syntactic predicate. No DNS resolution or live check is performed.
"""
import ipaddress
import os
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
    if os.environ.get("URL_VALIDITY_MODE", "syntactic") == "functional":
        return is_functional_url(url)
    return True


# ---------------------------------------------------------------------------
# Functional validity (setting "F"): enabled by URL_VALIDITY_MODE=functional.
# In addition to the syntactic checks above, a candidate must have
#   * no scheme, or an http/https scheme (an edit can never turn "http://" into
#     an unregistered scheme such as "htt0://");
#   * a host that is a dotted-quad IPv4 address, or whose labels all follow the
#     DNS letter-digit-hyphen rule (RFC 1123) and that ends in a real public
#     suffix with a non-empty registrable label (offline Public Suffix List
#     snapshot bundled with tldextract; no network access).
# Path, query and fragment are not restricted (browsers percent-encode them).
# ---------------------------------------------------------------------------
_SCHEME_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)://")
_LDH_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_TLD_EXTRACT = None


def _extract(host):
    global _TLD_EXTRACT
    if _TLD_EXTRACT is None:
        import tldextract
        _TLD_EXTRACT = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)
    return _TLD_EXTRACT(host)


def functional_host(host: str) -> bool:
    host = (host or "").lower().rstrip(".")
    if not host:
        return False
    try:
        ipaddress.IPv4Address(host)
        return True
    except ValueError:
        pass
    labels = host.split(".")
    if len(labels) < 2 or not all(_LDH_LABEL.match(l) for l in labels):
        return False
    ext = _extract(host)
    return bool(ext.suffix) and bool(ext.domain)


def is_functional_url(url: str) -> bool:
    m = _SCHEME_RE.match(url)
    if m and m.group(1).lower() not in ("http", "https"):
        return False
    try:
        parts = urlsplit(url if "://" in url else "http://" + url)
    except Exception:
        return False
    host = parts.netloc.split("@")[-1].split(":")[0]
    return functional_host(host)


if __name__ == "__main__":
    # Quick self-test -- run directly with: python validity.py (or attacks/validity.py);
    # expectations follow URL_VALIDITY_MODE (syntactic by default, or functional)
    _FUNCTIONAL = os.environ.get("URL_VALIDITY_MODE", "syntactic") == "functional"
    print(f"URL_VALIDITY_MODE = {'functional' if _FUNCTIONAL else 'syntactic'}")
    cases = [
        ("br-icloud.com.br", False, True),
        ("mp3raid.com/music/krizz_kaliko.html", False, True),
        ("http://example.com/path?x=1", False, True),
        ("not a url with spaces .com", False, False),
        ("example.com/%zz", False, False),          # malformed percent-encoding
        ("http://", False, False),                    # empty host
        ("example.c", False, False),                  # TLD too short, mutation not allowed
        # Under the syntactic predicate a TLD-mutating edit is exempt from the TLD pattern;
        # the functional predicate still requires a real public suffix, so these two fail there.
        ("example.c", True, not _FUNCTIONAL),         # TLD too short, but mutation IS allowed
        ("example.xyz123notreal", True, not _FUNCTIONAL),  # not a public suffix
    ]
    n_ok = 0
    for url, allow_mut, expected in cases:
        got = is_valid_url(url, allow_tld_mutation=allow_mut)
        status = "OK" if got == expected else "MISMATCH"
        n_ok += (status == "OK")
        print(f"[{status}] is_valid_url({url!r}, allow_tld_mutation={allow_mut}) "
              f"= {got} (expected {expected})")
    print(f"\n{n_ok}/{len(cases)} self-test cases passed.")
