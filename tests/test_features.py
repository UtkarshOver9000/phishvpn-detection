import pytest

from phishurl.features import FEATURE_COLUMNS, canonical_host, domain_features, featurize, split_domain


@pytest.mark.parametrize(
    "url, host",
    [
        ("https://www.Example.com/path?q=1", "example.com"),
        ("http://www2.example.com:8080", "example.com"),
        ("example.com", "example.com"),
        ("https://user:pw@login.bank.co.uk/x", "login.bank.co.uk"),
        ("", ""),
    ],
)
def test_canonical_host(url, host):
    assert canonical_host(url) == host


def test_registrable_domain_keeps_hosting_platform_subdomains():
    assert split_domain("https://jdbdkdbsosu.blogspot.com/").registrable == "jdbdkdbsosu.blogspot.com"
    assert split_domain("https://x-y.pages.dev/a").is_private is True
    assert split_domain("https://mail.bbc.co.uk").registrable == "bbc.co.uk"


def test_ip_hosts_are_detected():
    parts = split_domain("http://198.51.100.23/bank/login")
    assert parts.is_ip and parts.registrable == "198.51.100.23"
    feats = domain_features("http://198.51.100.23/bank/login")
    assert feats["is_ip"] == 1 and feats["tld"] == "<ip>" and feats["name_len"] == 0


def test_path_scheme_and_www_do_not_change_features():
    # The model must not be able to learn URL format (see audit.py).
    a = domain_features("https://www.secure-verify-paypal.com")
    b = domain_features("http://login.secure-verify-paypal.com/signin/index.php?id=7")
    assert a == b


def test_token_and_character_features():
    feats = domain_features("https://paypal-login-verify123.com")
    assert feats["brand_token_count"] == 1
    assert feats["lure_token_count"] == 2
    assert feats["name_hyphens"] == 2
    assert feats["name_digits"] == 3
    assert feats["name_max_digit_run"] == 3
    assert feats["tld"] == "com"
    assert feats["is_punycode"] == 0
    assert domain_features("https://xn--pple-43d.com")["is_punycode"] == 1


def test_host_that_is_a_public_suffix():
    feats = domain_features("https://cloudflare-ipfs.com/ipfs/abc")
    assert feats["host_is_suffix"] == 1 and feats["is_private_suffix"] == 1


def test_featurize_column_order():
    frame = featurize(["https://a.com", "b.org"])
    assert list(frame.columns) == FEATURE_COLUMNS
    assert len(frame) == 2
