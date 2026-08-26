import json

import pytest

from wcq.execution.kalshi import (ApiKeyAuth, BearerAuth, FakeTransport, KalshiClient,
                                  KalshiError, OrderBook, OrderRequest, Quote, RetryPolicy,
                                  RiskLimits, TokenBucket, client_order_id,
                                  expected_value_after_fees, scan_and_quote,
                                  trading_fee_cents)


class FakeClock:
    def __init__(self):
        self.t = 0.0
    def __call__(self):
        return self.t
    def sleep(self, s):
        self.t += s


# -- order book ------------------------------------------------------------

BOOK = {"orderbook": {"yes": [[42, 100], [41, 250]], "no": [[55, 80], [54, 300]]}}


def test_yes_ask_is_the_mirror_of_the_best_no_bid():
    """The bug this test exists to prevent: reading the top of the `yes` array
    as an ask.  That would report a 42c ask when the real cost to buy YES is
    45c, i.e. fabricate 3c of edge on every single market."""
    b = OrderBook.parse("X", BOOK)
    assert b.best_yes_bid == 42
    assert b.best_no_bid == 55
    assert b.yes_ask == 45
    assert b.no_ask == 58
    assert b.spread == 3


def test_empty_book_is_not_a_crash():
    b = OrderBook.parse("X", {"orderbook": {"yes": [], "no": []}})
    assert b.yes_ask is None and b.best_yes_bid is None and b.spread is None


def test_size_at_price_level():
    assert OrderBook.parse("X", BOOK).size_at("no", 55) == 80


# -- fees ------------------------------------------------------------------

def test_fee_peaks_at_fifty_cents():
    fees = [trading_fee_cents(p, 100) for p in range(5, 100, 5)]
    assert max(fees) == trading_fee_cents(50, 100)


def test_fee_can_exceed_a_small_edge():
    """A 3c gross edge on a 45c contract is mostly eaten by the fee.  This is
    the number that kills naive prediction-market strategies."""
    gross = 0.48 * 100 - 45
    assert trading_fee_cents(45, 1) >= 1
    assert expected_value_after_fees(0.48, 45, 1) < gross


def test_fee_is_zero_for_no_contracts():
    assert trading_fee_cents(50, 0) == 0


# -- rate limiting ---------------------------------------------------------

def test_token_bucket_allows_a_burst_then_throttles():
    c = FakeClock()
    b = TokenBucket(rate_per_sec=10, capacity=10, clock=c, sleep=c.sleep)
    for _ in range(10):
        assert b.acquire() == 0.0          # burst is free
    assert b.acquire() == pytest.approx(0.1)   # then paced at the rate


def test_token_bucket_refills_over_time():
    c = FakeClock()
    b = TokenBucket(rate_per_sec=5, capacity=5, clock=c, sleep=c.sleep)
    for _ in range(5):
        b.acquire()
    c.t += 1.0
    assert b.acquire() == 0.0


def test_rate_must_be_positive():
    with pytest.raises(ValueError):
        TokenBucket(0)


# -- retries ---------------------------------------------------------------

def test_backoff_is_bounded_and_jittered():
    import random
    p = RetryPolicy(base_delay=0.5, max_delay=8.0)
    rng = random.Random(0)
    ds = [p.delay(a, rng=rng) for a in range(10)]
    assert all(0 <= d <= 8.0 for d in ds)
    assert len(set(ds)) > 1               # jitter, not a fixed schedule


def test_retry_after_header_wins_over_our_guess():
    p = RetryPolicy(base_delay=0.5, max_delay=30.0)
    assert p.delay(0, retry_after=7.0) == 7.0


def test_client_retries_a_429_then_succeeds():
    c = FakeClock()
    t = FakeTransport()
    t.queue("/markets/X/orderbook", 429, {"error": "slow down"})
    t.queue("/markets/X/orderbook", 200, BOOK)
    client = KalshiClient(transport=t, sleep=c.sleep)
    assert client.get_orderbook("X").yes_ask == 45
    assert len(t.calls) == 2
    assert c.t > 0                        # it actually waited


def test_client_does_not_retry_a_400():
    t = FakeTransport()
    t.queue("/markets/X/orderbook", 400, {"error": "bad ticker"})
    client = KalshiClient(transport=t, sleep=lambda s: None)
    with pytest.raises(KalshiError) as e:
        client.get_orderbook("X")
    assert e.value.status == 400
    assert len(t.calls) == 1              # a client error is not transient


def test_client_gives_up_after_max_attempts():
    t = FakeTransport()
    for _ in range(5):
        t.queue("/markets/X/orderbook", 503, {})
    client = KalshiClient(transport=t, sleep=lambda s: None,
                          retry=RetryPolicy(max_attempts=3))
    with pytest.raises(KalshiError):
        client.get_orderbook("X")
    assert len(t.calls) == 3


# -- idempotency -----------------------------------------------------------

def test_client_order_id_is_deterministic():
    """A retry after an ambiguous timeout must reuse the id, or the venue may
    accept the same trade twice."""
    a = client_order_id("WC-BRA", "yes", "buy", "2026-06-14")
    b = client_order_id("WC-BRA", "yes", "buy", "2026-06-14")
    assert a == b


def test_client_order_id_survives_a_price_move():
    """The recovery path after a crash re-reads the book.  If the key were
    derived from the ask, a 1c move would mint a fresh key for the same trade
    and the venue's dedup could not fire -- the position doubles.  The key is
    therefore price-free: the re-scan produces the same id at 45c or 46c."""
    t1, t2 = FakeTransport(), FakeTransport()
    t1.queue("X", 200, {})  # unused; scripted below
    book_45 = {"orderbook": {"yes": [[42, 100]], "no": [[55, 80]]}}
    book_46 = {"orderbook": {"yes": [[42, 100]], "no": [[54, 80]]}}
    ids = []
    for transport, book in ((t1, book_45), (t2, book_46)):
        transport.queue("/markets/X/orderbook", 200, book)
        client = KalshiClient(transport=transport, dry_run=True, sleep=lambda s: None)
        orders = scan_and_quote(client, [Quote("X", 0.80, "2026-06-14")])
        ids.append(orders[0].client_order_id)
    assert ids[0] == ids[1]


def test_client_order_id_separates_distinct_intents():
    """A buy and a sell of the same market on the same day are different
    trades; colliding keys would make the venue silently drop the exit."""
    buy = client_order_id("WC-BRA", "yes", "buy", "2026-06-14")
    sell = client_order_id("WC-BRA", "yes", "sell", "2026-06-14")
    assert buy != sell
    second_round = client_order_id("WC-BRA", "yes", "buy", "2026-06-14", intent="round2")
    assert second_round != buy


# -- transport failures ------------------------------------------------------

class FlakyTransport(FakeTransport):
    """Raises OSError a set number of times before delegating to the queue."""

    def __init__(self, failures: int, exc: Exception):
        super().__init__()
        self.failures = failures
        self.exc = exc
        self.attempts = 0

    def request(self, method, url, headers, body):
        self.attempts += 1
        if self.attempts <= self.failures:
            raise self.exc
        return super().request(method, url, headers, body)


def test_timeout_is_retried_with_the_identical_body():
    """The ambiguous-timeout case the idempotency key exists for: the retry
    must go through the normal retry loop and re-send byte-identical JSON --
    same client_order_id -- so the venue can deduplicate a double fill."""
    import socket
    t = FlakyTransport(1, socket.timeout("timed out"))
    t.queue("/portfolio/orders", 200, {"order": {"status": "resting"}})
    client = KalshiClient(transport=t, dry_run=False, sleep=lambda s: None)
    order = OrderRequest("X", "yes", "buy", 5, 45,
                         client_order_id("X", "yes", "buy", "2026-06-14"))
    client.place_order(order)
    posts = [c for c in t.calls if c[0] == "POST"]
    assert len(posts) == 1                     # one *arriving* call after the drop
    assert t.attempts == 2                     # but two attempts on the wire
    assert posts[0][2]["client_order_id"] == order.client_order_id


def test_transport_errors_surface_as_kalshi_error_after_retries():
    t = FlakyTransport(99, ConnectionResetError("peer reset"))
    client = KalshiClient(transport=t, sleep=lambda s: None,
                          retry=RetryPolicy(max_attempts=3))
    with pytest.raises(KalshiError) as e:
        client.get_orderbook("X")
    assert e.value.status == 0
    assert t.attempts == 3


def test_auth_headers_are_recomputed_per_attempt():
    """RSA-PSS signs a timestamp.  A signature computed once before the retry
    loop is stale after the backoff sleep and the server rejects it, so every
    attempt must re-sign."""
    stamps = []
    def sign(msg):
        stamps.append(bytes(msg))
        return b"sig"
    t = FakeTransport()
    t.queue("/markets/X/orderbook", 503, {})
    t.queue("/markets/X/orderbook", 200, BOOK)
    client = KalshiClient(auth=ApiKeyAuth("kid", sign), transport=t, sleep=lambda s: None)
    client.get_orderbook("X")
    assert len(stamps) == 2                    # one signature per attempt


# -- auth ------------------------------------------------------------------

def test_bearer_auth_header():
    assert BearerAuth("tok").headers("GET", "/x", b"")["Authorization"] == "Bearer tok"


def test_api_key_auth_signs_timestamp_method_and_path():
    seen = {}
    def sign(msg):
        seen["msg"] = msg
        return b"sig"
    h = ApiKeyAuth("kid", sign).headers("POST", "/trade-api/v2/portfolio/orders", b"{}")
    assert h["KALSHI-ACCESS-KEY"] == "kid"
    assert seen["msg"].endswith(b"POST/trade-api/v2/portfolio/orders")
    assert seen["msg"][:13].isdigit()      # timestamp is inside the signature


# -- strategy --------------------------------------------------------------

def test_dry_run_never_posts():
    t = FakeTransport()
    t.queue("/markets/X/orderbook", 200, BOOK)
    client = KalshiClient(transport=t, dry_run=True, sleep=lambda s: None)
    orders = scan_and_quote(client, [Quote("X", 0.80, "2026-06-14")])
    assert orders and client.sent_orders
    assert all(m != "POST" for m, _, _ in t.calls)


def test_no_order_when_the_edge_does_not_clear_fees():
    t = FakeTransport()
    t.queue("/markets/X/orderbook", 200, BOOK)   # yes ask 45c
    client = KalshiClient(transport=t, dry_run=True, sleep=lambda s: None)
    assert scan_and_quote(client, [Quote("X", 0.46, "2026-06-14")]) == []


def test_order_size_never_exceeds_visible_depth():
    t = FakeTransport()
    t.queue("/markets/X/orderbook", 200, BOOK)   # 80 contracts resting at 55c NO
    client = KalshiClient(transport=t, dry_run=True, sleep=lambda s: None)
    orders = scan_and_quote(client, [Quote("X", 0.95, "2026-06-14")],
                            bankroll_cents=10_000_000)
    assert orders[0].count <= 80


def test_risk_limit_caps_orders_per_run():
    t = FakeTransport()
    for i in range(10):
        t.queue(f"/markets/M{i}/orderbook", 200, BOOK)
    client = KalshiClient(transport=t, dry_run=True, sleep=lambda s: None)
    orders = scan_and_quote(client, [Quote(f"M{i}", 0.90, "2026-06-14") for i in range(10)],
                            limits=RiskLimits(max_orders_per_run=3))
    assert len(orders) == 3


def test_order_payload_uses_the_right_price_field():
    o = OrderRequest("X", "yes", "buy", 10, 45, "cid")
    assert o.payload()["yes_price"] == 45 and "no_price" not in o.payload()
    n = OrderRequest("X", "no", "buy", 10, 55, "cid")
    assert n.payload()["no_price"] == 55 and "yes_price" not in n.payload()


def test_notional_cap_spans_the_whole_run_not_each_order():
    """$500 of open notional means $500 across the run.  Enforcing it per
    order would let ten markets each take the full allowance and expose 10x
    the configured risk."""
    t = FakeTransport()
    for i in range(6):
        t.queue(f"/markets/M{i}/orderbook", 200, BOOK)   # ask 45c, depth 80
    client = KalshiClient(transport=t, dry_run=True, sleep=lambda s: None)
    limits = RiskLimits(max_open_notional_cents=5_000,   # $50
                        max_contracts_per_market=1_000, max_orders_per_run=25)
    orders = scan_and_quote(client, [Quote(f"M{i}", 0.90, "2026-06-14") for i in range(6)],
                            limits=limits, bankroll_cents=100_000_000)
    total_notional = sum(o.count * o.price_cents for o in orders)
    assert orders
    assert total_notional <= 5_000
