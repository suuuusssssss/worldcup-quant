"""Kalshi execution layer: order book -> edge -> limit order.

Scope and honesty
-----------------
This is a complete, tested client *against a fake transport*.  It has never
been pointed at a funded account, and `DRY_RUN` is the default that has to be
switched off deliberately.  Every network call goes through one `Transport`
protocol so the entire strategy path is exercised in unit tests with a scripted
order book and zero sockets.

Kalshi has changed its authentication scheme (email/password JWT -> RSA-PSS
request signing) and its host, so auth is behind an `Authenticator` protocol
with two implementations rather than baked into the client.  Endpoint paths and
the fee schedule must be re-checked against current docs before anything here
is trusted with real money.

Why this venue at all
---------------------
There is no latency race to lose here.  The thesis is slow value betting on
retail-dominated flow in markets whose prices lag a competent model, so the
binding constraints are pricing accuracy, fees, and adverse selection -- not
microseconds.  That makes the engineering problems ordinary and unglamorous:
rate limits, idempotency, reconciliation, and not blowing up on a reconnect.

The three details that actually cost money
------------------------------------------
1. **Fees.**  Kalshi's fee is proportional to P*(1-P), which peaks at a 50c
   contract.  On a market priced near even money the fee can exceed a 2-3%
   modelled edge outright.  `expected_value_after_fees` is therefore the only
   EV function this module exposes -- there is no fee-free variant to call by
   accident.
2. **Which price you can actually hit.**  The YES ask is not in the `yes`
   array.  Resting NO bids are the liquidity a YES buyer crosses, so
   yes_ask = 100 - best_no_bid.  Reading the top of the `yes` array as an ask
   is the single most common Kalshi integration bug and it silently reports
   edge that does not exist.
3. **Idempotency.**  A POST that times out may still have filled.  Every order
   carries a deterministic client-side id derived from (ticker, side, price,
   date), so a retry after an ambiguous failure cannot double the position.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, Sequence

DEFAULT_HOST = "https://api.elections.kalshi.com"
LEGACY_HOST = "https://trading-api.kalshi.com"
API_PREFIX = "/trade-api/v2"


# ---------------------------------------------------------------------------
# Rate limiting and retries
# ---------------------------------------------------------------------------

class TokenBucket:
    """Classic token bucket.

    Chosen over a fixed window because a window lets you fire the whole quota
    in the first millisecond and then stall -- which is exactly the burst that
    trips a 429.  A bucket smooths the average while still permitting a bounded
    burst, which matches how the limit is actually enforced.
    """

    __slots__ = ("rate", "capacity", "_tokens", "_last", "_clock", "_sleep")

    def __init__(self, rate_per_sec: float, capacity: Optional[float] = None,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        if rate_per_sec <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate_per_sec
        self.capacity = capacity if capacity is not None else rate_per_sec
        self._tokens = self.capacity
        self._clock = clock
        self._sleep = sleep
        self._last = clock()

    def acquire(self, tokens: float = 1.0) -> float:
        """Block until `tokens` are available.  Returns seconds waited."""
        waited = 0.0
        while True:
            now = self._clock()
            self._tokens = min(self.capacity, self._tokens + (now - self._last) * self.rate)
            self._last = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return waited
            need = (tokens - self._tokens) / self.rate
            self._sleep(need)
            waited += need


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    base_delay: float = 0.25
    max_delay: float = 20.0
    retry_status: frozenset[int] = frozenset({429, 500, 502, 503, 504})

    def delay(self, attempt: int, retry_after: Optional[float] = None,
              rng: random.Random | None = None) -> float:
        """Exponential backoff with *full* jitter.

        Full jitter (uniform in [0, cap]) rather than a fixed exponential:
        without jitter, every client that got rate-limited by the same burst
        retries at the same instant and rebuilds the burst.  A server-supplied
        Retry-After always wins over our guess.
        """
        if retry_after is not None:
            return min(retry_after, self.max_delay)
        cap = min(self.max_delay, self.base_delay * (2 ** attempt))
        return (rng or random).uniform(0.0, cap)


# ---------------------------------------------------------------------------
# Transport / auth seams
# ---------------------------------------------------------------------------

class Authenticator(Protocol):
    def headers(self, method: str, path: str, body: bytes) -> dict[str, str]: ...


@dataclass
class BearerAuth:
    """Legacy email/password JWT flow."""
    token: str

    def headers(self, method: str, path: str, body: bytes) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"}


@dataclass
class ApiKeyAuth:
    """RSA-PSS request signing.

    Signs `timestamp_ms + METHOD + path`.  The timestamp is inside the signed
    payload specifically so a captured request cannot be replayed later; the
    server rejects a stale one.  `sign` is injected rather than importing a
    crypto library here, which keeps the package dependency-free and lets tests
    substitute a stub.
    """
    key_id: str
    sign: Callable[[bytes], bytes]
    b64: Callable[[bytes], str] = field(default=lambda b: __import__("base64").b64encode(b).decode())

    def headers(self, method: str, path: str, body: bytes) -> dict[str, str]:
        ts = str(int(time.time() * 1000))
        msg = (ts + method.upper() + path).encode()
        return {
            "KALSHI-ACCESS-KEY": self.key_id,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": self.b64(self.sign(msg)),
        }


class Transport(Protocol):
    def request(self, method: str, url: str, headers: dict[str, str],
                body: Optional[bytes]) -> tuple[int, dict[str, str], bytes]: ...


class UrllibTransport:
    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def request(self, method, url, headers, body):
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers or {}), e.read()


class FakeTransport:
    """Scripted transport for tests: queue up (status, body) per path."""

    def __init__(self):
        self.responses: dict[str, list[tuple[int, Any]]] = {}
        self.calls: list[tuple[str, str, Optional[dict]]] = []

    def queue(self, path: str, status: int, payload: Any) -> None:
        self.responses.setdefault(path, []).append((status, payload))

    def request(self, method, url, headers, body):
        path = url.split(API_PREFIX, 1)[-1].split("?", 1)[0]
        self.calls.append((method, path, json.loads(body) if body else None))
        queue = self.responses.get(path)
        if not queue:
            return 404, {}, b'{"error":"no scripted response"}'
        status, payload = queue.pop(0)
        return status, {"Content-Type": "application/json"}, json.dumps(payload).encode()


class KalshiError(RuntimeError):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:400]}")
        self.status = status
        self.body = body


# ---------------------------------------------------------------------------
# Domain objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderBook:
    """Kalshi order book, prices in integer cents.

    `yes` and `no` are both lists of resting BIDS on their respective sides.
    There is no ask array: an ask on YES is the mirror of a bid on NO.
    """
    ticker: str
    yes: list[tuple[int, int]] = field(default_factory=list)   # (price_cents, size)
    no: list[tuple[int, int]] = field(default_factory=list)

    @property
    def best_yes_bid(self) -> Optional[int]:
        return max((p for p, _ in self.yes), default=None)

    @property
    def best_no_bid(self) -> Optional[int]:
        return max((p for p, _ in self.no), default=None)

    @property
    def yes_ask(self) -> Optional[int]:
        """Cheapest price at which YES can be bought = 100 - best NO bid."""
        b = self.best_no_bid
        return None if b is None else 100 - b

    @property
    def no_ask(self) -> Optional[int]:
        b = self.best_yes_bid
        return None if b is None else 100 - b

    @property
    def spread(self) -> Optional[int]:
        a, b = self.yes_ask, self.best_yes_bid
        return None if a is None or b is None else a - b

    def size_at(self, side: str, price: int) -> int:
        book = self.yes if side == "yes" else self.no
        return sum(s for p, s in book if p == price)

    @staticmethod
    def parse(ticker: str, payload: dict) -> "OrderBook":
        ob = payload.get("orderbook", payload) or {}
        def lvls(x):
            return [(int(p), int(s)) for p, s in (x or []) if p is not None]
        return OrderBook(ticker, lvls(ob.get("yes")), lvls(ob.get("no")))


def trading_fee_cents(price_cents: int, contracts: int, rate: float = 0.07) -> int:
    """Kalshi trading fee: ceil(rate * C * P * (1-P)) in cents, P in dollars.

    Quadratic in price, so it is maximal at 50c -- precisely where a
    probabilistic model is least certain and most likely to think it has a
    small edge.  Re-check the live schedule before trusting this constant.
    """
    p = price_cents / 100.0
    return int(math.ceil(rate * contracts * p * (1.0 - p) * 100.0)) if contracts > 0 else 0


def expected_value_after_fees(model_prob: float, price_cents: int, contracts: int = 1) -> float:
    """EV in cents of buying YES at `price_cents`, net of entry fee.

    Payout is 100c on a win, 0 on a loss.  A contract at 45c with a model
    probability of 48% shows 3c of gross edge and roughly 1.7c of fee -- more
    than half the edge gone before considering slippage or being wrong.
    """
    gross = contracts * (model_prob * 100.0 - price_cents)
    return gross - trading_fee_cents(price_cents, contracts)


@dataclass(frozen=True)
class OrderRequest:
    ticker: str
    side: str          # 'yes' | 'no'
    action: str        # 'buy' | 'sell'
    count: int
    price_cents: int
    client_order_id: str

    def payload(self) -> dict:
        body = {
            "ticker": self.ticker, "action": self.action, "side": self.side,
            "type": "limit", "count": self.count,
            "client_order_id": self.client_order_id,
        }
        body["yes_price" if self.side == "yes" else "no_price"] = self.price_cents
        return body


def client_order_id(ticker: str, side: str, price_cents: int, day: str) -> str:
    """Deterministic idempotency key.

    Derived from the trade's identity, not from a random UUID: after an
    ambiguous timeout the retry must produce the *same* id, and a fresh random
    one would let the venue accept the order twice.
    """
    raw = f"{ticker}|{side}|{price_cents}|{day}"
    return "wcq-" + hashlib.sha256(raw.encode()).hexdigest()[:24]


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

@dataclass
class RiskLimits:
    max_contracts_per_market: int = 100
    max_open_notional_cents: int = 50_000     # $500
    max_orders_per_run: int = 25
    min_edge_cents: float = 2.0
    """Minimum post-fee EV per contract.  Below this the edge is inside the
    model's own error bars and the trade is noise with a fee attached."""


class KalshiClient:
    def __init__(self, auth: Optional[Authenticator] = None,
                 transport: Optional[Transport] = None,
                 host: str = DEFAULT_HOST,
                 rate: float = 8.0,
                 retry: Optional[RetryPolicy] = None,
                 dry_run: bool = True,
                 sleep: Callable[[float], None] = time.sleep,
                 rng: Optional[random.Random] = None):
        self.auth = auth
        self.transport = transport or UrllibTransport()
        self.host = host.rstrip("/")
        self.bucket = TokenBucket(rate, sleep=sleep)
        self.retry = retry or RetryPolicy()
        self.dry_run = dry_run
        self._sleep = sleep
        self._rng = rng or random.Random(0)
        self.sent_orders: list[OrderRequest] = []

    def _call(self, method: str, path: str, body: Optional[dict] = None,
              query: str = "") -> dict:
        full_path = API_PREFIX + path
        raw = json.dumps(body).encode() if body is not None else None
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.auth:
            headers.update(self.auth.headers(method, full_path, raw or b""))

        last: Optional[Exception] = None
        for attempt in range(self.retry.max_attempts):
            self.bucket.acquire()
            status, resp_headers, payload = self.transport.request(
                method, f"{self.host}{full_path}{query}", headers, raw)
            if 200 <= status < 300:
                return json.loads(payload or b"{}")
            if status in self.retry.retry_status and attempt < self.retry.max_attempts - 1:
                ra = resp_headers.get("Retry-After")
                self._sleep(self.retry.delay(attempt, float(ra) if ra else None, self._rng))
                last = KalshiError(status, payload.decode("utf-8", "replace"))
                continue
            raise KalshiError(status, payload.decode("utf-8", "replace"))
        raise last or KalshiError(0, "exhausted retries")

    # ---- reads ----------------------------------------------------------
    def get_orderbook(self, ticker: str, depth: int = 10) -> OrderBook:
        data = self._call("GET", f"/markets/{ticker}/orderbook", query=f"?depth={depth}")
        return OrderBook.parse(ticker, data)

    def list_markets(self, series_ticker: str = "", limit: int = 100) -> list[dict]:
        q = f"?limit={limit}" + (f"&series_ticker={series_ticker}" if series_ticker else "")
        return self._call("GET", "/markets", query=q).get("markets", [])

    def get_positions(self) -> list[dict]:
        return self._call("GET", "/portfolio/positions").get("market_positions", [])

    # ---- writes ---------------------------------------------------------
    def place_order(self, order: OrderRequest) -> dict:
        """Post a limit order.  A no-op that logs when `dry_run` is set."""
        self.sent_orders.append(order)
        if self.dry_run:
            return {"dry_run": True, "order": order.payload()}
        return self._call("POST", "/portfolio/orders", body=order.payload())


# ---------------------------------------------------------------------------
# Strategy loop
# ---------------------------------------------------------------------------

@dataclass
class Quote:
    ticker: str
    model_prob: float
    day: str


def scan_and_quote(client: KalshiClient, quotes: Sequence[Quote],
                   limits: Optional[RiskLimits] = None,
                   kelly_fraction: float = 0.25,
                   bankroll_cents: int = 100_000) -> list[OrderRequest]:
    """Compare each model quote to the book and post where it pays after fees.

    Buys YES only when the model's probability beats the *ask* it would have to
    cross, and posts a passive limit at the ask rather than a market order, so
    the price is known and the fill is not.  Sizing is fractional Kelly capped
    by the risk limits; an unfilled order is a strictly better outcome than an
    unbounded fill at an unknown price.
    """
    from wcq.market.kelly import kelly_fraction as kelly_f

    limits = limits or RiskLimits()
    placed: list[OrderRequest] = []

    for q in quotes:
        if len(placed) >= limits.max_orders_per_run:
            break
        book = client.get_orderbook(q.ticker)
        ask = book.yes_ask
        if ask is None or not (1 <= ask <= 99):
            continue

        ev = expected_value_after_fees(q.model_prob, ask, contracts=1)
        if ev < limits.min_edge_cents:
            continue

        decimal_odds = 100.0 / ask
        f = kelly_f(q.model_prob, decimal_odds) * kelly_fraction
        contracts = int(min(
            f * bankroll_cents / max(ask, 1),
            limits.max_contracts_per_market,
            book.size_at("no", 100 - ask),      # never size past visible depth
            limits.max_open_notional_cents / max(ask, 1),
        ))
        if contracts < 1:
            continue

        order = OrderRequest(
            ticker=q.ticker, side="yes", action="buy", count=contracts,
            price_cents=ask, client_order_id=client_order_id(q.ticker, "yes", ask, q.day),
        )
        client.place_order(order)
        placed.append(order)

    return placed
