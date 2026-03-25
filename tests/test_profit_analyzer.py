"""Tests for the ProfitAnalyzer class."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.agent.indicators import compute_all
from src.agent.profit_analyzer import ProfitAnalysis, ProfitAnalyzer


def _make_df(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Return an indicator-enriched OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    prices = 1.10 + np.cumsum(rng.normal(0, 0.001, n))
    spread = rng.uniform(0.0002, 0.001, n)
    df = pd.DataFrame(
        {
            "open": prices - spread / 2,
            "high": prices + spread,
            "low": prices - spread,
            "close": prices,
            "volume": rng.integers(100, 1000, n).astype(float),
        }
    )
    return compute_all(df)


def _analyzer(**kwargs) -> ProfitAnalyzer:
    return ProfitAnalyzer(**kwargs)


def _default_params(direction: str = "BUY", entry: float = 1.10):
    """Return a plausible set of position parameters."""
    stop_distance = entry * 0.02
    tp_ratio = 3.0
    if direction == "BUY":
        stop_loss = entry - stop_distance
        take_profit = entry + stop_distance * tp_ratio
    else:
        stop_loss = entry + stop_distance
        take_profit = entry - stop_distance * tp_ratio
    risk_amount = 200.0  # $200 risk
    return dict(
        pair="EUR/USD",
        direction=direction,
        entry_price=entry,
        stop_loss=stop_loss,
        take_profit=take_profit,
        risk_amount=risk_amount,
        confirmations=3,
    )


class TestProfitAnalyzerReturnType:
    def setup_method(self):
        self.analyzer = _analyzer()

    def test_returns_profit_analysis_object(self):
        params = _default_params()
        result = self.analyzer.analyse(**params)
        assert isinstance(result, ProfitAnalysis)

    def test_to_dict_has_expected_keys(self):
        params = _default_params()
        result = self.analyzer.analyse(**params)
        d = result.to_dict()
        for key in (
            "pair", "direction", "entry_price", "stop_loss", "take_profit",
            "atr", "risk_reward_ratio", "estimated_win_rate", "expected_value",
            "expected_profit", "expected_loss", "is_viable", "reasons",
        ):
            assert key in d, f"Missing key: {key}"


class TestRiskRewardRatio:
    def setup_method(self):
        self.analyzer = _analyzer(min_risk_reward=1.5)

    def test_buy_rr_correct(self):
        entry, sl, tp = 1.10, 1.08, 1.16  # profit=0.06, loss=0.02 → RR=3
        result = self.analyzer.analyse(
            pair="EUR/USD", direction="BUY", entry_price=entry,
            stop_loss=sl, take_profit=tp, risk_amount=200.0, confirmations=3,
        )
        assert abs(result.risk_reward_ratio - 3.0) < 0.01

    def test_sell_rr_correct(self):
        entry, sl, tp = 1.10, 1.12, 1.04  # profit=0.06, loss=0.02 → RR=3
        result = self.analyzer.analyse(
            pair="EUR/USD", direction="SELL", entry_price=entry,
            stop_loss=sl, take_profit=tp, risk_amount=200.0, confirmations=3,
        )
        assert abs(result.risk_reward_ratio - 3.0) < 0.01

    def test_below_min_rr_not_viable(self):
        # RR = 1.0 (below 1.5 minimum)
        entry, sl, tp = 1.10, 1.09, 1.11
        result = self.analyzer.analyse(
            pair="EUR/USD", direction="BUY", entry_price=entry,
            stop_loss=sl, take_profit=tp, risk_amount=200.0, confirmations=4,
        )
        assert result.is_viable is False

    def test_above_min_rr_viable_otherwise(self):
        # RR = 3.0, confirmations = 3 → should be viable
        params = _default_params("BUY")
        result = self.analyzer.analyse(**params)
        assert result.risk_reward_ratio >= 1.5


class TestWinRateEstimation:
    def setup_method(self):
        self.analyzer = _analyzer(
            min_confirmations=3,
            max_confirmations=4,
            base_win_rate=0.50,
            max_win_rate=0.65,
        )

    def test_min_confirmations_gives_base_win_rate(self):
        params = _default_params()
        params["confirmations"] = 3
        result = self.analyzer.analyse(**params)
        assert abs(result.estimated_win_rate - 0.50) < 0.01

    def test_max_confirmations_gives_max_win_rate(self):
        params = _default_params()
        params["confirmations"] = 4
        result = self.analyzer.analyse(**params)
        assert abs(result.estimated_win_rate - 0.65) < 0.01

    def test_win_rate_increases_with_confirmations(self):
        params3 = {**_default_params(), "confirmations": 3}
        params4 = {**_default_params(), "confirmations": 4}
        r3 = self.analyzer.analyse(**params3)
        r4 = self.analyzer.analyse(**params4)
        assert r4.estimated_win_rate >= r3.estimated_win_rate

    def test_win_rate_capped_at_max(self):
        params = {**_default_params(), "confirmations": 100}  # above max
        result = self.analyzer.analyse(**params)
        assert result.estimated_win_rate <= self.analyzer.max_win_rate


class TestExpectedValue:
    def setup_method(self):
        self.analyzer = _analyzer()

    def test_expected_value_positive_for_good_trade(self):
        # 3:1 RR + 50% win rate → clearly positive EV
        params = _default_params("BUY")
        result = self.analyzer.analyse(**params)
        assert result.expected_value > 0

    def test_expected_profit_greater_than_expected_loss_when_high_rr(self):
        params = _default_params("BUY")
        result = self.analyzer.analyse(**params)
        assert result.expected_profit > result.expected_loss

    def test_expected_loss_equals_risk_amount(self):
        params = _default_params("BUY")
        result = self.analyzer.analyse(**params)
        # expected_loss = risk_amount (by definition: units × stop_distance)
        assert abs(result.expected_loss - params["risk_amount"]) < 0.01

    def test_sell_expected_value_positive(self):
        params = _default_params("SELL")
        result = self.analyzer.analyse(**params)
        assert result.expected_value > 0


class TestATRCheck:
    def setup_method(self):
        self.analyzer = _analyzer(min_sl_atr_ratio=0.3)

    def test_no_df_skips_atr_check(self):
        params = _default_params()
        result = self.analyzer.analyse(**params, df=None)
        assert result.atr == 0.0
        # Should still be viable (ATR check skipped)
        assert result.is_viable is True

    def test_atr_from_df_is_extracted(self):
        df = _make_df()
        params = _default_params()
        result = self.analyzer.analyse(**params, df=df)
        assert result.atr > 0.0

    def test_tight_sl_fails_atr_check(self):
        df = _make_df()
        atr_val = float(df["atr"].dropna().iloc[-1])
        # Make SL much tighter than 0.3 × ATR
        entry = 1.10
        tiny_sl_distance = atr_val * 0.1  # 0.1× ATR, below 0.3 min
        result = self.analyzer.analyse(
            pair="EUR/USD",
            direction="BUY",
            entry_price=entry,
            stop_loss=entry - tiny_sl_distance,
            take_profit=entry + tiny_sl_distance * 3,
            risk_amount=200.0,
            confirmations=4,
            df=df,
        )
        assert result.is_viable is False

    def test_normal_sl_passes_atr_check(self):
        df = _make_df()
        atr_val = float(df["atr"].dropna().iloc[-1])
        entry = 1.10
        sl_distance = atr_val * 1.0  # 1× ATR, comfortably above 0.3 min
        result = self.analyzer.analyse(
            pair="EUR/USD",
            direction="BUY",
            entry_price=entry,
            stop_loss=entry - sl_distance,
            take_profit=entry + sl_distance * 3,
            risk_amount=200.0,
            confirmations=3,
            df=df,
        )
        assert result.is_viable is True


class TestViabilityFlag:
    def setup_method(self):
        self.analyzer = _analyzer()

    def test_good_trade_is_viable(self):
        params = _default_params("BUY")
        result = self.analyzer.analyse(**params)
        assert result.is_viable is True

    def test_low_rr_not_viable(self):
        entry = 1.10
        result = self.analyzer.analyse(
            pair="EUR/USD", direction="BUY",
            entry_price=entry, stop_loss=entry - 0.02, take_profit=entry + 0.01,
            risk_amount=200.0, confirmations=3,
        )
        assert result.is_viable is False

    def test_reasons_list_non_empty(self):
        params = _default_params()
        result = self.analyzer.analyse(**params)
        assert len(result.reasons) > 0

    def test_viable_reason_mentioned_when_viable(self):
        params = _default_params()
        result = self.analyzer.analyse(**params)
        if result.is_viable:
            combined = " ".join(result.reasons).lower()
            assert "viable" in combined

    def test_rejected_reason_mentioned_when_not_viable(self):
        entry = 1.10
        result = self.analyzer.analyse(
            pair="EUR/USD", direction="BUY",
            entry_price=entry, stop_loss=entry - 0.02, take_profit=entry + 0.01,
            risk_amount=200.0, confirmations=3,
        )
        assert result.is_viable is False
        combined = " ".join(result.reasons).upper()
        assert "REJECTED" in combined
