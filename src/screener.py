import json
import logging
import os
import time
from datetime import date, timedelta

import pandas as pd
import pandas_ta as ta
import yfinance as yf

from .market_cache import MarketDataCache

_SYMBOLS_CACHE_PATH = "cache/symbols_cache.json"
_SYMBOL_TTL_DAYS    = 7
_MIN_VOLUME         = 50_000
_BATCH_SIZE         = 50
_BATCH_SLEEP        = 2

_JPX_LIST_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)


class StockScreener:
    def __init__(self, cache: MarketDataCache | None = None):
        self.cache = cache or MarketDataCache()

    def get_all_symbols(self) -> tuple[list[str], dict[str, str]]:
        cached = self._load_symbols_cache()
        if cached:
            return cached["symbols"], cached.get("names", {})

        try:
            df = pd.read_excel(_JPX_LIST_URL, header=0)
            code_col = [c for c in df.columns if "\u30b3\u30fc\u30c9" in str(c)][0]
            name_cols = [c for c in df.columns if "\u9280\u67c4\u540d" in str(c) or "\u540d\u79f0" in str(c)]

            codes   = df[code_col].dropna().astype(str).str.strip()
            symbols = [f"{code}.T" for code in codes]

            names: dict[str, str] = {}
            if name_cols:
                for code, name in zip(codes, df[name_cols[0]].fillna("")):
                    names[f"{code}.T"] = str(name).strip()

            logging.info(f"JPX\u9298\u67c4\u30ea\u30b9\u30c8\u53d6\u5f97\u5b8c\u4e86: {len(symbols)}\u9298\u67c4")
            self._save_symbols_cache(symbols, names)
            return symbols, names

        except Exception as e:
            logging.error(f"JPX\u9298\u67c4\u30ea\u30b9\u30c8\u53d6\u5f97\u5931\u6557: {e}")
            return [], {}

    def build_cache_only(self, exclude_symbols: set[str]) -> None:
        symbols, _ = self.get_all_symbols()
        if not symbols:
            logging.error("\u9298\u67c4\u30ea\u30b9\u30c8\u53d6\u5f97\u5931\u6557\u3002\u30ad\u30e3\u30c3\u30b7\u30e5\u69cb\u7bc9\u3092\u4e2d\u65ad\u3057\u307e\u3059\u3002")
            return

        exclude_normalized = {s if s.endswith(".T") else f"{s}.T" for s in exclude_symbols}
        targets = [s for s in symbols if s not in exclude_normalized]
        logging.info(f"\u30ad\u30e3\u30c3\u30b7\u30e5\u69cb\u7bc9\u5bfe\u8c61: {len(targets)}\u9298\u67c4\uff08\u9664\u5916\u5f8c\uff09")

        self._batch_fetch(targets)
        logging.info("\u30ad\u30e3\u30c3\u30b7\u30e5\u69cb\u7bc9\u30d0\u30c3\u30c1\u51e6\u7406\u5b8c\u4e86")

    def screen(
        self,
        exclude_symbols: set[str],
        max_price: int = 2000,
        top_n: int = 10,
    ) -> list[dict]:
        symbols, name_map = self.get_all_symbols()
        if not symbols:
            return []

        exclude_normalized = {s if s.endswith(".T") else f"{s}.T" for s in exclude_symbols}
        targets = [s for s in symbols if s not in exclude_normalized]
        logging.info(f"\u30b9\u30af\u30ea\u30fc\u30cb\u30f3\u30b0\u5bfe\u8c61: {len(targets)}\u9298\u67c4\uff08\u9664\u5916\u5f8c\uff09")

        self._batch_fetch(targets)

        candidates = []
        for symbol in targets:
            result = self._evaluate(symbol, name_map, max_price)
            if result:
                candidates.append(result)

        top = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_n]
        logging.info(f"\u30b9\u30af\u30ea\u30fc\u30cb\u30f3\u30b0\u5b8c\u4e86: {len(candidates)}\u9298\u67c4\u4e2d \u4e0a\u4f4d{len(top)}\u9298\u67c4\u3092\u9078\u51fa")
        return top

    def _batch_fetch(self, symbols: list[str]) -> None:
        stale = [s for s in symbols if self.cache.needs_update(s)]
        logging.info(f"\u5dee\u5206\u53d6\u5f97\u5bfe\u8c61: {len(stale)}\u9298\u67c4 / {len(symbols)}\u9298\u67c4\u4e2d")

        if not stale:
            return

        end_str = date.today().isoformat()

        groups: dict[str, list[str]] = {}
        for symbol in stale:
            last = self.cache.last_date(symbol)
            if last:
                start_str = (last + timedelta(days=1)).isoformat()
            else:
                start_str = (date.today() - timedelta(days=365)).isoformat()
            groups.setdefault(start_str, []).append(symbol)

        logging.info(f"\u958b\u59cb\u65e5\u30b0\u30eb\u30fc\u30d7\u6570: {len(groups)}\u30d1\u30bf\u30fc\u30f3")

        total_processed = 0
        for start_str, group in sorted(groups.items()):
            if start_str > end_str:
                logging.info(f"\u30b9\u30ad\u30c3\u30d7\uff08\u958b\u59cb\u65e5\u304c\u672a\u6765\uff09: {start_str} / {len(group)}\u9298\u67c4")
                total_processed += len(group)
                continue

            for i in range(0, len(group), _BATCH_SIZE):
                batch = group[i : i + _BATCH_SIZE]
                try:
                    raw = yf.download(
                        tickers=batch,
                        start=start_str,
                        end=end_str,
                        interval="1d",
                        group_by="ticker",
                        auto_adjust=True,
                        progress=False,
                        threads=True,
                    )
                except Exception as e:
                    logging.warning(f"\u30d0\u30c3\u30c1\u30c0\u30a6\u30f3\u30ed\u30fc\u30c9\u5931\u6557 [{i}\uff5e{i+_BATCH_SIZE}]: {e}")
                    time.sleep(_BATCH_SLEEP)
                    continue

                for symbol in batch:
                    try:
                        df = raw[symbol].copy() if len(batch) > 1 else raw.copy()
                        df = df.dropna()
                        if not df.empty:
                            self.cache.update(symbol, df)
                    except Exception:
                        continue

                total_processed += len(batch)
                logging.info(f"\u30d0\u30c3\u30c1\u53d6\u5f97\u5b8c\u4e86: {total_processed}/{len(stale)}")
                time.sleep(_BATCH_SLEEP)

    def _evaluate(
        self,
        symbol: str,
        name_map: dict[str, str],
        max_price: int,
    ) -> dict | None:
        try:
            df = self.cache.get_dataframe(symbol)
            if df is None or df.empty or len(df) < 25:
                return None

            df = df.copy().sort_index()
            current_price = round(float(df["Close"].iloc[-1]), 1)
            if current_price > max_price:
                return None

            avg_volume = df["Volume"].rolling(20).mean().iloc[-1]
            if avg_volume < _MIN_VOLUME:
                return None

            rsi  = ta.rsi(df["Close"], length=14).iloc[-1]
            atr  = ta.atr(df["High"], df["Low"], df["Close"], length=14).iloc[-1]
            ma25_series  = df["Close"].rolling(25).mean()
            ma25         = ma25_series.iloc[-1]
            ma25_diff    = round(((current_price - ma25) / ma25) * 100, 2)
            volume_ratio = round(float(df["Volume"].iloc[-1]) / avg_volume, 2)

            if len(ma25_series.dropna()) >= 6:
                ma25_5d_ago = ma25_series.dropna().iloc[-6]
                if ma25 <= ma25_5d_ago:
                    return None

            score = 0.0
            score += min(volume_ratio / 2.0, 1.0) * 40
            if 25 <= rsi <= 45:
                score += (1 - abs(rsi - 35) / 10) * 30
            if -10 <= ma25_diff <= -3:
                score += (1 - abs(ma25_diff + 6.5) / 6.5) * 30

            display_name = name_map.get(symbol) or symbol.replace(".T", "")

            return {
                "symbol":         symbol.replace(".T", ""),
                "name":           display_name,
                "price":          current_price,
                "score":          round(score, 2),
                "category_label": "\u30b9\u30af\u30ea\u30fc\u30cb\u30f3\u30b0",
                "is_held":        False,
                "purchase_price": None,
                "pl_rate":        0,
                "metrics": {
                    "RSI":      round(float(rsi), 1),
                    "ATR":      round(float(atr), 1),
                    "MA25\u4e56\u9e62": ma25_diff,
                    "\u7a81\u7834":     current_price > df["High"].rolling(20).max().iloc[-2],
                    "\u51fa\u6765\u9ad8\u6bd4": volume_ratio,
                },
                "fundamentals": {"PBR": 0, "\u5229\u56de\u308a": "0.00%"},
            }
        except Exception as e:
            logging.debug(f"\u30b9\u30ad\u30c3\u30d7 [{symbol}]: {e}")
            return None

    def _load_symbols_cache(self) -> dict | None:
        if not os.path.exists(_SYMBOLS_CACHE_PATH):
            return None
        try:
            with open(_SYMBOLS_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_at = date.fromisoformat(data["cached_at"])
            if (date.today() - cached_at).days > _SYMBOL_TTL_DAYS:
                logging.info("\u9298\u67c4\u30ea\u30b9\u30c8\u30ad\u30e3\u30c3\u30b7\u30e5\u671f\u9650\u5207\u308c\u3002\u518d\u53d6\u5f97\u3057\u307e\u3059\u3002")
                return None
            logging.info(f"\u9298\u67c4\u30ea\u30b9\u30c8\u3092\u30ad\u30e3\u30c3\u30b7\u30e5\u304b\u3089\u8aad\u307f\u8fbc\u307f: {len(data['symbols'])}\u9298\u67c4")
            return data
        except Exception as e:
            logging.warning(f"\u9298\u67c4\u30ea\u30b9\u30c8\u30ad\u30e3\u30c3\u30b7\u30e5\u8aad\u307f\u8fbc\u307f\u5931\u6557: {e}")
            return None

    def _save_symbols_cache(self, symbols: list[str], names: dict[str, str]) -> None:
        os.makedirs(os.path.dirname(_SYMBOLS_CACHE_PATH), exist_ok=True)
        try:
            with open(_SYMBOLS_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(
                    {"cached_at": date.today().isoformat(), "symbols": symbols, "names": names},
                    f,
                    ensure_ascii=False,
                )
            logging.info(f"\u9298\u67c4\u30ea\u30b9\u30c8\u3092\u30ad\u30e3\u30c3\u30b7\u30e5\u306b\u4fdd\u5b58: {len(symbols)}\u9298\u67c4")
        except Exception as e:
            logging.warning(f"\u9298\u67c4\u30ea\u30b9\u30c8\u30ad\u30e3\u30c3\u30b7\u30e5\u4fdd\u5b58\u5931\u6557: {e}")
