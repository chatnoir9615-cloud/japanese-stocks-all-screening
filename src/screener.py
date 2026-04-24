"""JPX全銘柄スクリーニング。

- JPXの銘柄リストを取得し、価格・テクニカル指標でフィルタリング
- OHLCVキャッシュは MarketDataCache（market_cache.py）に統一
- 銘柄リスト自体は symbols_cache.json に7日間キャッシュ
"""

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
# rebuild_cacheモードで一度に取得する期間（日数）
# 4回実行で1年分をカバーするため1回あたり約90日
# ただし初回（キャッシュなし）は REBUILD_PERIOD_DAYS を起点とする
_REBUILD_PERIOD_DAYS = 365  # 1年分を上限として取得

_JPX_LIST_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xls"
)


class StockScreener:
    def __init__(self, cache: MarketDataCache | None = None):
        self.cache = cache or MarketDataCache()

    # ── 銘柄リスト取得 ───────────────────────────────────────────────────

    def get_all_symbols(self) -> tuple[list[str], dict[str, str]]:
        """JPX全銘柄のシンボルリストと名称マップを返す。7日間キャッシュ。"""
        cached = self._load_symbols_cache()
        if cached:
            return cached["symbols"], cached.get("names", {})

        try:
            df = pd.read_excel(_JPX_LIST_URL, header=0)
            code_col = [c for c in df.columns if "コード" in str(c)][0]
            name_cols = [c for c in df.columns if "銘柄名" in str(c) or "名称" in str(c)]

            codes   = df[code_col].dropna().astype(str).str.strip()
            symbols = [f"{code}.T" for code in codes]

            names: dict[str, str] = {}
            if name_cols:
                for code, name in zip(codes, df[name_cols[0]].fillna("")):
                    names[f"{code}.T"] = str(name).strip()

            logging.info(f"JPX銘柄リスト取得完了: {len(symbols)}銘柄")
            self._save_symbols_cache(symbols, names)
            return symbols, names

        except Exception as e:
            logging.error(f"JPX銘柄リスト取得失敗: {e}")
            return [], {}

    # ── キャッシュ構築のみ（rebuild_cacheモード用） ──────────────────────

    def build_cache_only(self, exclude_symbols: set[str]) -> None:
        """全銘柄のOHLCVキャッシュ構築のみ実行。AI解析・LINE通知は行わない。

        rebuild_cacheモード専用。needs_update の判定に依存せず、
        各銘柄のキャッシュ最終日の翌日から現在までを強制的に差分取得する。
        キャッシュが存在しない銘柄は _REBUILD_PERIOD_DAYS 日前を起点とする。
        4回実行すれば約1年分のキャッシュが蓄積される設計。
        """
        symbols, _ = self.get_all_symbols()
        if not symbols:
            logging.error("銘柄リスト取得失敗。キャッシュ構築を中断します。")
            return

        exclude_normalized = {s if s.endswith(".T") else f"{s}.T" for s in exclude_symbols}
        targets = [s for s in symbols if s not in exclude_normalized]
        logging.info(f"キャッシュ構築対象: {len(targets)}銘柄（除外後）")

        # rebuild_cacheモードは needs_update に関わらず全銘柄を強制取得
        self._batch_fetch_rebuild(targets)
        logging.info("キャッシュ構築バッチ処理完了")

    # ── メインスクリーニング ─────────────────────────────────────────────

    def screen(
        self,
        exclude_symbols: set[str],
        max_price: int = 2000,
        top_n: int = 10,
    ) -> list[dict]:
        symbols, name_map = self.get_all_symbols()
        if not symbols:
            return []

        # 除外銘柄を .T 付きに正規化して除外
        exclude_normalized = {s if s.endswith(".T") else f"{s}.T" for s in exclude_symbols}
        targets = [s for s in symbols if s not in exclude_normalized]
        logging.info(f"スクリーニング対象: {len(targets)}銘柄（除外後）")

        # 通常スクリーニングは差分取得のみ（needs_update=Trueの銘柄のみ対象）
        self._batch_fetch(targets)

        candidates = []
        for symbol in targets:
            result = self._evaluate(symbol, name_map, max_price)
            if result:
                candidates.append(result)

        top = sorted(candidates, key=lambda x: x["score"], reverse=True)[:top_n]
        logging.info(f"スクリーニング完了: {len(candidates)}銘柄中 上位{len(top)}銘柄を選出")
        return top

    # ── バッチ価格取得（通常モード：差分のみ） ───────────────────────────

    def _batch_fetch(self, symbols: list[str]) -> None:
        """差分取得が必要な銘柄をバッチで yfinance から取得しキャッシュに保存する。
        通常スクリーニングモード用。needs_update=True の銘柄のみ対象。
        """
        stale = [s for s in symbols if self.cache.needs_update(s)]
        logging.info(f"差分取得対象: {len(stale)}銘柄 / {len(symbols)}銘柄中")

        if not stale:
            return

        # キャッシュ済み銘柄は last_date の翌日から、未取得銘柄は90日前から
        # （通常モードは直近データの補完が目的のため90日で十分）
        last_dates = [self.cache.last_date(s) for s in stale if self.cache.last_date(s)]
        if last_dates:
            start_date = min(last_dates) + timedelta(days=1)
        else:
            start_date = date.today() - timedelta(days=90)

        self._download_and_cache(stale, start_date)

    # ── バッチ価格取得（rebuild_cacheモード：強制全件） ──────────────────

    def _batch_fetch_rebuild(self, symbols: list[str]) -> None:
        """rebuild_cacheモード専用バッチ取得。

        needs_update に関わらず全銘柄を対象とし、各銘柄ごとに
        「キャッシュ最終日の翌日」を起点として差分取得する。
        キャッシュが存在しない銘柄は _REBUILD_PERIOD_DAYS 日前を起点とする。

        これにより4回のrebuild_cacheで約1年分が蓄積される：
          1回目: キャッシュなし → 365日前〜現在を取得（初回は365日分）
          2〜4回目: 前回取得済みの翌日〜現在を追加取得
        """
        today = date.today()

        # 銘柄を「キャッシュあり」「キャッシュなし」に分けて処理
        no_cache   = [s for s in symbols if self.cache.last_date(s) is None]
        has_cache  = [s for s in symbols if self.cache.last_date(s) is not None]

        # キャッシュなし銘柄: _REBUILD_PERIOD_DAYS 日前から一括取得
        if no_cache:
            start_date = today - timedelta(days=_REBUILD_PERIOD_DAYS)
            logging.info(
                f"[rebuild] キャッシュなし銘柄: {len(no_cache)}銘柄 "
                f"({start_date} 〜 {today})"
            )
            self._download_and_cache(no_cache, start_date)

        # キャッシュあり銘柄: 各銘柄の最終日翌日から差分取得
        # needs_update=False（最新済み）の銘柄はスキップ
        stale_with_cache = [
            s for s in has_cache if self.cache.needs_update(s)
        ]
        if stale_with_cache:
            last_dates = [self.cache.last_date(s) for s in stale_with_cache]
            start_date = min(last_dates) + timedelta(days=1)
            logging.info(
                f"[rebuild] 差分取得対象: {len(stale_with_cache)}銘柄 "
                f"({start_date} 〜 {today})"
            )
            self._download_and_cache(stale_with_cache, start_date)

        already_fresh = len(has_cache) - len(stale_with_cache)
        if already_fresh > 0:
            logging.info(f"[rebuild] 最新済みのためスキップ: {already_fresh}銘柄")

    # ── 共通ダウンロード処理 ─────────────────────────────────────────────

    def _download_and_cache(self, symbols: list[str], start_date: date) -> None:
        """指定銘柄リストを start_date〜今日でバッチダウンロードしキャッシュに保存する。"""
        start_str = start_date.isoformat()
        end_str   = date.today().isoformat()

        if start_str > end_str:
            logging.info("取得範囲なし（start_date が今日以降）。スキップ。")
            return

        for i in range(0, len(symbols), _BATCH_SIZE):
            batch = symbols[i : i + _BATCH_SIZE]
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
                logging.warning(f"バッチダウンロード失敗 [{i}〜{i+_BATCH_SIZE}]: {e}")
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

            logging.info(f"バッチ取得完了: {i+len(batch)}/{len(symbols)}")
            time.sleep(_BATCH_SLEEP)

    # ── 個別銘柄評価 ────────────────────────────────────────────────────

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

            # MA25上向きフィルター: 5日前より現在のMA25が高い場合のみ通過
            if len(ma25_series.dropna()) >= 6:
                ma25_5d_ago = ma25_series.dropna().iloc[-6]
                if ma25 <= ma25_5d_ago:
                    return None

            # スコアリング（出来高比・RSI・MA25乖離の複合）
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
                "category_label": "スクリーニング",
                "is_held":        False,
                "purchase_price": None,
                "pl_rate":        0,
                "metrics": {
                    "RSI":      round(float(rsi), 1),
                    "ATR":      round(float(atr), 1),
                    "MA25乖離": ma25_diff,
                    "突破":     current_price > df["High"].rolling(20).max().iloc[-2],
                    "出来高比": volume_ratio,
                },
                "fundamentals": {"PBR": 0, "利回り": "0.00%"},
            }
        except Exception as e:
            logging.debug(f"スキップ [{symbol}]: {e}")
            return None

    # ── 銘柄リストキャッシュ I/O ────────────────────────────────────────

    def _load_symbols_cache(self) -> dict | None:
        if not os.path.exists(_SYMBOLS_CACHE_PATH):
            return None
        try:
            with open(_SYMBOLS_CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            cached_at = date.fromisoformat(data["cached_at"])
            if (date.today() - cached_at).days > _SYMBOL_TTL_DAYS:
                logging.info("銘柄リストキャッシュ期限切れ。再取得します。")
                return None
            logging.info(f"銘柄リストをキャッシュから読み込み: {len(data['symbols'])}銘柄")
            return data
        except Exception as e:
            logging.warning(f"銘柄リストキャッシュ読み込み失敗: {e}")
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
            logging.info(f"銘柄リストをキャッシュに保存: {len(symbols)}銘柄")
        except Exception as e:
            logging.warning(f"銘柄リストキャッシュ保存失敗: {e}")
