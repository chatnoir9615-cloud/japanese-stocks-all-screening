import os
import re
import time
import logging

from google import genai

_MAX_RETRIES = 3
_RETRY_WAIT_BASE = 10  # 秒（503時は短縮してフォールバックモデルへ切り替え）

# 503エラー時のフォールバックモデルリスト（順番に試行）
_FALLBACK_MODELS = [
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash-lite-preview-06-17",
]

_SIGNAL_ICONS = {
    "ADD_BUY":      "🟢買い乗せ",
    "STOP_LOSS":    "🔴損切り",
    "TAKE_PROFIT":  "🟡利確警告",
    "WEEKEND_EXIT": "🏳️週末手仕舞い",
}


class AIAdvisor:
    def __init__(self, api_key_env: str = "GEMINI_API_KEY"):
        api_key = os.environ.get(api_key_env)
        self.model_id = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            logging.info(f"AIAdvisor初期化完了（キー変数: {api_key_env}）")
        else:
            self.client = None
            logging.warning(f"{api_key_env} が未設定です。")

    def get_batch_advice(self, results_list: list, signals: list | None = None) -> str:
        """
        AIレポートを生成し、シグナルがある銘柄には末尾にシグナル情報を追記する。

        Args:
            results_list: 銘柄の分析結果リスト
            signals: signal_detector が返したシグナルリスト（省略可）
        """
        if not results_list:
            return ""

        # シンボル → シグナルリスト のマップを作成
        signal_map: dict[str, list] = {}
        for sig in (signals or []):
            sym = sig.get("symbol", "")
            signal_map.setdefault(sym, []).append(sig)

        entries = []
        for r in results_list:
            try:
                tag = f"【{r['category_label']}】"
                pl_info = ""
                if r.get('is_held'):
                    pl_rate = r.get('pl_rate', 0.0)
                    pl_info = f"(取得:{r['purchase_price']}円, 損益:{pl_rate}%)\n"
                    if pl_rate <= -5.0:
                        tag = "【🚨損切り警告】" + tag

                entry = (
                    f"■{r['name']}({r['symbol']}){tag}\n"
                    f"  現在値:{r['price']}円 {pl_info}"
                    f"  RSI:{r['metrics']['RSI']}, ATR:{r['metrics']['ATR']}円"
                )
                entries.append(entry)
            except KeyError as e:
                logging.warning(f"KeyError: {e}")
                continue

        if not entries:
            return ""

        prompt = f"""
以下の指示に従い、各銘柄の投資助言を出力してください。
前置き・挨拶・説明文は一切不要です。最初の銘柄から即座に出力を開始してください。

【出力フォーマット】
■ [銘柄名] ([コード])
[🔴信頼度A / 🔵信頼度B / 🟡信頼度C] [アクション]を推奨。
🎯目標価格：[価格]円 / 🛡️損切り価格：[価格]円
[テクニカル的理由。RSIやATRに触れること。100文字以内。]

【算出・判定ルール】
1. 損切り価格：保有株で含み益がある場合は「取得単価」、それ以外は「現在値の-5%」。
2. 目標価格：ATRを考慮した上昇目処。
3. 信頼度の絵文字は必ず1つだけ（🔴🔵🟡のいずれか1つ）。

【銘柄リスト】
{"\n".join(entries)}
"""
        ai_text = self._safe_generate(prompt)

        # AI解析が失敗した場合は簡易レポートにフォールバック
        if "【⚠️ Gemini APIエラー" in ai_text:
            logging.warning("AI解析失敗: 簡易レポートにフォールバック")
            ai_text = self._build_fallback_report(results_list)

        # シグナルをAIレポートの該当銘柄末尾に追記
        ai_text = self._append_signals(ai_text, results_list, signal_map)
        return ai_text

    def _build_fallback_report(self, results_list: list) -> str:
        """AI解析失敗時の簡易テキストレポートを生成する。
        シグナル・現在値・RSI・ATR・損益率は表示し、AIコメントは省略。
        """
        lines = ["⚠️ AI解析失敗のため簡易レポートを表示しています\n"]
        for r in results_list:
            try:
                name    = r.get("name", r.get("symbol", ""))
                symbol  = r.get("symbol", "")
                price   = r.get("price", "-")
                rsi     = r.get("metrics", {}).get("RSI", "-")
                atr     = r.get("metrics", {}).get("ATR", "-")
                pl_rate = r.get("pl_rate", None)

                line = f"■{name}({symbol})\n"
                line += f"  現在値: {price}円"
                if pl_rate is not None:
                    line += f" / 損益: {pl_rate}%"
                line += f"\n  RSI: {rsi} / ATR: {atr}円"
                lines.append(line)
            except Exception:
                continue
        return "\n".join(lines)

    def _append_signals(
        self,
        ai_text: str,
        results_list: list,
        signal_map: dict[str, list],
    ) -> str:
        """AIレポートの各銘柄ブロック先頭（■行の直後）にシグナル情報を挿入し、
        目標価格・損切り価格に現在値からの乖離率を付加する。
        """
        if not signal_map and not results_list:
            return ai_text

        # シンボル → 銘柄データ のマップ（乖離率計算用）
        result_map = {r["symbol"]: r for r in results_list}

        lines = ai_text.split("\n")
        output = []
        current_symbol = None
        i = 0

        while i < len(lines):
            line = lines[i]

            # 銘柄行（■ で始まる行）を検出
            if line.startswith("■"):
                current_symbol = self._extract_symbol(line, results_list)
                output.append(line)

                # シグナルがある銘柄はAIコメントの前に挿入
                if current_symbol and current_symbol in signal_map:
                    r = result_map.get(current_symbol, {})
                    purchase = r.get("purchase_price", 0)
                    quantity = r.get("quantity", 0)

                    for sig in signal_map[current_symbol]:
                        icon  = _SIGNAL_ICONS.get(sig["type"], sig["type"])
                        price = sig["price"]
                        pl_rate = sig.get("pl_rate", "")

                        if sig["type"] == "STOP_LOSS":
                            pl_yen = round((price - purchase) * quantity, 0) if purchase > 0 and quantity > 0 else ""
                            pl_yen_str = f"{int(pl_yen):+,}円({pl_rate}%)" if pl_yen != "" else f"{pl_rate}%"
                            loss_reason = sig.get("reasons", [""])[0]
                            output.append(
                                f"⚠️ 【ルール】{icon}｜現在値: {price}円 / 損益: {pl_yen_str}\n"
                                f"\t根拠：{loss_reason}"
                            )
                        elif sig["type"] in ("TAKE_PROFIT", "WEEKEND_EXIT"):
                            reasons = "、".join(sig.get("reasons", []))
                            pl_str = f"損益: {pl_rate}%" if pl_rate != "" else ""
                            output.append(
                                f"⚠️ 【ルール】{icon}｜{pl_str}\n"
                                f"\t根拠：{reasons}"
                            )
                        else:  # ADD_BUY
                            reasons = "、".join(sig.get("reasons", []))
                            output.append(
                                f"✅ 【ルール】{icon}\n"
                                f"\t根拠：{reasons}"
                            )

            # 🎯目標価格・🛡️損切り価格に乖離率を付加
            elif current_symbol and ("🎯目標価格" in line or "🛡️損切り価格" in line):
                r = result_map.get(current_symbol, {})
                current_price = r.get("price", 0)
                if current_price > 0:
                    line = self._add_rate_to_prices(line, current_price)
                output.append(line)

            # AI根拠テキスト: 信頼度行・目標価格行・空行・シグナル行以外の通常テキスト
            elif current_symbol and line.strip()                     and not line.startswith("■")                     and not line.startswith("⚠️")                     and not line.startswith("✅")                     and not line.startswith("🎯")                     and not line.startswith("🛡")                     and not re.search(r'[🔴🔵🟡]信頼度', line):
                output.append(f"\t根拠：{line.strip()}")

            else:
                output.append(line)

            i += 1

        return "\n".join(output)

    @staticmethod
    def _add_rate_to_prices(line: str, current_price: float) -> str:
        """目標価格・損切り価格の数値に現在値からの乖離率を付加する。
        例: 5990円 → 5990円(+3.28%)
        """
        import re

        def replace_price(m):
            price_val = float(m.group(1).replace(",", ""))
            rate = round((price_val - current_price) / current_price * 100, 2)
            sign = "+" if rate >= 0 else ""
            return f"{m.group(1)}円({sign}{rate}%)"

        # 「数値円」のパターンを検出して乖離率を追加（既に乖離率がついている場合はスキップ）
        return re.sub(r"([\d,]+(?:\.\d+)?)円(?!\()", replace_price, line)

    @staticmethod
    def _extract_symbol(line: str, results_list: list) -> str | None:
        """銘柄行からシンボルを抽出する。results_list と照合して特定。"""
        for r in results_list:
            name   = r.get("name", "")
            symbol = r.get("symbol", "")
            if name and name in line:
                return symbol
            if symbol and symbol.replace(".T", "") in line:
                return symbol
        return None

    def _safe_generate(self, prompt: str) -> str:
        """Gemini APIを呼び出す。503エラー時はフォールバックモデルに即切り替え。"""
        if not self.client:
            return "（AI解析スキップ：APIキー未設定）"

        # 試行するモデルリスト: メインモデル → フォールバックモデル順
        models_to_try = [self.model_id] + _FALLBACK_MODELS
        last_error_msg = ""

        for i, model in enumerate(models_to_try):
            try:
                if i > 0:
                    logging.info(f"フォールバック: {models_to_try[i-1]} → {model} に切り替え")
                res = self.client.models.generate_content(model=model, contents=prompt)
                if i > 0:
                    logging.info(f"フォールバックモデル {model} で成功")
                text = res.text.strip()
                text = re.sub(r'(?<![🔴🔵🟡])信頼度A', '📖【AI分析】🔴信頼度A', text)
                text = re.sub(r'(?<![🔴🔵🟡])信頼度B', '📖【AI分析】🔵信頼度B', text)
                text = re.sub(r'(?<![🔴🔵🟡])信頼度C', '📖【AI分析】🟡信頼度C', text)
                return text
            except Exception as e:
                err_str = str(e)
                last_error_msg = f"エラー型: {type(e).__name__}\n内容: {err_str}"
                is_503 = "503" in err_str or "Service Unavailable" in err_str or "overloaded" in err_str.lower()

                if is_503 and i < len(models_to_try) - 1:
                    # 503の場合は短い待機後にフォールバックモデルへ
                    logging.warning(f"503エラー [{model}]: フォールバックモデルに切り替えます")
                    time.sleep(_RETRY_WAIT_BASE)
                else:
                    # 503以外のエラーまたは全モデル失敗時
                    logging.warning(f"AI解析エラー [{model}]: {e}")
                    if i < len(models_to_try) - 1:
                        time.sleep(_RETRY_WAIT_BASE)

        return f"【⚠️ Gemini APIエラー（全モデル失敗）】\n{last_error_msg}"