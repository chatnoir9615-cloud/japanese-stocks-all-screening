import os
import re
import time
import logging

from google import genai

_RETRY_WAIT = 10

_SIGNAL_ICONS = {
    "ADD_BUY":      "\U0001f7e2\u8cb7\u3044\u4e57\u305b",
    "STOP_LOSS":    "\U0001f534\u640d\u5207\u308a",
    "TAKE_PROFIT":  "\U0001f7e1\u5229\u78ba\u8b66\u544a",
    "WEEKEND_EXIT": "\U0001f3f3\ufe0f\u9031\u672b\u624b\u4ed5\u821e\u3044",
}


class AIAdvisor:
    def __init__(self, api_key_env: str = "GEMINI_API_KEY"):
        api_key = os.environ.get(api_key_env)
        self.model_id = "gemini-2.5-flash"
        if api_key:
            self.client = genai.Client(api_key=api_key)
            logging.info(f"AIAdvisor\u521d\u671f\u5316\u5b8c\u4e86\uff08\u30ad\u30fc\u5909\u6570: {api_key_env}\uff09")
        else:
            self.client = None
            logging.warning(f"{api_key_env} \u304c\u672a\u8a2d\u5b9a\u3067\u3059\u3002")

    def get_batch_advice(self, results_list: list, signals: list | None = None) -> str:
        if not results_list:
            return ""

        signal_map: dict[str, list] = {}
        for sig in (signals or []):
            sym = sig.get("symbol", "")
            signal_map.setdefault(sym, []).append(sig)

        entries = []
        for r in results_list:
            try:
                tag = f"\u3010{r['category_label']}\u3011"
                pl_info = ""
                if r.get('is_held'):
                    pl_rate = r.get('pl_rate', 0.0)
                    pl_info = f"(\u53d6\u5f97:{r['purchase_price']}\u5186, \u640d\u76ca:{pl_rate}%)\n"
                    if pl_rate <= -5.0:
                        tag = "\u3010\U0001f6a8\u640d\u5207\u308a\u8b66\u544a\u3011" + tag

                entry = (
                    f"\u25a0{r['name']}({r['symbol']}){tag}\n"
                    f"  \u73fe\u5728\u5024:{r['price']}\u5186 {pl_info}"
                    f"  RSI:{r['metrics']['RSI']}, ATR:{r['metrics']['ATR']}\u5186"
                )
                entries.append(entry)
            except KeyError as e:
                logging.warning(f"KeyError: {e}")
                continue

        if not entries:
            return ""

        prompt = f"""
\u4ee5\u4e0b\u306e\u6307\u793a\u306b\u5f93\u3044\u3001\u5404\u9298\u67c4\u306e\u6295\u8cc4\u52a9\u8a00\u3092\u51fa\u529b\u3057\u3066\u304f\u3060\u3055\u3044\u3002
\u524d\u7f6e\u304d\u30fb\u6328\u62f6\u30fb\u8aac\u660e\u6587\u306f\u4e00\u5207\u4e0d\u8981\u3067\u3059\u3002\u6700\u521d\u306e\u9298\u67c4\u304b\u3089\u5373\u5ea7\u306b\u51fa\u529b\u3092\u958b\u59cb\u3057\u3066\u304f\u3060\u3055\u3044\u3002

\u3010\u51fa\u529b\u30d5\u30a9\u30fc\u30de\u30c3\u30c8\u3011
\u25a0 [\u9298\u67c4\u540d] ([\u30b3\u30fc\u30c9])
[\U0001f534\u4fe1\u983c\u5ea6A / \U0001f535\u4fe1\u983c\u5ea6B / \U0001f7e1\u4fe1\u983c\u5ea6C] [\u30a2\u30af\u30b7\u30e7\u30f3]\u3092\u63a8\u5968\u3002
\U0001f3af\u76ee\u6a19\u4fa1\u683c\uff1a[\u4fa1\u683c]\u5186 / \U0001f6e1\ufe0f\u640d\u5207\u308a\u4fa1\u683c\uff1a[\u4fa1\u683c]\u5186
[\u30c6\u30af\u30cb\u30ab\u30eb\u7684\u7406\u7531\u3002RSI\u3084ATR\u306b\u89e6\u308c\u308b\u3053\u3068\u3002100\u6587\u5b57\u4ee5\u5185\u3002]

\u3010\u7b97\u51fa\u30fb\u5224\u5b9a\u30eb\u30fc\u30eb\u3011
1. \u640d\u5207\u308a\u4fa1\u683c\uff1a\u4fdd\u6709\u682a\u3067\u542b\u307f\u76ca\u304c\u3042\u308b\u5834\u5408\u306f\u300c\u53d6\u5f97\u5358\u4fa1\u300d\u3001\u305d\u308c\u4ee5\u5916\u306f\u300c\u73fe\u5728\u5024\u306e-5%\u300d\u3002
2. \u76ee\u6a19\u4fa1\u683c\uff1aATR\u3092\u8003\u616e\u3057\u305f\u4e0a\u6607\u76ee\u51e6\u3002
3. \u4fe1\u983c\u5ea6\u306e\u7d75\u6587\u5b57\u306f\u5fc5\u305a1\u3064\u3060\u3051\uff08\U0001f534\U0001f535\U0001f7e1\u306e\u3044\u305a\u308c\u304b1\u3064\uff09\u3002

\u3010\u9298\u67c4\u30ea\u30b9\u30c8\u3011
{chr(10).join(entries)}
"""
        ai_text = self._safe_generate(prompt)

        if "\u3010\u26a0\ufe0f Gemini API\u30a8\u30e9\u30fc" in ai_text:
            logging.warning("AI\u89e3\u6790\u5931\u6557: \u7c21\u6613\u30ec\u30dd\u30fc\u30c8\u306b\u30d5\u30a9\u30fc\u30eb\u30d0\u30c3\u30af")
            ai_text = self._build_fallback_report(results_list)

        ai_text = self._append_signals(ai_text, results_list, signal_map)
        return ai_text

    def _build_fallback_report(self, results_list: list) -> str:
        lines = ["\u26a0\ufe0f AI\u89e3\u6790\u5931\u6557\u306e\u305f\u3081\u7c21\u6613\u30ec\u30dd\u30fc\u30c8\u3092\u8868\u793a\u3057\u3066\u3044\u307e\u3059\n"]
        for r in results_list:
            try:
                name    = r.get("name", r.get("symbol", ""))
                symbol  = r.get("symbol", "")
                price   = r.get("price", "-")
                rsi     = r.get("metrics", {}).get("RSI", "-")
                atr     = r.get("metrics", {}).get("ATR", "-")
                pl_rate = r.get("pl_rate", None)

                line = f"\u25a0{name}({symbol})\n"
                line += f"  \u73fe\u5728\u5024: {price}\u5186"
                if pl_rate is not None:
                    line += f" / \u640d\u76ca: {pl_rate}%"
                line += f"\n  RSI: {rsi} / ATR: {atr}\u5186"
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
        if not signal_map and not results_list:
            return ai_text

        result_map = {r["symbol"]: r for r in results_list}
        lines = ai_text.split("\n")
        output = []
        current_symbol = None
        i = 0

        while i < len(lines):
            line = lines[i]

            if line.startswith("\u25a0"):
                current_symbol = self._extract_symbol(line, results_list)
                output.append(line)

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
                            pl_yen_str = f"{int(pl_yen):+,}\u5186({pl_rate}%)" if pl_yen != "" else f"{pl_rate}%"
                            loss_reason = sig.get("reasons", [""])[0]
                            output.append(
                                f"\u26a0\ufe0f \u3010\u30eb\u30fc\u30eb\u3011{icon}\uff5c\u73fe\u5728\u5024: {price}\u5186 / \u640d\u76ca: {pl_yen_str}\n"
                                f"\t\u6839\u62e0\uff1a{loss_reason}"
                            )
                        elif sig["type"] in ("TAKE_PROFIT", "WEEKEND_EXIT"):
                            reasons = "\u3001".join(sig.get("reasons", []))
                            pl_str = f"\u640d\u76ca: {pl_rate}%" if pl_rate != "" else ""
                            output.append(
                                f"\u26a0\ufe0f \u3010\u30eb\u30fc\u30eb\u3011{icon}\uff5c{pl_str}\n"
                                f"\t\u6839\u62e0\uff1a{reasons}"
                            )
                        else:
                            reasons = "\u3001".join(sig.get("reasons", []))
                            output.append(
                                f"\u2705 \u3010\u30eb\u30fc\u30eb\u3011{icon}\n"
                                f"\t\u6839\u62e0\uff1a{reasons}"
                            )

            elif current_symbol and ("\U0001f3af\u76ee\u6a19\u4fa1\u683c" in line or "\U0001f6e1\ufe0f\u640d\u5207\u308a\u4fa1\u683c" in line):
                r = result_map.get(current_symbol, {})
                current_price = r.get("price", 0)
                if current_price > 0:
                    line = self._add_rate_to_prices(line, current_price)
                output.append(line)

            elif current_symbol and line.strip() \
                    and not line.startswith("\u25a0") \
                    and not line.startswith("\u26a0\ufe0f") \
                    and not line.startswith("\u2705") \
                    and not line.startswith("\U0001f3af") \
                    and not line.startswith("\U0001f6e1") \
                    and not re.search(r'[\U0001f534\U0001f535\U0001f7e1]\u4fe1\u983c\u5ea6', line):
                output.append(f"\t\u6839\u62e0\uff1a{line.strip()}")

            else:
                output.append(line)

            i += 1

        return "\n".join(output)

    @staticmethod
    def _add_rate_to_prices(line: str, current_price: float) -> str:
        import re

        def replace_price(m):
            price_val = float(m.group(1).replace(",", ""))
            rate = round((price_val - current_price) / current_price * 100, 2)
            sign = "+" if rate >= 0 else ""
            return f"{m.group(1)}\u5186({sign}{rate}%)"

        return re.sub(r"([\d,]+(?:\.\d+)?)\u5186(?!\()", replace_price, line)

    @staticmethod
    def _extract_symbol(line: str, results_list: list) -> str | None:
        for r in results_list:
            name   = r.get("name", "")
            symbol = r.get("symbol", "")
            if name and name in line:
                return symbol
            if symbol and symbol.replace(".T", "") in line:
                return symbol
        return None

    def _safe_generate(self, prompt: str) -> str:
        if not self.client:
            return "（AI\u89e3\u6790\u30b9\u30ad\u30c3\u30d7\uff1aAPI\u30ad\u30fc\u672a\u8a2d\u5b9a\uff09"

        for attempt in range(1, 4):
            try:
                res = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt
                )
                text = res.text.strip()
                text = re.sub(r'(?<![\U0001f534\U0001f535\U0001f7e1])\u4fe1\u983c\u5ea6A', '\U0001f4d6\u3010AI\u5206\u6790\u3011\U0001f534\u4fe1\u983c\u5ea6A', text)
                text = re.sub(r'(?<![\U0001f534\U0001f535\U0001f7e1])\u4fe1\u983c\u5ea6B', '\U0001f4d6\u3010AI\u5206\u6790\u3011\U0001f535\u4fe1\u983c\u5ea6B', text)
                text = re.sub(r'(?<![\U0001f534\U0001f535\U0001f7e1])\u4fe1\u983c\u5ea6C', '\U0001f4d6\u3010AI\u5206\u6790\u3011\U0001f7e1\u4fe1\u983c\u5ea6C', text)
                return text
            except Exception as e:
                err_str = str(e)
                logging.warning(f"AI\u89e3\u6790\u30a8\u30e9\u30fc (\u8a66\u884c{attempt}/3) [{self.model_id}]: {e}")
                if attempt < 3:
                    time.sleep(_RETRY_WAIT)

        return f"\u3010\u26a0\ufe0f Gemini API\u30a8\u30e9\u30fc\uff08\u5168\u8a66\u884c\u5931\u6557\uff09\u3011\n{err_str}"
