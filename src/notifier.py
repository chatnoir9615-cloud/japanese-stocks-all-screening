import os
import logging
import time

import linebot.v3.messaging as bot

# LINEの1吹き出しあたりの文字数上限（余裕を持って設定）
_CHUNK_SIZE = 4000
# Push通知1回あたりの最大吹き出し数（LINE仕様上限）
_MAX_MESSAGES_PER_PUSH = 5
# 複数回Push間のスリープ（秒）
_PUSH_INTERVAL = 1


class LineNotifier:
    def __init__(self):
        self.token = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN")
        self.user_id = os.environ.get("LINE_USER_ID")

    def send_report(self, report_text: str):
        """
        レポートテキストを LINE Push通知で送信する。

        4,000文字ごとに分割し、1回のPushで最大5吹き出しまで送信。
        5吹き出しを超える場合は切り捨てず、複数回に分けてPushする。
        これにより暴落時など多数のシグナルが発生しても全件通知される。
        """
        if not (self.token and self.user_id):
            logging.warning("LINE_CHANNEL_ACCESS_TOKEN または LINE_USER_ID が未設定です。通知をスキップします。")
            return

        # 4,000文字ごとに分割
        chunks = [report_text[i:i + _CHUNK_SIZE] for i in range(0, len(report_text), _CHUNK_SIZE)]
        total = len(chunks)

        if total > _MAX_MESSAGES_PER_PUSH:
            logging.info(
                f"レポートが長いため複数回に分けて送信します "
                f"（{len(report_text)}文字 / {total}吹き出し）"
            )

        configuration = bot.Configuration(access_token=self.token)
        with bot.ApiClient(configuration) as api:
            api_instance = bot.MessagingApi(api)

            # _MAX_MESSAGES_PER_PUSH 件ずつ送信
            for i in range(0, total, _MAX_MESSAGES_PER_PUSH):
                batch = chunks[i:i + _MAX_MESSAGES_PER_PUSH]
                messages = [bot.TextMessage(text=chunk) for chunk in batch]
                try:
                    api_instance.push_message(bot.PushMessageRequest(
                        to=self.user_id,
                        messages=messages,
                    ))
                    logging.debug(f"LINE送信: {i+1}〜{i+len(batch)}/{total}吹き出し")
                except Exception as e:
                    logging.error(f"LINE送信失敗（{i+1}〜{i+len(batch)}件目）: {e}")

                # 複数回Pushの場合はインターバルを設ける
                if i + _MAX_MESSAGES_PER_PUSH < total:
                    time.sleep(_PUSH_INTERVAL)