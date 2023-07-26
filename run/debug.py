# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""NLP Data Debugger for Ideas."""

from ..run.chatgpt import ChatFreeWilly2
from ..base.timer import timeit


class Debugger:
    """Debugger."""

    def __init__(self) -> None:
        """Init."""
        self.name = "name"
        # proxy_url = "http://bpusr023:bppwd023@107.172.64.163:12345"
        proxy_url = ""
        self.chatbot = ChatFreeWilly2(proxy_url=proxy_url)

    @timeit
    def run(self) -> None:
        """Run."""
        message = "Write me a poem please"
        response = self.chatbot.get_response(message=message)
        print(response)


if __name__ == "__main__":
     Debugger().run()