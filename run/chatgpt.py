#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ChatGPT, LLdam, etc.."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


__all__ = (
    "ChatFreeWilly2",
)


class ChatFreeWilly2:
    """Chat Model FreeWilly2."""

    # https://huggingface.co/stabilityai/FreeWilly2

    def __init__(self, proxy_url: str = "") -> None:
        """Init Sentiment Analysis."""
        raise ValueError("Model Files Too Big > 275G !")

        proxies = self.to_proxies(proxy_url=proxy_url)

        resume_download = True
        self.model_id = "stabilityai/FreeWilly2"

        # NOTE: legacy=False
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True, proxies=proxies, legacy=False, resume_download=resume_download)

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto", proxies=proxies, resume_download=resume_download)

    def to_proxies(self, proxy_url: str) -> dict[str, str]:
        """Get Proxies dict."""
        proxies: dict[str, str] = {}
        if proxy_url:
            proxies = {"http": proxy_url, "https": proxy_url}
        return proxies

    def get_response(self, message: str) -> str:
        """Check emotion from text string."""
        system_prompt = "### System:\nYou are Free Willy, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"

        prompt = f"{system_prompt}### User: {message}\n\n### Assistant:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        output = self.model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)

        return self.tokenizer.decode(output[0], skip_special_tokens=True)