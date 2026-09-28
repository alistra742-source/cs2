# language: Python, file: vision_solver.py, target: DeepSeek-backed text answers
"""Drop-in replacement for the old Ollama vision client.

Image-grid solving is not implemented here — accessibility/text path uses
DeepSeek. solve() for shape='text' routes to deepseek_bridge.
"""
from __future__ import annotations

from typing import List, Optional

from deepseek_bridge import ask_deepseek


class OllamaVisionClient:
    def __init__(self, log=None):
        self.log = log or (lambda *a, **k: None)
        self.base = "deepseek"
        self.model = "deepseek-chat"
        self.last_check_error = ""
        self.last_check_http_status = None

    async def check(self):
        # always "ready" — DeepSeek is used on demand
        return True, [self.model]

    async def solve(self, prompt: str, images: Optional[List] = None, shape: str = "text", **kwargs):
        """Return dict matching old contract for text; empty for image shapes."""
        if shape in ("text", "count", "choice"):
            ans = await ask_deepseek(prompt, log=self.log)
            if not ans:
                return {}
            if shape == "count":
                digits = "".join(c for c in ans if c.isdigit())
                return {"type": "count", "count": int(digits) if digits else 0}
            return {"type": "text", "text": ans}
        # image/grid/drag not handled — caller should use accessibility path
        self.log(f"[Vision] shape={shape} not supported without offline models; use accessibility", level="warn")
        return {}
