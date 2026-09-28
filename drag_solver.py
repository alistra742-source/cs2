# language: Python, file: drag_solver.py, target: stub (no Arkose offline path)
"""Stub DragSolver — detect always False so FunCAPTCHA path no-ops cleanly."""
from __future__ import annotations


class DragSolver:
    def __init__(self, page=None, vision=None, log=None):
        self.page = page
        self.vision = vision
        self.log = log or (lambda *a, **k: None)

    async def detect(self, timeout: float = 8.0) -> bool:
        return False

    async def solve(self, timeout: float = 45.0) -> bool:
        return False
