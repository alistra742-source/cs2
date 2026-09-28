#!/usr/bin/env python3
"""Training data collector for DragSolver — saves every successful solve."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"
IMAGES_DIR = DATA_DIR / "images"
SAMPLES_FILE = DATA_DIR / "samples.jsonl"


class TrainingDataCollector:
    """Collects training samples from successful DragSolver solves."""

    def __init__(self, output_dir: Optional[str] = None):
        self._data_dir = Path(output_dir) if output_dir else DATA_DIR
        self._images_dir = self._data_dir / "images"
        self._samples_file = self._data_dir / "samples.jsonl"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._images_dir.mkdir(parents=True, exist_ok=True)
        self._counter = self._load_counter()

    def _load_counter(self) -> int:
        existing = list(self._images_dir.glob("sample_*.png"))
        if not existing:
            return 0
        nums = []
        for p in existing:
            try:
                nums.append(int(p.stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
        return max(nums) + 1 if nums else 0

    def save(self, image_bytes: bytes, prompt: str, answer: str,
             challenge_type: str = "slider") -> Optional[str]:
        """Save one training sample.

        Args:
            image_bytes: PNG/JPEG screenshot of the challenge.
            prompt:      The prompt sent to the vision model.
            answer:      The correct answer.
            challenge_type: "slider" | "tiles" | "match"

        Returns:
            The sample filename or None on failure.
        """
        try:
            idx = self._counter
            filename = f"sample_{idx:05d}"
            img_path = self._images_dir / f"{filename}.png"

            with open(img_path, "wb") as f:
                f.write(image_bytes)

            record = {
                "id": filename,
                "timestamp": time.time(),
                "challenge_type": challenge_type,
                "prompt": prompt,
                "answer": answer,
                "image": f"images/{filename}.png",
            }
            with open(self._samples_file, "a") as f:
                f.write(json.dumps(record) + "\n")

            self._counter += 1
            return filename

        except Exception as e:
            print(f"[TrainCollector] Save failed: {e}")
            return None

    def count(self) -> int:
        return self._counter


if __name__ == "__main__":
    c = TrainingDataCollector()
    print(f"Collector ready at {c._data_dir}, existing: {c.count()}")