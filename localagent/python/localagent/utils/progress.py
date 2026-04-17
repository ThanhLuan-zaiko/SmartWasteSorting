from __future__ import annotations

import shutil
import sys
import time
from dataclasses import dataclass, field
from typing import TextIO


@dataclass(slots=True)
class TerminalProgressBar:
    total: int
    description: str
    enabled: bool = True
    width: int = 30
    stream: TextIO = sys.stdout
    _start_time: float = field(default_factory=time.perf_counter, init=False)
    _current: int = field(default=0, init=False)
    _last_rendered: str = field(default="", init=False)
    _interactive: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.total = max(int(self.total), 0)
        self.enabled = bool(self.enabled)
        self._interactive = bool(self.enabled and getattr(self.stream, "isatty", lambda: False)())
        if self._interactive:
            self.render()

    def advance(self, step: int = 1, *, postfix: str = "") -> None:
        self.update(self._current + step, postfix=postfix)

    def update(self, current: int, *, postfix: str = "") -> None:
        if not self.enabled:
            return
        self._current = min(max(int(current), 0), max(self.total, 1))
        self.render(postfix=postfix)

    def render(self, *, postfix: str = "") -> None:
        if not self.enabled:
            return

        elapsed = max(time.perf_counter() - self._start_time, 1e-9)
        total = max(self.total, 1)
        ratio = self._current / total
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        rate = self._current / elapsed
        message = (
            f"{self.description:<18} [{bar}] {self._current:>6}/{self.total:<6} "
            f"{ratio * 100:>6.2f}% {rate:>7.2f}/s"
        )
        if postfix:
            message = f"{message} {postfix}"

        terminal_width = shutil.get_terminal_size((120, 20)).columns
        if len(message) > terminal_width - 1:
            message = message[: terminal_width - 4] + "..."

        if self._interactive:
            print(f"\r{message}", end="", file=self.stream, flush=True)
        elif message != self._last_rendered:
            print(message, file=self.stream, flush=True)
        self._last_rendered = message

    def close(self, *, summary: str = "") -> None:
        if not self.enabled:
            return
        self.render()
        if self._interactive:
            print(file=self.stream, flush=True)
        if summary:
            print(summary, file=self.stream, flush=True)
