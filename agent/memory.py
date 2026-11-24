"""Long-term memory: a small JSON user profile injected into the system prompt."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from settings import PROFILE_PATH


@dataclass
class Profile:
    data: dict = field(default_factory=dict)

    @classmethod
    def load(cls) -> "Profile":
        if PROFILE_PATH.exists():
            return cls(json.loads(PROFILE_PATH.read_text(encoding="utf-8")))
        return cls()

    def update(self, updates: dict) -> None:
        self.data.update({k: v for k, v in updates.items() if v not in (None, "")})
        PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
        PROFILE_PATH.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")

    def render(self) -> str:
        if not self.data:
            return "(empty – ask for key info when it matters)"
        return "\n".join(f"- {k}: {v}" for k, v in self.data.items())
