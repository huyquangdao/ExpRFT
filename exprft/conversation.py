from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from exprft.prompts import ASSISTANT_SYSTEM_PROMPT


@dataclass
class Trajectory:
    context_id: str
    context: dict
    turns: list[tuple[str, str]]          # (teacher_utterance, assistant_response) per turn
    temperature: float
    truncated: list[bool] = field(default_factory=list)  # per turn: hit max_new_tokens?
    reward: float | None = None           # judge Overall / 10, filled by compute_rewards
    metadata: dict = field(default_factory=dict)

    @property
    def any_truncated(self) -> bool:
        return any(self.truncated)

    def final_response(self) -> str:
        return self.turns[-1][1] if self.turns else ""

    def transcript(self) -> str:
        lines = []
        for i, (teacher, assistant) in enumerate(self.turns, 1):
            lines += [f"**Turn {i}**", f"**Teacher:** {teacher}", f"**Assistant:** {assistant}"]
        return "\n\n".join(lines)

    def messages(self, system_prompt: str = ASSISTANT_SYSTEM_PROMPT) -> list[dict]:
        msgs = [{"role": "system", "content": system_prompt}]
        for teacher, assistant in self.turns:
            msgs.append({"role": "user", "content": teacher})
            msgs.append({"role": "assistant", "content": assistant})
        return msgs

    def to_dict(self) -> dict:
        return {
            "context_id": self.context_id,
            "context": self.context,
            "turns": [list(t) for t in self.turns],
            "temperature": self.temperature,
            "truncated": self.truncated,
            "reward": self.reward,
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(d: dict) -> "Trajectory":
        return Trajectory(
            context_id=d["context_id"],
            context=d["context"],
            turns=[tuple(t) for t in d["turns"]],
            temperature=float(d["temperature"]),
            truncated=list(d.get("truncated", [])),
            reward=d.get("reward"),
            metadata=dict(d.get("metadata", {})),
        )


def save_trajectories(trajectories: list[Trajectory], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([t.to_dict() for t in trajectories], indent=1))


def load_trajectories(path: str | Path) -> list[Trajectory]:
    return [Trajectory.from_dict(d) for d in json.loads(Path(path).read_text())]
