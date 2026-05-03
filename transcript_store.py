from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Utterance:
    speaker_id: str
    text: str
    timestamp: float


@dataclass
class SpeakerStats:
    utterances: int = 0
    words: int = 0


class TranscriptStore:
    def __init__(self) -> None:
        self._utterances: list[Utterance] = []
        self._stats: dict[str, SpeakerStats] = defaultdict(SpeakerStats)
        self._names: dict[str, str] = {}
        self._agent_speaking: bool = False

    def add_utterance(self, speaker_id: str, text: str) -> None:
        if self._agent_speaking:
            return
        self._utterances.append(Utterance(speaker_id, text, time.time()))
        stats = self._stats[speaker_id]
        stats.utterances += 1
        stats.words += len(text.split())

    def get_transcript_for_llm(self, max_chars: int = 8000) -> str:
        lines = [
            f"[{self.display_name(u.speaker_id)}]: {u.text}"
            for u in self._utterances
        ]
        text = "\n".join(lines)
        return text[-max_chars:] if len(text) > max_chars else text

    def get_speaker_summary(self) -> str:
        if not self._stats:
            return "No speakers detected yet."
        total_words = sum(s.words for s in self._stats.values()) or 1
        lines = []
        for sid, stats in sorted(self._stats.items()):
            pct = round(stats.words / total_words * 100)
            name = self.display_name(sid)
            lines.append(f"{name}: {stats.words} words ({pct}%), {stats.utterances} utterances")
        return "\n".join(lines)

    def get_speaker_stats_dict(self) -> dict:
        total_words = sum(s.words for s in self._stats.values()) or 1
        return {
            sid: {
                "words": s.words,
                "utterances": s.utterances,
                "pct": round(s.words / total_words * 100),
                "name": self.display_name(sid),
            }
            for sid, s in self._stats.items()
        }

    def get_utterances_by_speaker(self, speaker_id: str) -> list[Utterance]:
        sid = speaker_id.upper().replace("SPEAKER ", "S").replace(" ", "")
        return [u for u in self._utterances if u.speaker_id == sid]

    def set_speaker_name(self, speaker_id: str, name: str) -> None:
        self._names[speaker_id] = name

    def display_name(self, speaker_id: str) -> str:
        return self._names.get(speaker_id, speaker_id)

    def mark_agent_speaking(self, speaking: bool) -> None:
        self._agent_speaking = speaking

    @property
    def speaker_ids(self) -> list[str]:
        return list(self._stats.keys())
