from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterable
from typing import Annotated, cast

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import AgentSession, Agent, JobContext, RoomInputOptions, RunContext
from livekit.agents import stt as stt_module
from livekit.agents.llm import function_tool
from livekit.plugins import openai, silero, speechmatics
from livekit.plugins.speechmatics import TurnDetectionMode
from pydantic import Field

from transcript_store import TranscriptStore

load_dotenv(".env.local")

logger = logging.getLogger("meeting-agent")

INSTRUCTIONS = """
You are a real-time meeting intelligence assistant powered by Speechmatics.
You are silently listening to a meeting between multiple participants.
Do NOT speak unless someone directly addresses you or asks you a question.
When you respond, keep it concise and spoken-friendly — no bullet points, no markdown.
The full meeting transcript is available to you at all times.
"""


class MeetingAgent(Agent):
    def __init__(self, transcript: TranscriptStore, room: rtc.Room) -> None:
        self._transcript = transcript
        self._room = room
        super().__init__(
            instructions=INSTRUCTIONS,
            stt=speechmatics.STT(
                turn_detection_mode=TurnDetectionMode.SMART_TURN,
                enable_diarization=True,
                max_speakers=8,
            ),
            llm=openai.LLM(model="gpt-4o-mini"),
            tts=speechmatics.TTS(voice="sarah"),
        )

    async def on_enter(self) -> None:
        await self.session.generate_reply(
            instructions=(
                "Briefly greet the meeting participants. "
                "Tell them you are their AI meeting assistant powered by Speechmatics. "
                "Say they can ask you for a summary, action items, or speaker stats at any time."
            )
        )

    async def stt_node(
        self,
        audio: AsyncIterable[rtc.AudioFrame],
        model_settings: agents.voice.ModelSettings,
    ) -> AsyncIterable[stt_module.SpeechEvent]:
        async def _intercept():
            async for event in Agent.default.stt_node(self, audio, model_settings):
                if (
                    event.type == stt_module.SpeechEventType.FINAL_TRANSCRIPT
                    and event.alternatives
                ):
                    sd = event.alternatives[0]
                    text = sd.text.strip()
                    if text:
                        speaker_id = sd.speaker_id or "UU"
                        self._transcript.add_utterance(speaker_id, text)
                        logger.info(f"[{speaker_id}]: {text}")
                        await self._publish("transcript", {
                            "speaker_id": speaker_id,
                            "text": text,
                            "ts": time.time(),
                        })
                        await self._publish("stats", {
                            "speakers": self._transcript.get_speaker_stats_dict(),
                        })
                yield event

        return _intercept()

    async def on_user_turn_completed(
        self,
        turn_ctx: agents.llm.ChatContext,
        new_message: agents.llm.ChatMessage,
    ) -> None:
        transcript = self._transcript.get_transcript_for_llm()
        if not transcript:
            return
        chat_ctx = self.chat_ctx.copy()
        chat_ctx.add_message(
            role="system",
            content=f"[LIVE MEETING TRANSCRIPT]\n{transcript}\n[END TRANSCRIPT]",
        )
        await self.update_chat_ctx(chat_ctx)

    # ── Function Tools ────────────────────────────────────────────────────────

    @function_tool
    async def get_meeting_summary(self, context: RunContext) -> str:
        """Called when someone asks for a summary of the meeting so far."""
        transcript = self._transcript.get_transcript_for_llm()
        stats = self._transcript.get_speaker_summary()
        if not transcript:
            return "The meeting hasn't started yet — no transcript available."
        return f"Transcript:\n{transcript}\n\nSpeaker participation:\n{stats}"

    @function_tool
    async def get_speaker_stats(self, context: RunContext) -> str:
        """Called when someone asks who has been speaking the most or wants participation stats."""
        return self._transcript.get_speaker_summary()

    @function_tool
    async def get_speaker_content(
        self,
        context: RunContext,
        speaker_label: Annotated[str, Field(description="Speaker label e.g. 'S1', 'Speaker 2', or a name")],
    ) -> str:
        """Called when someone asks what a specific speaker said."""
        utterances = self._transcript.get_utterances_by_speaker(speaker_label)
        if not utterances:
            return f"No transcript found for {speaker_label}."
        lines = [u.text for u in utterances[-20:]]
        return f"{speaker_label} said:\n" + "\n".join(lines)

    @function_tool
    async def generate_meeting_report(self, context: RunContext) -> str:
        """Called when someone asks for a full meeting report at the end of the meeting."""
        transcript = self._transcript.get_transcript_for_llm(max_chars=12000)
        stats = self._transcript.get_speaker_summary()
        return (
            f"Full transcript:\n{transcript}\n\n"
            f"Participation:\n{stats}\n\n"
            "Generate a formal meeting report with: "
            "1) Executive summary 2) Key decisions made 3) Action items with owners 4) Unresolved items"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _publish(self, event_type: str, data: dict) -> None:
        payload = json.dumps({"type": event_type, **data}).encode()
        await self._room.local_participant.publish_data(payload, reliable=True)


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    transcript = TranscriptStore()
    vad = silero.VAD.load(
        activation_threshold=0.25,
        min_silence_duration=0.3,
        min_speech_duration=0.05,
        prefix_padding_duration=0.3,
    )

    session = AgentSession(vad=vad)

    @session.on("agent_state_changed")
    def _on_state(ev):
        transcript.mark_agent_speaking(ev.new_state == "speaking")
        import asyncio
        agent = cast(MeetingAgent, session.current_agent)
        if agent:
            asyncio.ensure_future(agent._publish("agent_state", {"state": ev.new_state}))

    await session.start(
        room=ctx.room,
        agent=MeetingAgent(transcript=transcript, room=ctx.room),
        room_input_options=RoomInputOptions(),
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
