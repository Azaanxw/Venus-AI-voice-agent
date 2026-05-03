from __future__ import annotations

import asyncio
import json
import logging
import time
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


async def _publish(room: rtc.Room, event_type: str, data: dict) -> None:
    payload = json.dumps({"type": event_type, **data}).encode()
    await room.local_participant.publish_data(payload, reliable=True)


async def _send_history(
    room: rtc.Room,
    participant: rtc.RemoteParticipant,
    transcript: TranscriptStore,
) -> None:
    await asyncio.sleep(1.0)  # let the participant's data subscription initialize
    utterances = transcript.get_all_utterances()
    if not utterances:
        return
    payload = json.dumps({
        "type": "history",
        "utterances": [
            {"speaker_id": u.speaker_id, "text": u.text, "ts": u.timestamp}
            for u in utterances
        ],
        "speakers": transcript.get_speaker_stats_dict(),
    }).encode()
    await room.local_participant.publish_data(
        payload,
        reliable=True,
        destination_identities=[participant.identity],
    )


async def _transcribe_participant(
    room: rtc.Room,
    track: rtc.RemoteAudioTrack,
    label: str,
    transcript: TranscriptStore,
) -> None:
    """Run a dedicated Speechmatics STT stream for one participant."""
    audio_stream = rtc.AudioStream(track, sample_rate=16000, num_channels=1)
    stt_instance = speechmatics.STT(turn_detection_mode=TurnDetectionMode.SMART_TURN)

    try:
        async with stt_instance.stream() as stt_stream:
            async def _push() -> None:
                try:
                    async for ev in audio_stream:
                        stt_stream.push_frame(ev.frame)
                except Exception:
                    pass

            push_task = asyncio.ensure_future(_push())
            try:
                async for event in stt_stream:
                    if (
                        event.type == stt_module.SpeechEventType.FINAL_TRANSCRIPT
                        and event.alternatives
                    ):
                        text = event.alternatives[0].text.strip()
                        if text:
                            transcript.add_utterance(label, text)
                            logger.info("[%s]: %s", label, text)
                            await _publish(room, "transcript", {
                                "speaker_id": label,
                                "text": text,
                                "ts": time.time(),
                            })
                            await _publish(room, "stats", {
                                "speakers": transcript.get_speaker_stats_dict(),
                            })
            except asyncio.CancelledError:
                pass
            finally:
                push_task.cancel()
                try:
                    await push_task
                except asyncio.CancelledError:
                    pass
    except Exception as exc:
        logger.warning("STT stream ended for %s: %s", label, exc)


class MeetingAgent(Agent):
    def __init__(self, transcript: TranscriptStore, room: rtc.Room) -> None:
        self._transcript = transcript
        self._room = room
        super().__init__(
            instructions=INSTRUCTIONS,
            stt=speechmatics.STT(turn_detection_mode=TurnDetectionMode.SMART_TURN),
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

    async def _publish(self, event_type: str, data: dict) -> None:
        await _publish(self._room, event_type, data)


async def entrypoint(ctx: JobContext) -> None:
    await ctx.connect()

    transcript = TranscriptStore()
    vad = silero.VAD.load(
        activation_threshold=0.25,
        min_silence_duration=0.3,
        min_speech_duration=0.05,
        prefix_padding_duration=0.3,
    )

    transcription_tasks: dict[str, asyncio.Task] = {}
    speaker_labels: dict[str, str] = {}
    speaker_counter = 0

    def _label_for(identity: str) -> str:
        nonlocal speaker_counter
        if identity not in speaker_labels:
            speaker_counter += 1
            speaker_labels[identity] = f"S{speaker_counter}"
        return speaker_labels[identity]

    def _start_transcription(
        participant: rtc.RemoteParticipant, track: rtc.RemoteAudioTrack
    ) -> None:
        key = f"{participant.identity}:{track.sid}"
        if key in transcription_tasks:
            return
        label = _label_for(participant.identity)
        logger.info("Starting transcription: %s → %s", participant.identity, label)
        transcription_tasks[key] = asyncio.ensure_future(
            _transcribe_participant(ctx.room, track, label, transcript)
        )

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        if isinstance(track, rtc.RemoteAudioTrack):
            _start_transcription(participant, track)

    @ctx.room.on("track_unsubscribed")
    def on_track_unsubscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ) -> None:
        key = f"{participant.identity}:{track.sid}"
        task = transcription_tasks.pop(key, None)
        if task:
            task.cancel()

    @ctx.room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant) -> None:
        asyncio.ensure_future(_send_history(ctx.room, participant, transcript))

    # Handle participants already in the room when the agent joins
    for participant in ctx.room.remote_participants.values():
        for pub in participant.track_publications.values():
            if pub.track and isinstance(pub.track, rtc.RemoteAudioTrack):
                _start_transcription(participant, pub.track)

    session = AgentSession(vad=vad)

    @session.on("agent_state_changed")
    def _on_state(ev) -> None:
        transcript.mark_agent_speaking(ev.new_state == "speaking")
        agent = cast(MeetingAgent, session.current_agent)
        if agent:
            asyncio.ensure_future(
                agent._publish("agent_state", {"state": ev.new_state})
            )

    await session.start(
        room=ctx.room,
        agent=MeetingAgent(transcript=transcript, room=ctx.room),
        room_input_options=RoomInputOptions(),
    )


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
