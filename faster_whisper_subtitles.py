from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path
from time import monotonic_ns
from typing import TYPE_CHECKING, Iterator, Iterable, Sequence

import av
import numpy as np
from faster_whisper.transcribe import WhisperModel, Segment as WhisperSegment
import srt

if TYPE_CHECKING:
    from numpy.typing import NDArray


SAMPLE_RATE: int = 16000
MAX_TRANSCRIBE_SECONDS: float = 30.0
WHISPER_MODEL_SIZE: str = "large-v3"


def get_audio_chunks(file: Path) -> Iterator[NDArray[np.float32]]:
    """
    Generate audio chunks from the given audio/video file.

    If multiple audio streams are present, the first one is used.
    """
    container = av.open(str(file), "r")
    audio_stream = next(s for s in container.streams if s.type == "audio")

    # resampler for 16000hz float32 mono PCM
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)

    for frame in container.decode(audio_stream):
        for resampled_frame in resampler.resample(frame):
            frame_np = resampled_frame.to_ndarray()
            assert frame_np.shape[0] == 1  # mono
            yield frame_np[0]


def transcribe_segments(
    audio_chunks: Iterable[NDArray[np.float32]],
    *,
    device: str = "auto",
    language: str | None = None,
    translate: bool = False,
) -> Iterator[WhisperSegment]:
    """
    Transcribe the given audio chunks into text segments.
    """
    model = WhisperModel(WHISPER_MODEL_SIZE, device=device)
    transcribe_kwargs = dict(
        task="transcribe" if not translate else "translate",
        language=language,
    )
    acc_buffer: NDArray[np.float32] = np.empty(0, dtype=np.float32)
    buffer_offset: int = 0
    # fill the accumulator buffer with up to 30s of audio, then transcribe
    max_transcribe_frames = int(MAX_TRANSCRIBE_SECONDS * SAMPLE_RATE)
    last_chunk: bool = False
    segments_count = 0
    start_t: int = monotonic_ns()
    while True:
        try:
            chunk = next(audio_chunks)
        except StopIteration:
            last_chunk = True
        else:
            acc_buffer = np.concatenate([acc_buffer, chunk])

        extra_frames_count = max(0, acc_buffer.size - max_transcribe_frames)
        if last_chunk or extra_frames_count > 0:
            assert extra_frames_count >= 0
            assert acc_buffer.size >= extra_frames_count
            transcribe_frames = acc_buffer[:-extra_frames_count]
            acc_buffer = acc_buffer[-extra_frames_count:]

            segments, _ = model.transcribe(transcribe_frames, **transcribe_kwargs, vad_filter=True)
            segments = list(segments)

            # while mid-transcription, last segment is not considered done, its audio is put back into the buffer
            yield_all = last_chunk or len(segments) == 1
            done_segments = segments if yield_all else segments[:-1]
            offset_seconds = buffer_offset / SAMPLE_RATE
            for segment in done_segments:
                yield segment._replace(
                    seek=segment.seek + buffer_offset,
                    start=segment.start + offset_seconds,
                    end=segment.end + offset_seconds,
                )
            segments_count += len(done_segments)

            done_end_offset: int = transcribe_frames.size
            if not segments:
                pass
            elif not yield_all:
                done_end_offset = min(
                    int(done_segments[-1].end * SAMPLE_RATE), transcribe_frames.size
                )
                acc_buffer = np.concatenate([acc_buffer, transcribe_frames[done_end_offset:]])

            buffer_offset += done_end_offset

            transcribed_seconds = buffer_offset / SAMPLE_RATE
            elapsed_time = (monotonic_ns() - start_t) / 1e9
            realtime_factor = transcribed_seconds / elapsed_time
            print(
                f"\rProcessed {transcribed_seconds:.2f} seconds, transcribed {segments_count} segments, {realtime_factor:.1f}x realtime",
                end="",
                flush=True,
            )

        if last_chunk:
            break


def create_subtitles(
    segments: Iterable[WhisperSegment],
    split_lines_length: int | None = 50,
) -> Iterable[srt.Subtitle]:
    """
    Create a subtitle file from the given segments.
    """
    index = 0
    for segment in segments:
        segment_start = segment.start
        segment_end = segment.end
        segment_duration = segment_end - segment_start
        split_n = 1 + (len(segment.text) // split_lines_length if split_lines_length else 0)
        words = segment.text.strip().split(" ")
        words_per_line = len(words) // split_n
        for j in range(split_n):
            line_words = words[
                j * words_per_line : (j + 1) * words_per_line if j < split_n - 1 else None
            ]
            line_text = " ".join(line_words)
            line_duration = (
                len(line_text) / (len(segment.text) - (split_n - 1))
            ) * segment_duration
            line_start = segment_start + line_duration * j
            line_end = segment_start + line_duration * (j + 1)
            yield srt.Subtitle(
                index=index,
                start=timedelta(seconds=line_start),
                end=timedelta(seconds=line_end),
                content=line_text,
            )
            index += 1


def postprocess_subtitles(
    subtitles: Sequence[srt.Subtitle], join_gaps_duration: float | None = 3.0
) -> list[srt.Subtitle]:
    """
    Post-process the given subtitles by joining gaps shorter than the specified duration.
    """
    if join_gaps_duration is None:
        return list(subtitles)

    new_subtitles: list[srt.Subtitle] = []
    prev_end: float | None = None
    for sub in subtitles:
        if (
            join_gaps_duration
            and prev_end is not None
            and sub.start.total_seconds() - prev_end < join_gaps_duration
        ):
            new_subtitles[-1].end = sub.start
        new_subtitles.append(sub)
        prev_end = sub.end.total_seconds()
    return new_subtitles


def transcribe_to_subtitles(
    file: Path,
    output: Path,
    device: str = "auto",
    language: str | None = None,
    translate: bool = False,
    split_lines_length: int | None = 50,
    join_gaps_duration: float | None = 3.0,
) -> None:
    audio_chunks_iter = get_audio_chunks(file)
    segments_iter = transcribe_segments(
        audio_chunks_iter, device=device, language=language, translate=translate
    )
    subtitles = create_subtitles(segments_iter, split_lines_length=split_lines_length)
    subtitles = postprocess_subtitles(list(subtitles), join_gaps_duration=join_gaps_duration)
    with output.open("w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio/video files into subtitles")
    parser.add_argument("input", type=Path, help="Input audio/video file")
    parser.add_argument(
        "output",
        type=Path,
        help="Output SRT file. Defaults to input file with .srt extension.",
        nargs="?",
    )
    parser.add_argument(
        "--language",
        help="Language code for the transcription. If unspecified, it will be auto-detected.",
    )
    parser.add_argument(
        "--translate",
        action="store_true",
        help="Translate the transcribed text to English",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use for model inference",
    )
    parser.add_argument(
        "--split-long-lines",
        nargs="?",
        type=int,
        const=50,
        help="Split long lines into multiple subtitles",
    )
    parser.add_argument(
        "--join-gaps",
        nargs="?",
        type=float,
        const=3.0,
        help="Join subtitles with gaps shorter than the specified duration",
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_suffix(".srt")

    transcribe_to_subtitles(
        args.input,
        args.output,
        device=args.device,
        language=args.language,
        translate=args.translate,
        split_lines_length=args.split_long_lines,
        join_gaps_duration=args.join_gaps,
    )


if __name__ == "__main__":
    main()
