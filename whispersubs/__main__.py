from __future__ import annotations

import argparse
import contextlib
import logging
import os
import sys
import warnings
from datetime import timedelta
from itertools import chain
from pathlib import Path
from time import monotonic_ns
from typing import TYPE_CHECKING, Any, ClassVar, ContextManager, Iterable, Iterator, Sequence

import av
import blessed
import enlighten
import numpy as np
import srt
from faster_whisper.transcribe import Segment as WhisperSegment, TranscriptionInfo, WhisperModel
from faster_whisper.utils import available_models


if TYPE_CHECKING:
    from numpy.typing import NDArray


try:
    import nvidia.cublas.lib
    import nvidia.cudnn.lib
except ImportError:
    warnings.warn("NVIDIA CUDA libraries not found, inference will run on CPU only.", stacklevel=1)
else:
    _cublas_libs = os.path.dirname(nvidia.cublas.lib.__file__)  # noqa: PTH120
    _cudnn_libs = os.path.dirname(nvidia.cudnn.lib.__file__)  # noqa: PTH120
    _ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    if _cudnn_libs not in _ld_library_path or _cublas_libs not in _ld_library_path:
        os.environ["LD_LIBRARY_PATH"] = ":".join([
            _cublas_libs,
            _cudnn_libs,
            *_ld_library_path.split(":"),
        ])
        # restart the script with the updated environment
        os.execve(sys.executable, [sys.executable, *sys.argv], os.environ)  # noqa: S606


_logger = logging.getLogger("whispersubs")

term = blessed.Terminal()


SAMPLE_RATE: int = 16000
MAX_TRANSCRIBE_SECONDS: float = 30.0
LANGUAGE_PROB_THRESHOLD: float = 0.5

DEFAULT_WHISPER_MODEL_SIZE: str = "medium"
DEFAULT_MODEL_DEVICE: str = "auto"
DEFAULT_LANGUAGE: str | None = None
DEFAULT_SPLIT_LINES_LENGTH: int = 50
DEFAULT_JOIN_GAPS_DURATION: float | None = 1.5


def extract_audio(file: Path) -> Iterator[tuple[NDArray[np.float32], float]]:
    """
    Generate audio chunks from the given audio/video file.

    If multiple audio streams are present, the first one is used.
    """
    _logger.info(f"Opening input file '{file}'")
    container = av.open(str(file), "r")
    audio_stream = next(s for s in container.streams if s.type == "audio")

    duration: float = (audio_stream.duration or container.duration) / 1e6

    # resampler for 16000hz float32 mono PCM
    resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)

    for frame in container.decode(audio_stream):
        for resampled_frame in resampler.resample(frame):
            frame_np = resampled_frame.to_ndarray()
            assert frame_np.shape[0] == 1  # mono
            yield frame_np[0], duration


def transcribe_segments(  # noqa: PLR0914
    audio_chunks: Iterable[NDArray[np.float32]],
    *,
    model_size: str = DEFAULT_WHISPER_MODEL_SIZE,
    device: str = DEFAULT_MODEL_DEVICE,
    language: str | None = DEFAULT_LANGUAGE,
    translate: bool = False,
    total_duration: float | None = None,
    progress: bool = False,
) -> Iterator[tuple[WhisperSegment, TranscriptionInfo]]:
    """
    Transcribe the given audio chunks into text segments.
    """
    _logger.debug(f"Loading Whisper model '{model_size}' [device={device!r}]")
    model = WhisperModel(model_size, device=device)
    transcribe_kwargs = dict(
        task="transcribe" if not translate else "translate",
        language=language,
    )
    acc_buffer: NDArray[np.float32] = np.empty(0, dtype=np.float32)
    buffer_offset: int = 0
    max_transcribe_frames = int(MAX_TRANSCRIBE_SECONDS * SAMPLE_RATE)

    pbar_context: ContextManager
    if progress:

        @contextlib.contextmanager
        def pbar_cm() -> Iterator[enlighten.Counter]:
            manager = enlighten.get_manager(set_scroll=False)
            counter = manager.counter(
                total=total_duration,
                unit="s",
                desc="transcribing",
                rtf=0.0,
                bar_format=(
                    "{desc}{desc_pad}{percentage:3.0f}%|{bar}| {count:{len_total}.2f}/{total:.2f}"
                    "{unit}{unit_pad}[{elapsed}<{eta}, {rtf:.1f}x realtime]"
                ),
                counter_format=(
                    "{desc}{desc_pad}{count:.2f}{unit}{unit_pad}"
                    "[{elapsed}, {rtf:.1f}x realtime]{fill}"
                ),
            )
            with manager, counter:
                counter.update(0, force=True)
                yield counter

        pbar_context = pbar_cm()
    else:
        pbar_context = contextlib.nullcontext()

    _logger.log(logging.DEBUG if progress else logging.INFO, "Transcribing audio stream")
    pbar: enlighten.Counter | None
    with pbar_context as pbar:
        audio_chunks_iterator: Iterator[NDArray[np.float32]] = iter(audio_chunks)
        last_chunk: bool = False
        start_t: int = monotonic_ns()
        while True:
            try:
                chunk = next(audio_chunks_iterator)
            except StopIteration:
                last_chunk = True
            else:
                acc_buffer = np.concatenate([acc_buffer, chunk])

            # fill the accumulator buffer with up to 30s of audio, then transcribe
            extra_frames_count = max(0, acc_buffer.size - max_transcribe_frames)
            if last_chunk or extra_frames_count > 0:
                assert extra_frames_count >= 0
                assert acc_buffer.size >= extra_frames_count
                transcribe_frames = acc_buffer[: -extra_frames_count or None]
                acc_buffer = acc_buffer[transcribe_frames.size :]

                segments, transcribe_info = model.transcribe(
                    transcribe_frames, **transcribe_kwargs, vad_filter=True
                )
                segments = list(segments)

                # while mid-transcription, last segment is not considered done, its audio is put back into the buffer
                yield_all = last_chunk or len(segments) == 1
                done_segments = segments if yield_all else segments[:-1]
                offset_seconds = buffer_offset / SAMPLE_RATE
                for segment in done_segments:
                    offset_segment = segment._replace(
                        seek=segment.seek + buffer_offset,
                        start=segment.start + offset_seconds,
                        end=segment.end + offset_seconds,
                    )
                    yield offset_segment, transcribe_info

                done_end_offset: int = transcribe_frames.size
                if not segments:
                    pass
                elif not yield_all:
                    done_end_offset = min(
                        int(done_segments[-1].end * SAMPLE_RATE), transcribe_frames.size
                    )
                    acc_buffer = np.concatenate([acc_buffer, transcribe_frames[done_end_offset:]])

                if segments and not language:
                    if transcribe_info.language_probability > LANGUAGE_PROB_THRESHOLD:
                        language = transcribe_info.language
                        transcribe_kwargs["language"] = language
                    else:
                        _logger.warning(
                            "Failed to detect language from audio "
                            f"(best guess: {transcribe_info.language},"
                            f" prob: {transcribe_info.language_probability:.2f})"
                        )

                buffer_offset += done_end_offset

                transcribed_seconds = buffer_offset / SAMPLE_RATE
                elapsed_time = (monotonic_ns() - start_t) / 1e9
                realtime_factor = transcribed_seconds / elapsed_time

                if pbar is not None:
                    pbar.update(done_end_offset / SAMPLE_RATE, rtf=realtime_factor)

            if last_chunk:
                break


def create_subtitles(
    segments: Iterable[WhisperSegment],
    split_lines_length: int | None = DEFAULT_SPLIT_LINES_LENGTH,
) -> Iterable[srt.Subtitle]:
    """
    Create a subtitle file from the given segments.
    """
    segments_count: int = 0
    index: int = 0
    for segments_count, segment in enumerate(segments):  # noqa: B007
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

    _logger.debug(f"Generated {index} subtitles from {segments_count + 1} transcribed segments")


def postprocess_subtitles(
    subtitles: Sequence[srt.Subtitle], join_gaps_duration: float | None = DEFAULT_JOIN_GAPS_DURATION
) -> list[srt.Subtitle]:
    """
    Post-process the given subtitles by joining gaps shorter than the specified duration.
    """
    _logger.debug("Post-processing generated subtitles")

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
    input_file: Path,
    output_file: Path | None = None,
    *,
    model_size: str = DEFAULT_WHISPER_MODEL_SIZE,
    device: str = DEFAULT_MODEL_DEVICE,
    language: str | None = DEFAULT_LANGUAGE,
    translate: bool = False,
    progress: bool = False,
    split_lines_length: int | None = DEFAULT_SPLIT_LINES_LENGTH,
    join_gaps_duration: float | None = DEFAULT_JOIN_GAPS_DURATION,
) -> None:
    extract_audio_iter = extract_audio(input_file)

    audio_chunk, duration = next(extract_audio_iter)
    audio_chunks_iter = chain([audio_chunk], (c for c, _ in extract_audio_iter))

    transcribe_iter = transcribe_segments(
        audio_chunks_iter,
        model_size=model_size,
        device=device,
        language=language,
        translate=translate,
        total_duration=duration,
        progress=progress,
    )

    def iterate_segments() -> Iterator[WhisperSegment]:
        nonlocal language
        for segment, transcription_info in transcribe_iter:
            if not language and transcription_info.language_probability > LANGUAGE_PROB_THRESHOLD:
                language = transcription_info.language
            yield segment

    subtitles = create_subtitles(iterate_segments(), split_lines_length=split_lines_length)
    subtitles = postprocess_subtitles(list(subtitles), join_gaps_duration=join_gaps_duration)

    if not output_file:
        if translate:
            language = "en"
        elif not language:
            _logger.warning("Failed to detect language from audio")
        suffix = (f".{language}" if language else "") + ".srt"
        output_file = input_file.with_suffix(suffix)

    _logger.info(f"Writing subtitles to '{output_file}'")
    with output_file.open("w", encoding="utf-8") as f:
        f.write(srt.compose(subtitles))


class LogFormatter(logging.Formatter):
    """
    Custom formatter that adds colors using blessed and uses relative timestamps.
    """

    COLOR_MAP: ClassVar[dict[int, str]] = {
        logging.DEBUG: "blue",
        logging.INFO: "green",
        logging.WARNING: "yellow",
        logging.ERROR: "red",
        logging.CRITICAL: "bold_red",
    }
    LEVEL_TAG_MAP: ClassVar[dict[int, str]] = {
        logging.DEBUG: "[D]",
        logging.INFO: "[I]",
        logging.WARNING: "[W]",
        logging.ERROR: "[E]",
        logging.CRITICAL: "[C]",
    }
    FORMAT: ClassVar[str] = (
        "%(reltime)s %(color)s%(leveltag)s%(color_reset)s%(condname)s %(message)s"
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(self.FORMAT, *args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        record.condname = f" ({record.name})" if record.name != "__main__" else ""
        record.leveltag = self.LEVEL_TAG_MAP.get(record.levelno, "[?]")
        record.color = getattr(term, self.COLOR_MAP.get(record.levelno, "white"))
        record.color_reset = term.normal
        record.reltime = f"{term.indigo}{record.relativeCreated / 1000:5.1f}s{term.normal}"
        return super().format(record)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="whispersubs", description="Transcribe audio/video files into subtitles"
    )
    parser.add_argument("input", type=Path, help="Input audio/video file")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="Output SRT file. Defaults to input file name with '.<lang>' suffix and '.srt' extension.",
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
        "--model-size",
        choices=available_models(),
        default=DEFAULT_WHISPER_MODEL_SIZE,
        help="Whisper model size to use for transcription",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default=DEFAULT_MODEL_DEVICE,
        help="Device to use for model inference",
    )
    parser.add_argument(
        "--split-long-lines",
        nargs="?",
        type=int,
        const=DEFAULT_SPLIT_LINES_LENGTH,
        help="Split long lines into multiple subtitles",
    )
    parser.add_argument(
        "--join-gaps",
        nargs="?",
        type=float,
        const=DEFAULT_JOIN_GAPS_DURATION,
        help="Join subtitles with gaps shorter than the specified duration",
    )
    parser.add_argument(
        "--log-level",
        choices=["debug", "info", "warning", "error", "critical"],
        default="info",
        help="Set the logging level",
    )
    parser.add_argument(
        "--log-whisper",
        action="store_true",
        help="Show logs from the faster-whisper library",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar",
    )
    args = parser.parse_args()

    root_logger = logging.getLogger()
    root_logger.setLevel(args.log_level.upper())
    # get stream handler and set formatter
    stream_handler = next(h for h in root_logger.handlers if isinstance(h, logging.StreamHandler))
    stream_handler.setFormatter(LogFormatter())

    if not args.log_whisper:
        logging.getLogger("urllib3.connectionpool").setLevel(logging.INFO)  # spammy logs
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)  # spammy logs

    transcribe_to_subtitles(
        args.input,
        args.output,
        model_size=args.model_size,
        device=args.device,
        language=args.language,
        translate=args.translate,
        progress=not args.no_progress,
        split_lines_length=args.split_long_lines,
        join_gaps_duration=args.join_gaps,
    )


if __name__ == "__main__":
    main()
