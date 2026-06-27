import queue
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel


SAMPLE_RATE = 16000
OUTPUT_WAV = "recorded.wav"

audio_queue = queue.Queue()
recording = False
audio_chunks = []

model = WhisperModel(
    "small",
    device="cpu",
    compute_type="int8",
)


def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    if recording:
        audio_queue.put(indata.copy())


def record_worker():
    while recording:
        try:
            chunk = audio_queue.get(timeout=0.1)
            audio_chunks.append(chunk)
        except queue.Empty:
            pass


def transcribe_file(path):
    segments, info = model.transcribe(
        path,
        language="ko",
        vad_filter=True,
    )

    print(f"감지 언어: {info.language}, 확률: {info.language_probability:.2f}")

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())

    return " ".join(text_parts).strip()


def main():
    global recording
    global audio_chunks

    print("Enter를 누르면 녹음 시작")
    input()

    recording = True
    audio_chunks = []

    print("녹음 중... 다시 Enter를 누르면 종료")

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback,
    ):
        worker = threading.Thread(target=record_worker)
        worker.start()

        input()

        recording = False
        worker.join()

    print("녹음 종료")

    if not audio_chunks:
        print("녹음된 오디오가 없습니다.")
        return

    audio_data = np.concatenate(audio_chunks, axis=0)
    sf.write(OUTPUT_WAV, audio_data, SAMPLE_RATE)

    print(f"파일 저장: {OUTPUT_WAV}")
    print("텍스트 변환 중...")

    text = transcribe_file(OUTPUT_WAV)

    print("변환 결과:")
    print(text)


if __name__ == "__main__":
    main()

