from faster_whisper import WhisperModel

# GPU
#model = WhisperModel(
#    "large-v3",
#    device="cuda",
#    compute_type="float16"
#)
#
#segments, info = model.transcribe("audio.mp3", beam_size=5)
#
#print(f"Detected language: {info.language}, probability: {info.language_probability:.2f}")
#
#for segment in segments:
#    print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")


# CPU에서 돌릴 때는 보통 이렇게 씁니다.
model = WhisperModel(
    "small",                # large-v3
    device="cpu",           # cuda
    compute_type="int8"     # float16
)

segments, info = model.transcribe("recorded.wav", beam_size=5)

for segment in segments:
    print(segment.text)

# 한국어 음성임을 명시하려면:
#segments, info = model.transcribe(
#    "audio.mp3",
#    language="ko",
#    beam_size=5,
#    vad_filter=True
#)

# 단어별 타임스탬프:
#segments, _ = model.transcribe("audio.mp3", word_timestamps=True)
#
#for segment in segments:
#    for word in segment.words:
#        print(f"[{word.start:.2f}s -> {word.end:.2f}s] {word.word}")
