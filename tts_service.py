"""
tts_service.py
ElevenLabs / OpenAI TTS / Google TTS(gTTS) 통합 음성 생성 로직
"""

import io
import os
from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

load_dotenv()

# ── 성별 아이콘 ───────────────────────────────────────────────
GENDER_ICON = {"male": "👨", "female": "👩"}

# ── OpenAI 기본 목소리 목록 ───────────────────────────────────
OPENAI_VOICES = [
    {"voice_id": "oai_alloy",   "name": "Alloy",   "gender": "female", "category": "OpenAI"},
    {"voice_id": "oai_echo",    "name": "Echo",    "gender": "male",   "category": "OpenAI"},
    {"voice_id": "oai_fable",   "name": "Fable",   "gender": "male",   "category": "OpenAI"},
    {"voice_id": "oai_onyx",    "name": "Onyx",    "gender": "male",   "category": "OpenAI"},
    {"voice_id": "oai_nova",    "name": "Nova",    "gender": "female", "category": "OpenAI"},
    {"voice_id": "oai_shimmer", "name": "Shimmer", "gender": "female", "category": "OpenAI"},
]


# ════════════════════════════════════════════════════════════
# 클라이언트
# ════════════════════════════════════════════════════════════
def _get_elevenlabs_client() -> ElevenLabs:
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key or api_key == "your_elevenlabs_api_key_here":
        raise ValueError("❌ ElevenLabs API 키가 없어요! .env 파일을 확인해주세요.")
    return ElevenLabs(api_key=api_key)


def _get_openai_client():
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        raise ValueError("❌ OpenAI API 키가 없어요! .env 파일을 확인해주세요.")
    return OpenAI(api_key=api_key)


# ════════════════════════════════════════════════════════════
# 목소리 목록
# ════════════════════════════════════════════════════════════
def get_voices() -> list[dict]:
    """
    ElevenLabs + OpenAI 목소리 목록을 합쳐서 반환해요.
    반환값 예시:
      {"voice_id": "...", "name": "Rachel", "gender": "female",
       "gender_icon": "👩", "display_name": "👩 Rachel [ElevenLabs]", "category": "premade"}
    """
    voices = []

    # ── ElevenLabs 목소리 ──
    try:
        client = _get_elevenlabs_client()
        response = client.voices.get_all()
        for v in response.voices:
            labels = v.labels or {}
            gender = labels.get("gender", "unknown").lower()
            icon   = GENDER_ICON.get(gender, "🎙️")
            voices.append({
                "voice_id":     v.voice_id,
                "name":         v.name,
                "category":     v.category or "ElevenLabs",
                "gender":       gender,
                "gender_icon":  icon,
                "display_name": f"{icon} {v.name} [ElevenLabs]",
                "provider":     "elevenlabs",
            })
    except Exception as e:
        raise RuntimeError(f"❌ ElevenLabs 목소리 목록 오류: {e}")

    # ── OpenAI 목소리 ──
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key and openai_key != "your_openai_api_key_here":
        for v in OPENAI_VOICES:
            icon = GENDER_ICON.get(v["gender"], "🎙️")
            voices.append({
                "voice_id":     v["voice_id"],
                "name":         v["name"],
                "category":     "OpenAI",
                "gender":       v["gender"],
                "gender_icon":  icon,
                "display_name": f"{icon} {v['name']} [OpenAI]",
                "provider":     "openai",
            })

    # 여성 → 남성 → 기타 순 정렬
    order = {"female": 0, "male": 1}
    return sorted(voices, key=lambda x: (order.get(x["gender"], 2), x["name"]))


# ════════════════════════════════════════════════════════════
# TTS 엔진별 함수
# ════════════════════════════════════════════════════════════
def _tts_elevenlabs(text: str, voice_id: str) -> bytes:
    client = _get_elevenlabs_client()
    gen = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(stability=0.5, similarity_boost=0.75),
    )
    return b"".join(gen)


def _tts_openai(text: str, voice_id: str) -> bytes:
    """voice_id 예: oai_nova → nova"""
    real_voice = voice_id.replace("oai_", "")
    client = _get_openai_client()
    response = client.audio.speech.create(
        model="tts-1",
        voice=real_voice,
        input=text,
    )
    return response.content


def _tts_gtts(text: str) -> bytes:
    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text=text, lang="ko").write_to_fp(buf)
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════════
# 메인 TTS 함수
# ════════════════════════════════════════════════════════════
def text_to_speech(text: str, voice_id: str) -> tuple[bytes, str]:
    """
    텍스트를 음성으로 변환해요.
    반환값: (MP3 bytes, 사용된 서비스명)
    """
    if not text or not text.strip():
        raise ValueError("❌ 텍스트를 입력해주세요!")
    if len(text) > 2500:
        raise ValueError("❌ 텍스트가 너무 길어요! 2500자 이하로 입력해주세요.")

    # OpenAI 목소리 선택 시
    if voice_id.startswith("oai_"):
        try:
            return _tts_openai(text, voice_id), "openai"
        except Exception as e:
            raise RuntimeError(f"❌ OpenAI TTS 실패: {e}")

    # ElevenLabs → 실패 시 gTTS 자동 전환
    try:
        return _tts_elevenlabs(text, voice_id), "elevenlabs"
    except Exception as e:
        err = str(e).lower()
        if any(k in err for k in ("quota", "limit", "429", "402")):
            try:
                return _tts_gtts(text), "gtts"
            except Exception as e2:
                raise RuntimeError(f"⚠️ ElevenLabs 한도 초과, Google TTS도 실패: {e2}")
        raise RuntimeError(f"❌ 음성 생성 실패: {e}")


# ════════════════════════════════════════════════════════════
# 속도 / 피치 조절
# ════════════════════════════════════════════════════════════
def adjust_audio(audio_bytes: bytes, speed: float = 1.0, pitch: int = 0) -> bytes:
    """
    MP3 바이트의 속도와 피치를 조절해요.

    Args:
        audio_bytes : 원본 MP3 bytes
        speed       : 재생 속도 (0.5 ~ 2.0, 기본 1.0)
        pitch       : 피치 반음 단위 (-12 ~ +12, 기본 0)

    Returns:
        조절된 MP3 bytes
    """
    import io
    from pydub import AudioSegment

    # 변경 사항이 없으면 그대로 반환
    if speed == 1.0 and pitch == 0:
        return audio_bytes

    try:
        # bytes → AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")

        # ── 속도 조절 ──────────────────────────────────────
        if speed != 1.0:
            # frame_rate를 바꾸면 속도가 변해요 (음정도 같이 바뀌지만 pitch로 보정)
            new_frame_rate = int(seg.frame_rate * speed)
            seg = seg._spawn(seg.raw_data, overrides={"frame_rate": new_frame_rate})
            seg = seg.set_frame_rate(44100)

        # ── 피치 조절 ──────────────────────────────────────
        if pitch != 0:
            # 피치만 조절 (속도는 유지)
            pitch_frame_rate = int(seg.frame_rate * (2 ** (pitch / 12.0)))
            seg = seg._spawn(seg.raw_data, overrides={"frame_rate": pitch_frame_rate})
            seg = seg.set_frame_rate(44100)

        # AudioSegment → bytes
        buf = io.BytesIO()
        seg.export(buf, format="mp3")
        buf.seek(0)
        return buf.read()

    except Exception as e:
        raise RuntimeError(f"❌ 오디오 조절 실패: {e}")


# ════════════════════════════════════════════════════════════
# MP3 → 텍스트 변환 (ElevenLabs STT)
# ════════════════════════════════════════════════════════════
def speech_to_text(audio_bytes: bytes) -> str:
    """MP3 파일을 텍스트로 변환해요. (ElevenLabs STT)"""
    try:
        client = _get_elevenlabs_client()
        buf = io.BytesIO(audio_bytes)
        buf.name = "audio.mp3"
        result = client.speech_to_text.convert(
            file=buf,
            model_id="scribe_v1",
        )
        return result.text or ""
    except Exception as e:
        raise RuntimeError(f"❌ 음성 → 텍스트 변환 실패: {e}")


# ════════════════════════════════════════════════════════════
# AI 텍스트 다듬기 (OpenAI)
# ════════════════════════════════════════════════════════════
def refine_text_with_ai(text: str, mode: str = "summarize") -> str:
    """
    OpenAI로 텍스트를 다듬어요.
    mode: "summarize" (요약) | "refine" (교정) | "translate_ko" (한국어 번역)
    """
    prompts = {
        "summarize":    f"다음 텍스트를 3문장 이내로 핵심만 요약해줘. 결과만 출력해:\n\n{text}",
        "refine":       f"다음 텍스트의 맞춤법과 문장을 자연스럽게 다듬어줘. 결과만 출력해:\n\n{text}",
        "translate_ko": f"다음 텍스트를 자연스러운 한국어로 번역해줘. 결과만 출력해:\n\n{text}",
    }

    prompt = prompts.get(mode, prompts["refine"])

    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"❌ AI 텍스트 다듬기 실패: {e}")
