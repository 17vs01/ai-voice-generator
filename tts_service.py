"""
tts_service.py
ElevenLabs / OpenAI TTS / Google TTS(gTTS) 통합 음성 생성 로직

주요 기능
  - 여러 엔진의 목소리 목록 통합 (엔진 하나가 실패해도 앱은 계속 동작)
  - 긴 텍스트 자동 분할 + 문단 사이 쉼(pause)
  - ffmpeg 기반 속도/피치 독립 조절 (치핑멍크 없이)
  - MP3 → 텍스트 변환(STT) + 자막(SRT) 생성
  - OpenAI 로 텍스트 교정/요약/번역
"""

import io
import os
import re
import shutil
import subprocess

from dotenv import load_dotenv

load_dotenv()

# ── 성별 아이콘 ───────────────────────────────────────────────
GENDER_ICON = {"male": "👨", "female": "👩"}

# ── OpenAI 목소리 목록 (gpt-4o-mini-tts 지원 11종) ────────────
OPENAI_MODEL = "gpt-4o-mini-tts"          # 최신 TTS 모델 (구형 6종 + 신규 5종 지원)
_OPENAI_ORIGINAL = {"alloy", "echo", "fable", "onyx", "nova", "shimmer"}  # tts-1 로도 되는 것들
OPENAI_VOICES = [
    {"voice_id": "oai_alloy",   "name": "Alloy",   "gender": "female"},
    {"voice_id": "oai_ash",     "name": "Ash",     "gender": "male"},
    {"voice_id": "oai_ballad",  "name": "Ballad",  "gender": "male"},
    {"voice_id": "oai_coral",   "name": "Coral",   "gender": "female"},
    {"voice_id": "oai_echo",    "name": "Echo",    "gender": "male"},
    {"voice_id": "oai_fable",   "name": "Fable",   "gender": "male"},
    {"voice_id": "oai_nova",    "name": "Nova",    "gender": "female"},
    {"voice_id": "oai_onyx",    "name": "Onyx",    "gender": "male"},
    {"voice_id": "oai_sage",    "name": "Sage",    "gender": "female"},
    {"voice_id": "oai_shimmer", "name": "Shimmer", "gender": "female"},
    {"voice_id": "oai_verse",   "name": "Verse",   "gender": "male"},
]

# ── ElevenLabs 모델 목록 ──────────────────────────────────────
ELEVEN_DEFAULT_MODEL = "eleven_multilingual_v2"   # 폴백용 안정 모델
ELEVEN_MODELS = {
    "eleven_v3":              "v3 — 가장 표현력·감정 (최신, 계정 권한 필요할 수 있음)",
    "eleven_multilingual_v2": "Multilingual v2 — 고품질 안정 (권장)",
    "eleven_turbo_v2_5":      "Turbo v2.5 — 빠름·저지연",
    "eleven_flash_v2_5":      "Flash v2.5 — 가장 빠름",
}
_EL_UNAVAILABLE: set = set()   # 이 프로세스에서 접근 불가로 확인된 모델(반복 재시도 방지)

# ── Azure Speech 목소리 목록 ──────────────────────────────────
# HD(DragonHD) = LLM 기반, 문맥을 읽고 감정을 '자동으로' 연기 (태그 불필요).
# 다국어 HD/Multilingual은 한국어 입력도 그 음색으로 읽어요.
# ※ HD는 일부 리전 전용(southeastasia/eastus 등) — koreacentral 키면 표준만 써요.
AZURE_VOICES = [
    # HD (감정 자동 연기)
    {"voice_id": "az_en-US-Ava:DragonHDLatestNeural",    "name": "Ava HD (감정연기)",    "gender": "female", "hd": True},
    {"voice_id": "az_en-US-Emma:DragonHDLatestNeural",   "name": "Emma HD (감정연기)",   "gender": "female", "hd": True},
    {"voice_id": "az_en-US-Andrew:DragonHDLatestNeural", "name": "Andrew HD (감정연기)", "gender": "male",   "hd": True},
    {"voice_id": "az_en-US-Brian:DragonHDLatestNeural",  "name": "Brian HD (감정연기)",  "gender": "male",   "hd": True},
    # 한국어 표준 뉴럴
    {"voice_id": "az_ko-KR-SunHiNeural",              "name": "선히 (Azure)", "gender": "female", "hd": False},
    {"voice_id": "az_ko-KR-InJoonNeural",             "name": "인준 (Azure)", "gender": "male",   "hd": False},
    {"voice_id": "az_ko-KR-HyunsuMultilingualNeural", "name": "현수 (Azure)", "gender": "male",   "hd": False},
    # 영어 음색 다국어 (표준)
    {"voice_id": "az_en-US-AvaMultilingualNeural",    "name": "Ava (다국어)",    "gender": "female", "hd": False},
    {"voice_id": "az_en-US-AndrewMultilingualNeural", "name": "Andrew (다국어)", "gender": "male",   "hd": False},
]

# ── gTTS 목소리(언어) 목록 → API 키가 하나도 없어도 항상 사용 가능 ──
GTTS_VOICES = [
    {"voice_id": "gtts_ko",    "name": "Google 한국어"},
    {"voice_id": "gtts_en",    "name": "Google English"},
    {"voice_id": "gtts_ja",    "name": "Google 日本語"},
    {"voice_id": "gtts_zh-CN", "name": "Google 中文"},
    {"voice_id": "gtts_es",    "name": "Google Español"},
    {"voice_id": "gtts_fr",    "name": "Google Français"},
    {"voice_id": "gtts_de",    "name": "Google Deutsch"},
]

# 텍스트 제한 / 분할 기준
MAX_CHARS  = 5000   # 앱에서 허용하는 최대 글자 수
CHUNK_SIZE = 1500   # 한 번에 엔진으로 보낼 최대 글자 수 (긴 텍스트 자동 분할)


# ════════════════════════════════════════════════════════════
# 클라이언트
# ════════════════════════════════════════════════════════════
def _has_key(name: str, placeholder: str) -> bool:
    key = os.getenv(name, "")
    return bool(key) and key != placeholder


def _get_elevenlabs_client():
    from elevenlabs.client import ElevenLabs
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


def has_elevenlabs() -> bool:
    return _has_key("ELEVENLABS_API_KEY", "your_elevenlabs_api_key_here")


def has_openai() -> bool:
    return _has_key("OPENAI_API_KEY", "your_openai_api_key_here")


def has_azure() -> bool:
    return _has_key("AZURE_SPEECH_KEY", "your_azure_speech_key_here")


def _azure_region() -> str:
    return os.getenv("AZURE_SPEECH_REGION", "").strip() or "southeastasia"


# ════════════════════════════════════════════════════════════
# 목소리 목록
# ════════════════════════════════════════════════════════════
def get_voices() -> list[dict]:
    """
    ElevenLabs + OpenAI + gTTS 목소리 목록을 합쳐서 반환해요.
    엔진 하나가 실패하더라도 나머지 목소리는 그대로 돌려줘요.
    (gTTS 는 키가 필요 없어서 항상 최소 몇 개는 나와요.)
    """
    voices: list[dict] = []

    # ── ElevenLabs 목소리 (실패해도 앱 전체를 멈추지 않음) ──
    if has_elevenlabs():
        try:
            client = _get_elevenlabs_client()
            response = client.voices.get_all()
            for v in response.voices:
                labels = v.labels or {}
                gender = (labels.get("gender") or "unknown").lower()
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
        except Exception:
            # ElevenLabs 실패는 조용히 건너뛰고 다른 엔진으로 계속 진행
            pass

    # ── OpenAI 목소리 ──
    if has_openai():
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

    # ── Azure 목소리 (책읽기 앱과 같은 키 공유 — 무료 월 50만 자) ──
    if has_azure():
        for v in AZURE_VOICES:
            icon = "🎭" if v["hd"] else GENDER_ICON.get(v["gender"], "🎙️")
            voices.append({
                "voice_id":     v["voice_id"],
                "name":         v["name"],
                "category":     "Azure HD" if v["hd"] else "Azure",
                "gender":       v["gender"],
                "gender_icon":  icon,
                "display_name": f"{icon} {v['name']} [Azure]",
                "provider":     "azure",
            })

    # ── gTTS 목소리 (항상 추가) ──
    for v in GTTS_VOICES:
        voices.append({
            "voice_id":     v["voice_id"],
            "name":         v["name"],
            "category":     "Google",
            "gender":       "unknown",
            "gender_icon":  "🌐",
            "display_name": f"🌐 {v['name']} [Google]",
            "provider":     "gtts",
        })

    # 여성 → 남성 → 기타 순 정렬
    order = {"female": 0, "male": 1}
    return sorted(voices, key=lambda x: (order.get(x["gender"], 2), x["name"]))


# ════════════════════════════════════════════════════════════
# ElevenLabs 사용량 (남은 크레딧)
# ════════════════════════════════════════════════════════════
def get_elevenlabs_usage() -> dict | None:
    """ElevenLabs 문자 사용량을 반환해요. {"used": int, "limit": int} 또는 None."""
    if not has_elevenlabs():
        return None
    try:
        client = _get_elevenlabs_client()
        sub = None
        # SDK 버전에 따라 메서드 이름이 달라서 방어적으로 시도
        for getter in (
            lambda: client.user.get_subscription(),
            lambda: client.user.subscription.get(),
        ):
            try:
                sub = getter()
                break
            except Exception:
                continue
        if sub is None:
            return None
        used  = getattr(sub, "character_count", None)
        limit = getattr(sub, "character_limit", None)
        if used is None:
            return None
        return {"used": int(used), "limit": int(limit) if limit else None}
    except Exception:
        return None


# ════════════════════════════════════════════════════════════
# TTS 엔진별 함수
# ════════════════════════════════════════════════════════════
def _tts_elevenlabs(text: str, voice_id: str, settings: dict | None = None) -> bytes:
    from elevenlabs import VoiceSettings
    client = _get_elevenlabs_client()
    s = settings or {}
    vs = VoiceSettings(
        stability=s.get("stability", 0.4),            # 낮을수록 더 감정적/사람다움
        similarity_boost=s.get("similarity_boost", 0.75),
        style=s.get("style", 0.2),                    # 표현력(0~1)
        use_speaker_boost=s.get("use_speaker_boost", True),
    )

    model_id = s.get("model_id") or ELEVEN_DEFAULT_MODEL
    if model_id in _EL_UNAVAILABLE:                   # 이미 안 되는 걸로 확인된 모델은 건너뜀
        model_id = ELEVEN_DEFAULT_MODEL

    def _convert(mid: str) -> bytes:
        gen = client.text_to_speech.convert(
            voice_id=voice_id, text=text, model_id=mid, voice_settings=vs,
        )
        return b"".join(gen)

    try:
        return _convert(model_id)
    except Exception as e:
        # 모델 접근 불가/미지원이면 안정 모델로 1회 폴백 (한도 초과는 상위에서 gTTS 로 처리)
        if model_id != ELEVEN_DEFAULT_MODEL and not _is_quota_error(e):
            _EL_UNAVAILABLE.add(model_id)
            return _convert(ELEVEN_DEFAULT_MODEL)
        raise


def _tts_openai(text: str, voice_id: str, instructions: str | None = None) -> bytes:
    """voice_id 예: oai_nova → nova. instructions 로 말투/감정 지정 가능(gpt-4o-mini-tts)."""
    real_voice = voice_id.replace("oai_", "")
    client = _get_openai_client()
    kwargs = {"model": OPENAI_MODEL, "voice": real_voice, "input": text}
    if instructions:
        kwargs["instructions"] = instructions
    try:
        return client.audio.speech.create(**kwargs).content
    except Exception as e:
        # 최신 모델이 계정에서 안 되면, 구형 6종은 tts-1 로 안전하게 재시도
        # (tts-1 은 instructions 미지원이라 말투 지시는 빠져요)
        if not _is_quota_error(e) and real_voice in _OPENAI_ORIGINAL:
            return client.audio.speech.create(model="tts-1", voice=real_voice, input=text).content
        raise


def _tts_gtts(text: str, lang: str = "ko") -> bytes:
    from gtts import gTTS
    buf = io.BytesIO()
    gTTS(text=text, lang=lang).write_to_fp(buf)
    buf.seek(0)
    return buf.read()


def _tts_azure(text: str, voice_id: str) -> bytes:
    """voice_id 예: az_ko-KR-SunHiNeural → ko-KR-SunHiNeural.
    Azure 공식 REST — HD 음성은 문맥 기반으로 감정을 자동 연기해요."""
    import requests

    voice = voice_id.removeprefix("az_")
    key = os.getenv("AZURE_SPEECH_KEY", "")
    region = _azure_region()

    # SSML의 xml:lang 은 음성 이름 앞부분(ko-KR / en-US)에서 추출
    parts = voice.split("-")
    lang = f"{parts[0]}-{parts[1]}" if len(parts) >= 2 else "ko-KR"

    esc = (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
               .replace("'", "&apos;").replace('"', "&quot;"))
    ssml = (f"<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
            f"xml:lang='{lang}'><voice name='{voice}'>{esc}</voice></speak>")

    resp = requests.post(
        f"https://{region}.tts.speech.microsoft.com/cognitiveservices/v1",
        headers={
            "Ocp-Apim-Subscription-Key": key,
            "Content-Type": "application/ssml+xml",
            "X-Microsoft-OutputFormat": "audio-24khz-48kbitrate-mono-mp3",
            "User-Agent": "ai-voice-generator",
        },
        data=ssml.encode("utf-8"),
        timeout=30,
    )
    if resp.status_code == 200:
        if len(resp.content) < 64:
            raise RuntimeError("Azure 응답이 비어있어요 (음성 이름 확인)")
        return resp.content
    if resp.status_code in (401, 403):
        raise RuntimeError(f"Azure 키/리전 오류 ({resp.status_code}) — .env 확인")
    if resp.status_code == 429:
        raise RuntimeError("Azure 429: 무료 한도 초과 또는 요청 제한")
    if resp.status_code == 400:
        raise RuntimeError(
            f"Azure 400: '{voice}' 음성이 {region} 리전에 없어요 (HD는 일부 리전 전용)")
    raise RuntimeError(f"Azure 합성 실패 HTTP {resp.status_code}: {resp.text[:150]}")


# ════════════════════════════════════════════════════════════
# 언어 감지 / 텍스트 분할 / 오디오 이어붙이기
# ════════════════════════════════════════════════════════════
def _detect_lang(text: str) -> str:
    """gTTS 폴백에 쓸 언어를 대략 감지해요 (한글 있으면 ko, 아니면 영어 위주면 en)."""
    for ch in text:
        if "가" <= ch <= "힣":   # 한글
            return "ko"
        if "぀" <= ch <= "ヿ":   # 히라가나/가타카나
            return "ja"
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    return "en" if ascii_ratio > 0.6 else "ko"


def _split_into_chunks(text: str, max_len: int = CHUNK_SIZE) -> list[str]:
    """긴 텍스트를 문장 단위로 max_len 이하 덩어리로 나눠요."""
    text = text.strip()
    if len(text) <= max_len:
        return [text]

    sentences = re.split(r"(?<=[.!?。！？\n])\s*", text)
    chunks: list[str] = []
    cur = ""
    for s in sentences:
        if not s:
            continue
        if len(cur) + len(s) <= max_len:
            cur += s
        else:
            if cur:
                chunks.append(cur)
            if len(s) <= max_len:
                cur = s
            else:
                # 한 문장이 너무 길면 강제로 잘라요
                for i in range(0, len(s), max_len):
                    chunks.append(s[i:i + max_len])
                cur = ""
    if cur:
        chunks.append(cur)
    return chunks or [text]


def _concat_audios(audio_list: list[bytes], pause_ms: int = 0) -> bytes:
    """여러 MP3 바이트를 이어붙여요. pause_ms 만큼 사이에 무음을 넣을 수 있어요."""
    from pydub import AudioSegment

    combined = None
    silence = AudioSegment.silent(duration=pause_ms) if pause_ms > 0 else None
    for b in audio_list:
        seg = AudioSegment.from_file(io.BytesIO(b), format="mp3")
        if combined is None:
            combined = seg
        else:
            if silence is not None:
                combined += silence
            combined += seg

    buf = io.BytesIO()
    combined.export(buf, format="mp3")
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════════
# 메인 TTS 함수
# ════════════════════════════════════════════════════════════
def _is_quota_error(e: Exception) -> bool:
    err = str(e).lower()
    return any(k in err for k in ("quota", "limit", "429", "402", "credit", "insufficient"))


def _synth_one(text: str, voice_id: str,
               voice_settings: dict | None = None,
               instructions: str | None = None) -> tuple[bytes, str]:
    """단일 덩어리를 음성으로 변환. (bytes, service) 반환. 한도 초과 시 gTTS 로 폴백."""
    # ── gTTS 목소리 ──
    if voice_id.startswith("gtts_"):
        lang = voice_id.split("_", 1)[1] or "ko"
        return _tts_gtts(text, lang), "gtts"

    # ── Azure 목소리 (한도 초과 시 gTTS 폴백) ──
    if voice_id.startswith("az_"):
        try:
            return _tts_azure(text, voice_id), "azure"
        except Exception as e:
            if _is_quota_error(e):
                return _tts_gtts(text, _detect_lang(text)), "gtts"
            raise RuntimeError(f"❌ Azure TTS 실패: {e}")

    # ── OpenAI 목소리 (한도 초과 시 gTTS 폴백) ──
    if voice_id.startswith("oai_"):
        try:
            return _tts_openai(text, voice_id, instructions), "openai"
        except Exception as e:
            if _is_quota_error(e):
                return _tts_gtts(text, _detect_lang(text)), "gtts"
            raise RuntimeError(f"❌ OpenAI TTS 실패: {e}")

    # ── ElevenLabs (한도 초과 시 gTTS 폴백) ──
    try:
        return _tts_elevenlabs(text, voice_id, voice_settings), "elevenlabs"
    except Exception as e:
        if _is_quota_error(e):
            try:
                return _tts_gtts(text, _detect_lang(text)), "gtts"
            except Exception as e2:
                raise RuntimeError(f"⚠️ ElevenLabs 한도 초과, Google TTS도 실패: {e2}")
        raise RuntimeError(f"❌ 음성 생성 실패: {e}")


def _synth_engine(text: str, voice_id: str,
                  voice_settings: dict | None = None,
                  instructions: str | None = None) -> tuple[bytes, str]:
    """긴 텍스트는 자동 분할해서 합쳐요."""
    chunks = _split_into_chunks(text)
    if len(chunks) == 1:
        return _synth_one(chunks[0], voice_id, voice_settings, instructions)

    parts: list[bytes] = []
    service = None
    for c in chunks:
        b, s = _synth_one(c, voice_id, voice_settings, instructions)
        parts.append(b)
        service = "gtts" if (service == "gtts" or s == "gtts") else s
    return _concat_audios(parts, 0), service or "unknown"


def text_to_speech(text: str, voice_id: str, pause_ms: int = 0,
                   voice_settings: dict | None = None,
                   instructions: str | None = None) -> tuple[bytes, str]:
    """
    텍스트를 음성으로 변환해요.

    Args:
        text           : 읽어줄 텍스트 (자동으로 긴 텍스트는 분할)
        voice_id       : 목소리 ID
        pause_ms       : 문단(빈 줄) 사이에 넣을 무음 길이(ms). 0이면 붙여서 생성.
        voice_settings : ElevenLabs 감정 설정 {stability, style, similarity_boost, use_speaker_boost}
        instructions   : OpenAI 말투/감정 지시문 (gpt-4o-mini-tts)

    Returns:
        (MP3 bytes, 사용된 서비스명)
    """
    if not text or not text.strip():
        raise ValueError("❌ 텍스트를 입력해주세요!")
    if len(text) > MAX_CHARS:
        raise ValueError(f"❌ 텍스트가 너무 길어요! {MAX_CHARS}자 이하로 입력해주세요.")

    # 문단 쉼이 있으면 빈 줄 기준으로 나눠서 사이에 무음을 넣어요
    if pause_ms and pause_ms > 0:
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    else:
        paragraphs = [text]

    parts: list[bytes] = []
    service = None
    for p in paragraphs:
        b, s = _synth_engine(p, voice_id, voice_settings, instructions)
        parts.append(b)
        service = "gtts" if (service == "gtts" or s == "gtts") else s

    if len(parts) == 1 and not (pause_ms and pause_ms > 0):
        return parts[0], service or "unknown"
    return _concat_audios(parts, pause_ms or 0), service or "unknown"


# ════════════════════════════════════════════════════════════
# 속도 / 피치 조절 (ffmpeg 기반 → 속도·피치 독립 제어)
# ════════════════════════════════════════════════════════════
def _adjust_audio_pydub(audio_bytes: bytes, speed: float, pitch: int) -> bytes:
    """ffmpeg 직접 호출이 실패할 때를 위한 예비 방식 (속도/피치가 서로 간섭할 수 있음)."""
    from pydub import AudioSegment

    seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    if speed != 1.0:
        seg = seg._spawn(seg.raw_data, overrides={"frame_rate": int(seg.frame_rate * speed)})
        seg = seg.set_frame_rate(44100)
    if pitch != 0:
        rate = int(seg.frame_rate * (2 ** (pitch / 12.0)))
        seg = seg._spawn(seg.raw_data, overrides={"frame_rate": rate})
        seg = seg.set_frame_rate(44100)
    buf = io.BytesIO()
    seg.export(buf, format="mp3")
    buf.seek(0)
    return buf.read()


def adjust_audio(audio_bytes: bytes, speed: float = 1.0, pitch: int = 0) -> bytes:
    """
    MP3 의 속도와 피치를 조절해요.

    ffmpeg 필터를 써서 속도와 피치를 서로 독립적으로 조절해요.
      - 피치: asetrate 로 음정을 바꾼 뒤 atempo 로 길이를 원래대로 되돌림
      - 속도: atempo 로 길이만 바꿈 (음정 유지)

    Args:
        speed : 재생 속도 (0.5 ~ 2.0)
        pitch : 피치 반음 (-12 ~ +12)
    """
    if speed == 1.0 and pitch == 0:
        return audio_bytes

    # 원본 샘플레이트 파악
    try:
        from pydub import AudioSegment
        sr = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3").frame_rate or 44100
    except Exception:
        sr = 44100

    filters: list[str] = []
    if pitch != 0:
        new_sr = int(sr * (2 ** (pitch / 12.0)))
        filters.append(f"asetrate={new_sr}")
        filters.append(f"aresample={sr}")
        filters.append(f"atempo={2 ** (-pitch / 12.0):.6f}")   # 피치로 바뀐 길이를 복원 (0.5~2.0)
    if speed != 1.0:
        filters.append(f"atempo={speed:.6f}")                  # 순수 속도 (0.5~2.0)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg and filters:
        try:
            proc = subprocess.run(
                [ffmpeg, "-hide_banner", "-loglevel", "error",
                 "-i", "pipe:0", "-af", ",".join(filters), "-f", "mp3", "pipe:1"],
                input=audio_bytes,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
            if proc.returncode == 0 and proc.stdout:
                return proc.stdout
        except Exception:
            pass  # 아래 pydub 예비 방식으로

    # 예비 방식
    try:
        return _adjust_audio_pydub(audio_bytes, speed, pitch)
    except Exception as e:
        raise RuntimeError(f"❌ 오디오 조절 실패: {e}")


# ════════════════════════════════════════════════════════════
# MP3 → 텍스트 변환 (ElevenLabs STT) + 자막(SRT)
# ════════════════════════════════════════════════════════════
def _fmt_ts(seconds: float) -> str:
    """초 → SRT 타임스탬프 (HH:MM:SS,mmm)"""
    if seconds is None or seconds < 0:
        seconds = 0
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _build_srt(result) -> str | None:
    """STT 결과에서 단어 타임스탬프를 읽어 SRT 자막을 만들어요. 타임스탬프가 없으면 None."""
    words = getattr(result, "words", None)
    if not words:
        return None

    toks = [w for w in words if getattr(w, "start", None) is not None]
    if not toks:
        return None

    # 12단어 또는 문장부호 기준으로 자막 줄을 끊어요
    cues: list[list] = []
    cur: list = []
    for w in toks:
        cur.append(w)
        txt = getattr(w, "text", "") or ""
        if len(cur) >= 12 or any(p in txt for p in ".?!。！？\n"):
            cues.append(cur)
            cur = []
    if cur:
        cues.append(cur)

    lines: list[str] = []
    for i, cue in enumerate(cues, 1):
        start = getattr(cue[0], "start", 0)
        end   = getattr(cue[-1], "end", start)
        text  = "".join(getattr(w, "text", "") or "" for w in cue).strip()
        if not text:
            continue
        lines.append(f"{i}\n{_fmt_ts(start)} --> {_fmt_ts(end)}\n{text}\n")
    return "\n".join(lines) if lines else None


def speech_to_text(audio_bytes: bytes, language: str | None = None) -> tuple[str, str | None]:
    """
    MP3/WAV/M4A 파일을 텍스트로 변환해요. (ElevenLabs STT)

    Returns:
        (텍스트, SRT 자막 또는 None)
    """
    try:
        client = _get_elevenlabs_client()
        buf = io.BytesIO(audio_bytes)
        buf.name = "audio.mp3"
        kwargs = {"file": buf, "model_id": "scribe_v1"}
        if language:
            kwargs["language_code"] = language
        result = client.speech_to_text.convert(**kwargs)
        text = (getattr(result, "text", None) or "").strip()
        srt  = _build_srt(result)
        return text, srt
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
        "naturalize":   (
            "다음 텍스트를 사람이 실제로 말하듯 자연스럽게 다듬어줘. "
            "딱딱한 문어체는 구어체로 바꾸고, 너무 긴 문장은 짧게 나누고, "
            "자연스러운 호흡을 위해 쉼표·마침표를 알맞게 넣어줘. "
            "숫자·기호·약어는 소리 내어 읽는 방식대로 풀어써줘(예: 3kg → 삼 킬로그램). "
            "의미는 그대로 두고, 다듬은 텍스트만 출력해:\n\n" + text
        ),
    }
    prompt = prompts.get(mode, prompts["refine"])

    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"❌ AI 텍스트 다듬기 실패: {e}")


# ════════════════════════════════════════════════════════════
# 엔진 상태 점검 (헬스체크)
# ════════════════════════════════════════════════════════════
def health_check() -> dict:
    """
    각 엔진을 실제로 살짝 호출해서 작동 여부를 확인해요.
    반환: {"gtts": {"ok": bool|None, "msg": str}, "openai": {...}, "elevenlabs": {...}}
      ok = True  → 정상
      ok = False → 문제 있음(메시지 참고, 대부분 한도 초과 → Google TTS 대체)
      ok = None  → 키 없음(목록에 아예 표시되지 않음)
    한 엔진의 모든 목소리는 상태를 공유해요(같은 API를 쓰므로).
    """
    results: dict = {}

    # ── Google TTS (키 불필요, 항상 테스트) ──
    try:
        _tts_gtts("테스트", "ko")
        results["gtts"] = {"ok": True, "msg": "정상 (키 불필요)"}
    except Exception as e:
        results["gtts"] = {"ok": False, "msg": str(e)[:100]}

    # ── OpenAI (짧은 문장으로 실제 테스트) ──
    if has_openai():
        try:
            _tts_openai("hi", "oai_alloy")
            results["openai"] = {"ok": True, "msg": f"정상 (모델: {OPENAI_MODEL})"}
        except Exception as e:
            if _is_quota_error(e):
                results["openai"] = {"ok": False, "msg": "한도 초과 → Google TTS로 자동 대체"}
            else:
                results["openai"] = {"ok": False, "msg": str(e)[:100]}
    else:
        results["openai"] = {"ok": None, "msg": "키 없음 (목록에 표시 안 됨)"}

    # ── Azure (음성목록 API로 키 검증 — 무료 글자수 소모 없음) ──
    if has_azure():
        try:
            import requests
            region = _azure_region()
            r = requests.get(
                f"https://{region}.tts.speech.microsoft.com/cognitiveservices/voices/list",
                headers={"Ocp-Apim-Subscription-Key": os.getenv("AZURE_SPEECH_KEY", "")},
                timeout=10,
            )
            if r.status_code == 200:
                n_hd = sum(1 for v in r.json() if ":DragonHD" in v.get("ShortName", ""))
                hd_note = f", HD {n_hd}종 사용 가능" if n_hd else ", HD 없음(리전 확인)"
                results["azure"] = {"ok": True, "msg": f"정상 ({region}{hd_note})"}
            elif r.status_code in (401, 403):
                results["azure"] = {"ok": False, "msg": f"키/리전 오류 ({r.status_code}) — .env 확인"}
            else:
                results["azure"] = {"ok": False, "msg": f"HTTP {r.status_code}"}
        except Exception as e:
            results["azure"] = {"ok": False, "msg": str(e)[:100]}
    else:
        results["azure"] = {"ok": None, "msg": "키 없음 (목록에 표시 안 됨)"}

    # ── ElevenLabs (크레딧은 사용량으로 확인, 불필요한 소모 방지) ──
    if has_elevenlabs():
        usage = get_elevenlabs_usage()
        if usage and usage.get("limit") and usage["used"] >= usage["limit"]:
            results["elevenlabs"] = {
                "ok": False,
                "msg": f"한도 초과 ({usage['used']:,}/{usage['limit']:,}자) → Google TTS로 대체",
            }
        else:
            remain = ""
            if usage and usage.get("limit"):
                remain = f" ({usage['limit'] - usage['used']:,}자 남음)"
            results["elevenlabs"] = {"ok": True, "msg": f"정상{remain}"}
    else:
        results["elevenlabs"] = {"ok": None, "msg": "키 없음 (목록에 표시 안 됨)"}

    return results
