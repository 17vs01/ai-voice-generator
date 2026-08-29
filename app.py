"""
app.py  –  AI 음성 생성기 v4.0
실행: streamlit run app.py

즐겨찾기·생성 기록은 브라우저 세션별로 따로 보관돼요 (사용자끼리 섞이지 않음).
"""

import io
import zipfile
from datetime import datetime

import streamlit as st
from tts_service import (
    get_voices,
    text_to_speech,
    speech_to_text,
    refine_text_with_ai,
    adjust_audio,
    get_elevenlabs_usage,
    health_check,
    ELEVEN_MODELS,
    MAX_CHARS,
)

# ── 페이지 설정 (반드시 첫 Streamlit 명령) ───────────────────
st.set_page_config(page_title="AI 음성 생성기", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f8f9fc; }
    section[data-testid="stSidebar"] { background: #1e1b4b; }
    section[data-testid="stSidebar"] * { color: white !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# 세션 상태 기반 저장 (사용자/세션별로 분리 → 서로 안 섞임)
# ════════════════════════════════════════════════════════════
def load_favorites() -> list:
    return st.session_state.setdefault("favorites", [])

def toggle_favorite(voice_id: str) -> None:
    favs = load_favorites()
    if voice_id in favs:
        favs.remove(voice_id)
    else:
        favs.append(voice_id)

def load_history() -> list:
    return st.session_state.setdefault("history", [])

def add_history(voice_name: str, text: str, service: str, audio: bytes) -> None:
    hist = load_history()
    hist.insert(0, {
        "time":      datetime.now().strftime("%Y-%m-%d %H:%M"),
        "voice":     voice_name,
        "text":      text[:80] + ("…" if len(text) > 80 else ""),
        "full_text": text,
        "service":   service,
        "audio":     audio,          # 실제 bytes 그대로 (정수 배열 X)
    })
    del hist[30:]                    # 최근 30개만 유지


# ════════════════════════════════════════════════════════════
# 대기 중인 프로그램적 상태 변경을 위젯 생성 '전에' 반영
#   (Streamlit 은 위젯 생성 후 그 key 를 바꿀 수 없어서 최상단에서 처리)
# ════════════════════════════════════════════════════════════
if "_goto" in st.session_state:
    st.session_state["nav"] = st.session_state.pop("_goto")
if "_pending_text" in st.session_state:
    st.session_state["text_key"] = st.session_state.pop("_pending_text")
if "_pending_stt" in st.session_state:
    st.session_state["stt_key"] = st.session_state.pop("_pending_stt")


SVC_LABELS = {"elevenlabs": "🟢 ElevenLabs", "openai": "🔵 OpenAI", "azure": "💠 Azure", "gtts": "🟡 Google TTS"}
LANGS = {"한국어": "ko", "English": "en", "日本語": "ja", "中文": "zh", "Español": "es", "Français": "fr"}

# OpenAI 말투 프리셋 (gpt-4o-mini-tts instructions)
OAI_TONE_PRESETS = {
    "기본": "",
    "다정하고 따뜻하게": "Speak in a warm, friendly, and caring tone, like talking to a close friend, with natural pauses.",
    "차분한 내레이션": "Speak calmly and clearly like a professional audiobook narrator, with natural breathing and pauses.",
    "밝고 활기차게": "Speak in a bright, upbeat, energetic and cheerful tone.",
    "뉴스 앵커": "Speak in a clear, confident, neutral news-anchor tone.",
    "감성적으로": "Speak slowly and emotionally, with expressive, heartfelt delivery.",
    "속삭이듯 부드럽게": "Speak softly and gently, almost whispering, in an intimate tone.",
}


# ════════════════════════════════════════════════════════════
# 목소리 목록 / 사용량 로드 (캐시)
# ════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def load_voices():
    return get_voices()

@st.cache_data(ttl=300)
def load_usage():
    return get_elevenlabs_usage()

try:
    all_voices = load_voices()
except Exception as e:
    st.error(str(e))
    st.stop()

voice_by_id = {v["voice_id"]: v for v in all_voices}


# ════════════════════════════════════════════════════════════
# 사이드바
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.title("⭐ 즐겨찾기")
    favs = load_favorites()
    fav_voices = [voice_by_id[fid] for fid in favs if fid in voice_by_id]

    if fav_voices:
        for fv in fav_voices:
            c1, c2 = st.columns([4, 1])
            c1.write(fv["display_name"])
            if c2.button("✖", key=f"del_fav_{fv['voice_id']}"):
                toggle_favorite(fv["voice_id"])
                st.rerun()
    else:
        st.caption("아직 즐겨찾기가 없어요!\n목소리 아래 ☆ 버튼을 눌러보세요.")

    st.divider()
    st.subheader("🧬 ElevenLabs 모델")
    el_model = st.selectbox("모델 선택", list(ELEVEN_MODELS.keys()),
                            format_func=lambda m: ELEVEN_MODELS[m], key="el_model")
    st.caption("v3가 가장 자연스러워요. 계정에서 안 되면 자동으로 v2로 대체됩니다.")

    st.divider()
    st.subheader("🌐 언어")
    lang_name = st.selectbox("음성→텍스트 / Google TTS 언어", list(LANGS.keys()), key="lang_name")
    lang_code = LANGS[lang_name]

    st.divider()
    total  = len(all_voices)
    female = sum(1 for v in all_voices if v["gender"] == "female")
    male   = sum(1 for v in all_voices if v["gender"] == "male")
    st.caption(f"🎙️ 전체 목소리: {total}개")
    st.caption(f"👩 여성: {female}개  |  👨 남성: {male}개")

    usage = load_usage()
    if usage and usage.get("limit"):
        used, limit = usage["used"], usage["limit"]
        st.divider()
        st.caption("🟢 ElevenLabs 사용량")
        st.progress(min(used / limit, 1.0))
        st.caption(f"{used:,} / {limit:,} 자 ({limit - used:,}자 남음)")

    st.divider()
    st.subheader("🩺 엔진 상태 점검")
    if st.button("지금 점검하기", use_container_width=True):
        with st.spinner("각 엔진을 확인하고 있어요..."):
            st.session_state["health"] = health_check()
    health = st.session_state.get("health")
    if health:
        icon = {True: "🟢", False: "⚠️", None: "⚪"}
        names = {"elevenlabs": "ElevenLabs", "openai": "OpenAI", "azure": "Azure", "gtts": "Google TTS"}
        for prov in ("elevenlabs", "openai", "azure", "gtts"):
            info = health.get(prov, {})
            st.caption(f"{icon.get(info.get('ok'), '⚪')} **{names[prov]}** — {info.get('msg', '')}")
    else:
        st.caption("버튼을 누르면 각 엔진을 실제로 테스트해요.")

    st.divider()
    st.caption("AI 음성 생성기 v4.0")


# ════════════════════════════════════════════════════════════
# 제목 + 상단 네비게이션 (라디오 → 탭 전환 시에도 rerun 발생)
# ════════════════════════════════════════════════════════════
st.title("🎙️ AI 음성 생성기")
st.caption("텍스트·MP3·일괄 생성으로 다양한 목소리를 만들어보세요!")

NAV = ["✍️ 텍스트로 생성", "🎵 MP3로 변환", "📦 일괄 생성", "📝 생성 기록"]
nav = st.radio("모드", NAV, horizontal=True, key="nav", label_visibility="collapsed")
st.divider()


# ════════════════════════════════════════════════════════════
# 공통: 목소리 선택 위젯 (라디오 필터 + 단일 셀렉트박스 + 미리듣기)
#   → 화면에 보이는 선택이 곧 실제 선택 (탭 방식의 오작동 해결)
# ════════════════════════════════════════════════════════════
_PREVIEW_SAMPLES = {
    "gtts_en":    "Hello! This is a quick voice preview.",
    "gtts_ja":    "こんにちは。これは声のプレビューです。",
    "gtts_zh-CN": "你好，这是一个语音预览。",
    "gtts_es":    "Hola, esta es una vista previa de la voz.",
    "gtts_fr":    "Bonjour, ceci est un aperçu de la voix.",
    "gtts_de":    "Hallo, das ist eine kurze Sprachvorschau.",
}

def _preview_text(voice_id: str) -> str:
    return _PREVIEW_SAMPLES.get(voice_id, "안녕하세요, 만나서 반가워요. 목소리 미리듣기입니다.")

def _make_preview(voice_id: str) -> None:
    """해당 목소리의 미리듣기 오디오를 생성해 세션에 캐시해요 (이미 있으면 건너뜀)."""
    cache = st.session_state.setdefault("preview_audio", {})
    if voice_id in cache:
        return
    with st.spinner("미리듣기 생성 중..."):
        try:
            audio, _ = text_to_speech(_preview_text(voice_id), voice_id)
            cache[voice_id] = audio
        except Exception as e:
            st.error(str(e))

def _voice_health_note(voice_id: str):
    """헬스체크를 돌렸다면, 선택한 목소리의 엔진 상태를 한 줄로 보여줘요."""
    provider = voice_by_id.get(voice_id, {}).get("provider")
    health = st.session_state.get("health")
    if not health or not provider:
        return
    info = health.get(provider)
    if not info:
        return
    if info["ok"] is True:
        st.caption(f"🟢 엔진 상태: {info['msg']}")
    elif info["ok"] is False:
        st.caption(f"⚠️ 엔진 상태: {info['msg']}")

def voice_selector_widget(key_prefix: str):
    """(voice_id, display_name) 반환. 하나만 선택되므로 결과가 명확해요."""
    filt = st.radio(
        "목소리 종류",
        ["전체 🎙️", "여성 👩", "남성 👨", "즐겨찾기 ⭐"],
        horizontal=True, key=f"filt_{key_prefix}",
    )

    if filt == "여성 👩":
        pool = [v for v in all_voices if v["gender"] == "female"]
    elif filt == "남성 👨":
        pool = [v for v in all_voices if v["gender"] == "male"]
    elif filt == "즐겨찾기 ⭐":
        pool = [voice_by_id[i] for i in load_favorites() if i in voice_by_id]
    else:
        pool = all_voices

    if not pool:
        st.info("해당하는 목소리가 없어요. 다른 탭에서 ☆ 버튼으로 즐겨찾기를 추가해보세요.")
        return None, None

    # ── 목소리 둘러보기: 여러 목소리를 눌러가며 미리듣고 고르기 ──
    with st.expander(f"🎧 목소리 둘러보기 — 눌러서 미리듣기 ({len(pool)}개)"):
        shown = pool[:30]
        for v in shown:
            row = st.columns([5, 1])
            row[0].write(v["display_name"])
            if row[1].button("🔊", key=f"bp_{key_prefix}_{v['voice_id']}"):
                _make_preview(v["voice_id"])
            bp = st.session_state.get("preview_audio", {}).get(v["voice_id"])
            if bp:
                st.audio(bp, format="audio/mp3")
        if len(pool) > 30:
            st.caption(f"…외 {len(pool) - 30}개. 필터(여성/남성/즐겨찾기)로 좁혀보세요.")

    opts = {v["display_name"]: v for v in pool}
    picked_name = st.selectbox("✅ 사용할 목소리", list(opts.keys()), key=f"sel_{key_prefix}")
    picked = opts[picked_name]
    vid = picked["voice_id"]
    _voice_health_note(vid)

    c1, c2 = st.columns(2)
    is_fav = vid in load_favorites()
    if c1.button("⭐ 즐겨찾기 해제" if is_fav else "☆ 즐겨찾기 추가",
                 key=f"fav_{key_prefix}", use_container_width=True):
        toggle_favorite(vid)
        st.rerun()
    if c2.button("🔊 미리듣기", key=f"prev_{key_prefix}", use_container_width=True):
        _make_preview(vid)

    pv = st.session_state.get("preview_audio", {}).get(vid)
    if pv:
        st.audio(pv, format="audio/mp3")

    return vid, picked["display_name"]


def speed_pitch_widget(key_prefix: str):
    col_sp, col_pt = st.columns(2)
    with col_sp:
        speed = st.slider("🐇 속도", 0.5, 2.0, 1.0, 0.1, key=f"speed_{key_prefix}",
                          help="1.0 = 기본 / 음정은 유지돼요")
        st.caption(f"현재: **{speed}x**")
    with col_pt:
        pitch = st.slider("🎵 피치(반음)", -12, 12, 0, 1, key=f"pitch_{key_prefix}",
                          help="0 = 기본 / 음수 = 낮게 / 양수 = 높게 (속도는 유지돼요)")
        label = "기본 ✅" if pitch == 0 else (f"+{pitch} 높음 🔼" if pitch > 0 else f"{pitch} 낮음 🔽")
        st.caption(f"현재: **{label}**")
    return speed, pitch


def naturalness_widget(key_prefix: str, voice_id):
    """선택한 목소리 엔진에 맞는 '자연스러움' 설정. (voice_settings, instructions) 반환."""
    provider = voice_by_id.get(voice_id, {}).get("provider") if voice_id else None
    voice_settings, instructions = None, None

    with st.expander("🎭 자연스러움 설정 — 더 사람처럼"):
        if provider == "elevenlabs":
            st.caption("안정성을 낮추면 더 감정적이고 사람 같아요(대신 가끔 흔들려요). 표현력을 올리면 억양이 살아나요.")
            stability = st.slider("안정성 (낮을수록 감정적)", 0.0, 1.0, 0.4, 0.05, key=f"stab_{key_prefix}")
            style     = st.slider("표현력 / 감정", 0.0, 1.0, 0.2, 0.05, key=f"style_{key_prefix}")
            similar   = st.slider("목소리 유사도", 0.0, 1.0, 0.75, 0.05, key=f"sim_{key_prefix}")
            boost     = st.checkbox("스피커 부스트(또렷하게)", value=True, key=f"boost_{key_prefix}")
            voice_settings = {"stability": stability, "style": style,
                              "similarity_boost": similar, "use_speaker_boost": boost,
                              "model_id": st.session_state.get("el_model")}
            st.caption(f"모델: **{st.session_state.get('el_model', '기본')}** (사이드바에서 변경)")
        elif provider == "openai":
            st.caption("말투를 지정하면 훨씬 자연스러워져요 (gpt-4o-mini-tts).")
            preset = st.selectbox("말투 프리셋", list(OAI_TONE_PRESETS.keys()), key=f"tone_{key_prefix}")
            custom = st.text_input("직접 지정 (선택)", key=f"tonec_{key_prefix}",
                                   placeholder="예: 느리고 부드럽게, 조금 긴장한 듯")
            instructions = custom.strip() or OAI_TONE_PRESETS[preset]
        else:
            st.caption("💡 Google TTS는 감정/말투 조절이 어려워요. 더 사람 같은 결과는 "
                       "**ElevenLabs / OpenAI 목소리**를 골라주세요. "
                       "'✨ 자연스럽게 다듬기'로 텍스트를 먼저 손보면 조금 나아져요.")
    return voice_settings, instructions


def result_player(audio: bytes, voice_name: str, service: str, file_name: str, dl_key: str):
    c1, c2 = st.columns([3, 1])
    c1.caption(f"목소리: **{voice_name}**")
    c2.caption(SVC_LABELS.get(service, service))
    st.audio(audio, format="audio/mp3")
    st.download_button("⬇️ MP3 다운로드", data=audio, file_name=file_name,
                       mime="audio/mpeg", use_container_width=True, key=dl_key)


# ════════════════════════════════════════════════════════════
# 섹션 1: 텍스트로 생성
# ════════════════════════════════════════════════════════════
if nav == NAV[0]:
    st.subheader("1️⃣ 목소리 선택")
    v_id, v_name = voice_selector_widget("t")

    st.divider()
    st.subheader("2️⃣ 텍스트 입력")
    text_input = st.text_area(
        "읽어줄 텍스트를 입력하세요",
        placeholder="예) 안녕하세요! AI 음성 생성기 테스트 중이에요.",
        height=150, max_chars=MAX_CHARS, key="text_key",
    )
    st.caption(f"글자 수: {len(text_input)} / {MAX_CHARS}  ·  긴 글은 자동으로 나눠서 이어붙여요")

    cols = st.columns(4)
    ai_actions = [
        (cols[0], "✨ 자연스럽게", "naturalize", "구어체·쉼·숫자 풀어쓰기"),
        (cols[1], "✏️ AI 교정", "refine", "맞춤법/문장 다듬기"),
        (cols[2], "📝 AI 요약", "summarize", "3문장 이내로 요약"),
        (cols[3], "🌏 한국어 번역", "translate_ko", "한국어로 번역"),
    ]
    for col, label, mode, help_txt in ai_actions:
        with col:
            if st.button(label, use_container_width=True, help=help_txt, key=f"ai_{mode}"):
                if text_input.strip():
                    with st.spinner("AI 처리 중..."):
                        try:
                            st.session_state["_pending_text"] = refine_text_with_ai(text_input, mode)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
                else:
                    st.warning("⚠️ 먼저 텍스트를 입력해주세요!")

    st.divider()
    st.subheader("3️⃣ 속도 / 피치 / 쉼 / 자연스러움")
    speed, pitch = speed_pitch_widget("t")
    pause_sec = st.slider("⏸️ 문단(빈 줄) 사이 쉼", 0.0, 2.0, 0.0, 0.1,
                          help="빈 줄로 나눈 문단 사이에 무음을 넣어요", key="pause_t")
    vs_t, instr_t = naturalness_widget("t", v_id)

    st.divider()
    st.subheader("4️⃣ 음성 생성")
    if st.button("🎵 음성 만들기", type="primary", use_container_width=True, key="gen_text"):
        if not text_input.strip():
            st.warning("⚠️ 텍스트를 먼저 입력해주세요!")
        elif not v_id:
            st.warning("⚠️ 목소리를 선택해주세요!")
        else:
            with st.spinner("🎙️ 음성을 만들고 있어요..."):
                try:
                    audio, service = text_to_speech(text_input, v_id, pause_ms=int(pause_sec * 1000),
                                                    voice_settings=vs_t, instructions=instr_t)
                    audio = adjust_audio(audio, speed=speed, pitch=pitch)
                    st.session_state.update({
                        "audio_data": audio, "audio_voice": v_name, "audio_service": service,
                    })
                    add_history(v_name, text_input, service, audio)
                    if service == "gtts" and not v_id.startswith("gtts_"):
                        st.warning("⚠️ 유료 엔진 한도 초과! Google TTS로 자동 전환됐어요.")
                    else:
                        st.success("✅ 음성 완성!")
                except Exception as e:
                    st.error(str(e))

    if "audio_data" in st.session_state:
        st.divider()
        st.subheader("5️⃣ 듣기 & 다운로드")
        result_player(st.session_state["audio_data"], st.session_state["audio_voice"],
                      st.session_state["audio_service"], "generated_voice.mp3", "dl_text")


# ════════════════════════════════════════════════════════════
# 섹션 2: MP3로 변환
# ════════════════════════════════════════════════════════════
elif nav == NAV[1]:
    st.subheader("🎵 오디오 파일 업로드")
    st.caption("MP3/WAV/M4A를 올리면 텍스트로 변환하고, 원하는 목소리로 다시 만들어줘요!")

    uploaded = st.file_uploader("파일을 올려주세요", type=["mp3", "wav", "m4a"])

    if uploaded:
        audio_bytes = uploaded.getvalue()          # getvalue → 포인터 문제 없이 안전
        st.audio(audio_bytes, format="audio/mp3")

        if st.button("🔤 텍스트로 변환 (STT)", use_container_width=True, key="stt_go"):
            with st.spinner("AI가 음성을 텍스트로 변환하고 있어요..."):
                try:
                    extracted, srt = speech_to_text(audio_bytes, language=lang_code)
                    st.session_state["_pending_stt"] = extracted
                    st.session_state["stt_srt"] = srt
                    st.session_state["stt_ready"] = True
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

        if st.session_state.get("stt_ready"):
            st.divider()
            st.subheader("📄 변환된 텍스트")
            stt_text = st.text_area("텍스트를 수정할 수 있어요!", height=150, key="stt_key")

            # 자막(SRT) 다운로드
            if st.session_state.get("stt_srt"):
                st.download_button("💬 자막(SRT) 다운로드", data=st.session_state["stt_srt"],
                                   file_name="subtitle.srt", mime="text/plain", key="dl_srt")

            cols2 = st.columns(4)
            stt_actions = [
                (cols2[0], "✨ 자연스럽게", "naturalize"),
                (cols2[1], "✏️ AI 교정", "refine"),
                (cols2[2], "📝 AI 요약", "summarize"),
                (cols2[3], "🌏 한국어 번역", "translate_ko"),
            ]
            for col, label, mode in stt_actions:
                with col:
                    if st.button(label, use_container_width=True, key=f"stt_{mode}"):
                        if stt_text.strip():
                            with st.spinner("AI 처리 중..."):
                                try:
                                    st.session_state["_pending_stt"] = refine_text_with_ai(stt_text, mode)
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))
                        else:
                            st.warning("⚠️ 텍스트가 비어 있어요!")

            st.divider()
            st.subheader("🎙️ 새 목소리 선택")
            v_id2, v_name2 = voice_selector_widget("mp3")

            st.divider()
            st.subheader("🎚️ 속도 / 피치 / 자연스러움")
            speed2, pitch2 = speed_pitch_widget("mp3")
            vs_m, instr_m = naturalness_widget("mp3", v_id2)

            if st.button("🎵 새 목소리로 생성!", type="primary", use_container_width=True, key="gen_mp3"):
                if not stt_text.strip():
                    st.warning("⚠️ 텍스트가 없어요!")
                elif not v_id2:
                    st.warning("⚠️ 목소리를 선택해주세요!")
                else:
                    with st.spinner("🎙️ 새 목소리로 생성 중..."):
                        try:
                            audio, service = text_to_speech(stt_text, v_id2,
                                                            voice_settings=vs_m, instructions=instr_m)
                            audio = adjust_audio(audio, speed=speed2, pitch=pitch2)
                            st.session_state.update({
                                "mp3_audio": audio, "mp3_service": service, "mp3_voice": v_name2,
                            })
                            add_history(v_name2, stt_text, service, audio)
                            st.success("✅ 완성!")
                        except Exception as e:
                            st.error(str(e))

            if "mp3_audio" in st.session_state:
                st.divider()
                result_player(st.session_state["mp3_audio"], st.session_state["mp3_voice"],
                              st.session_state["mp3_service"], "converted_voice.mp3", "dl_mp3")


# ════════════════════════════════════════════════════════════
# 섹션 3: 일괄 생성 (여러 문장 → 여러 MP3 → ZIP)
# ════════════════════════════════════════════════════════════
elif nav == NAV[2]:
    st.subheader("📦 일괄 생성")
    st.caption("한 줄에 하나씩 입력하면 각각 음성으로 만들어 ZIP으로 받을 수 있어요. (최대 20줄)")

    st.subheader("1️⃣ 목소리 선택")
    vb_id, vb_name = voice_selector_widget("b")

    st.divider()
    st.subheader("2️⃣ 문장 입력 (한 줄 = 하나)")
    batch_text = st.text_area("여러 줄 입력", height=200, key="batch_key",
                              placeholder="첫 번째 문장\n두 번째 문장\n세 번째 문장")
    lines = [l.strip() for l in batch_text.splitlines() if l.strip()]
    st.caption(f"입력된 문장: {len(lines)}개")

    st.divider()
    st.subheader("3️⃣ 속도 / 피치 / 자연스러움")
    speed3, pitch3 = speed_pitch_widget("b")
    vs_b, instr_b = naturalness_widget("b", vb_id)

    st.divider()
    if st.button("📦 일괄 생성 시작", type="primary", use_container_width=True, key="gen_batch"):
        if not lines:
            st.warning("⚠️ 문장을 입력해주세요!")
        elif not vb_id:
            st.warning("⚠️ 목소리를 선택해주세요!")
        elif len(lines) > 20:
            st.warning("⚠️ 한 번에 최대 20줄까지만 가능해요!")
        else:
            results = []
            progress = st.progress(0.0)
            errors = []
            for i, line in enumerate(lines):
                try:
                    audio, service = text_to_speech(line, vb_id,
                                                    voice_settings=vs_b, instructions=instr_b)
                    audio = adjust_audio(audio, speed=speed3, pitch=pitch3)
                    results.append({"text": line, "audio": audio, "service": service})
                    add_history(vb_name, line, service, audio)
                except Exception as e:
                    errors.append(f"{i+1}번째 줄: {e}")
                progress.progress((i + 1) / len(lines))
            st.session_state["batch_results"] = results
            if errors:
                st.error("일부 실패:\n" + "\n".join(errors))
            if results:
                st.success(f"✅ {len(results)}개 생성 완료!")

    if st.session_state.get("batch_results"):
        st.divider()
        st.subheader("🎧 결과")
        results = st.session_state["batch_results"]
        for i, r in enumerate(results):
            st.caption(f"**{i+1}.** {r['text'][:60]}  ·  {SVC_LABELS.get(r['service'], r['service'])}")
            st.audio(r["audio"], format="audio/mp3")

        zbuf = io.BytesIO()
        with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
            for i, r in enumerate(results, 1):
                z.writestr(f"{i:02d}.mp3", r["audio"])
        st.download_button("⬇️ 전체 ZIP 다운로드", data=zbuf.getvalue(),
                           file_name="batch_voices.zip", mime="application/zip",
                           use_container_width=True, key="dl_zip")


# ════════════════════════════════════════════════════════════
# 섹션 4: 생성 기록
# ════════════════════════════════════════════════════════════
elif nav == NAV[3]:
    st.subheader("📝 생성 기록")
    history = load_history()

    if not history:
        st.info("아직 생성 기록이 없어요! 음성을 만들면 여기에 쌓여요 😊")
    else:
        if st.button("🗑️ 기록 전체 삭제"):
            st.session_state["history"] = []
            st.rerun()

        for i, h in enumerate(history):
            svc_label = SVC_LABELS.get(h["service"], h["service"])
            with st.expander(f"**{h['time']}** | {h['voice']} | {svc_label} — {h['text']}"):
                audio_bytes = h["audio"]
                st.audio(audio_bytes, format="audio/mp3")
                col_dl, col_reuse = st.columns(2)
                col_dl.download_button("⬇️ 다운로드", data=audio_bytes,
                                       file_name=f"voice_{i+1}.mp3", mime="audio/mpeg",
                                       key=f"dl_hist_{i}", use_container_width=True)
                if col_reuse.button("🔄 이 텍스트로 다시 만들기", key=f"reuse_{i}",
                                    use_container_width=True):
                    st.session_state["_pending_text"] = h.get("full_text", h["text"])
                    st.session_state["_goto"] = NAV[0]     # '텍스트로 생성' 으로 자동 이동
                    st.rerun()
