"""
app.py  –  AI 음성 생성기 v3.0
실행: py -3.11 -m streamlit run app.py
"""

import json
from datetime import datetime
from pathlib import Path

import streamlit as st
from tts_service import (
    get_voices,
    text_to_speech,
    speech_to_text,
    refine_text_with_ai,
    adjust_audio,
)

# ── 파일 경로 ────────────────────────────────────────────────
FAV_FILE  = Path("favorites.json")
HIST_FILE = Path("history.json")


# ════════════════════════════════════════════════════════════
# JSON 유틸
# ════════════════════════════════════════════════════════════
def _load(path: Path) -> list:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except Exception:
        return []

def _save(path: Path, data: list) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ════════════════════════════════════════════════════════════
# 즐겨찾기 헬퍼
# ════════════════════════════════════════════════════════════
def load_favorites() -> list[str]:
    return _load(FAV_FILE)

def save_favorites(favs: list[str]) -> None:
    _save(FAV_FILE, favs)

def toggle_favorite(voice_id: str) -> None:
    favs = load_favorites()
    if voice_id in favs:
        favs.remove(voice_id)
    else:
        favs.append(voice_id)
    save_favorites(favs)


# ════════════════════════════════════════════════════════════
# 히스토리 헬퍼
# ════════════════════════════════════════════════════════════
def load_history() -> list[dict]:
    return _load(HIST_FILE)

def add_history(voice_name: str, text: str, service: str, audio: bytes) -> None:
    hist = load_history()
    hist.insert(0, {
        "time":    datetime.now().strftime("%Y-%m-%d %H:%M"),
        "voice":   voice_name,
        "text":    text[:80] + ("…" if len(text) > 80 else ""),
        "full_text": text,
        "service": service,
        "audio":   list(audio),
    })
    _save(HIST_FILE, hist[:30])


# ════════════════════════════════════════════════════════════
# 페이지 설정
# ════════════════════════════════════════════════════════════
st.set_page_config(page_title="AI 음성 생성기", page_icon="🎙️", layout="wide")

st.markdown("""
<style>
    .stApp { background: #f8f9fc; }
    section[data-testid="stSidebar"] { background: #1e1b4b; }
    section[data-testid="stSidebar"] * { color: white !important; }
    .provider-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 99px;
        font-size: 0.75rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# 목소리 목록 로드 (캐시 1시간)
# ════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def load_voices():
    return get_voices()

try:
    all_voices = load_voices()
except Exception as e:
    st.error(str(e))
    st.stop()

voice_by_id = {v["voice_id"]: v for v in all_voices}


# ════════════════════════════════════════════════════════════
# 사이드바 – 즐겨찾기
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
        st.caption("아직 즐겨찾기가 없어요!\n목소리 옆 ⭐ 버튼을 눌러보세요.")

    st.divider()
    total = len(all_voices)
    female = sum(1 for v in all_voices if v["gender"] == "female")
    male   = sum(1 for v in all_voices if v["gender"] == "male")
    st.caption(f"🎙️ 전체 목소리: {total}개")
    st.caption(f"👩 여성: {female}개  |  👨 남성: {male}개")
    st.divider()
    st.caption("AI 음성 생성기 v3.0")


# ════════════════════════════════════════════════════════════
# 메인 화면 – 탭 구성
# ════════════════════════════════════════════════════════════
st.title("🎙️ AI 음성 생성기")
st.caption("텍스트 입력 또는 MP3 업로드로 다양한 목소리를 만들어보세요!")
st.divider()

main_tab1, main_tab2, main_tab3 = st.tabs(["✍️ 텍스트로 생성", "🎵 MP3로 변환", "📝 생성 기록"])


# ════════════════════════════════════════════════════════════
# 공통: 목소리 선택 위젯
# ════════════════════════════════════════════════════════════
def voice_selector_widget(key_prefix: str) -> tuple[str, str]:
    """목소리 선택 탭 + 즐겨찾기 버튼. (voice_id, display_name) 반환"""

    tab_all, tab_female, tab_male, tab_fav = st.tabs(
        ["전체 🎙️", "여성 👩", "남성 👨", "즐겨찾기 ⭐"]
    )

    selected_id   = None
    selected_name = None

    def _select(voices, prefix):
        nonlocal selected_id, selected_name
        if not voices:
            st.info("해당하는 목소리가 없어요.")
            return
        opts = {v["display_name"]: v for v in voices}
        picked_name = st.selectbox("목소리를 골라주세요", list(opts.keys()), key=f"sel_{prefix}")
        picked = opts[picked_name]
        selected_id   = picked["voice_id"]
        selected_name = picked["display_name"]

        is_fav = picked["voice_id"] in load_favorites()
        label  = "⭐ 즐겨찾기 해제" if is_fav else "☆ 즐겨찾기 추가"
        if st.button(label, key=f"fav_btn_{prefix}"):
            toggle_favorite(picked["voice_id"])
            st.rerun()

    with tab_all:
        _select(all_voices, f"{key_prefix}_all")
    with tab_female:
        _select([v for v in all_voices if v["gender"] == "female"], f"{key_prefix}_f")
    with tab_male:
        _select([v for v in all_voices if v["gender"] == "male"], f"{key_prefix}_m")
    with tab_fav:
        fav_list = [voice_by_id[fid] for fid in load_favorites() if fid in voice_by_id]
        _select(fav_list, f"{key_prefix}_fav")

    return selected_id, selected_name


# ════════════════════════════════════════════════════════════
# 탭 1: 텍스트로 생성
# ════════════════════════════════════════════════════════════
with main_tab1:
    st.subheader("1️⃣ 목소리 선택")
    v_id, v_name = voice_selector_widget("t")

    st.divider()
    st.subheader("2️⃣ 텍스트 입력")

    # 기록/AI결과에서 가져온 텍스트 자동 채우기
    if "reuse_text" in st.session_state:
        st.session_state["main_text"] = st.session_state.pop("reuse_text")

    text_input = st.text_area(
        "읽어줄 텍스트를 입력하세요",
        placeholder="예) 안녕하세요! AI 음성 생성기 테스트 중이에요.",
        height=150,
        max_chars=2500,
        key="main_text",
    )
    st.caption(f"글자 수: {len(text_input)} / 2500")

    # AI 다듬기 버튼
    col_r, col_s, col_t = st.columns(3)
    with col_r:
        if st.button("✏️ AI 교정", use_container_width=True, help="맞춤법/문장 다듬기"):
            if text_input.strip():
                with st.spinner("AI가 텍스트를 다듬고 있어요..."):
                    try:
                        refined = refine_text_with_ai(text_input, "refine")
                        st.session_state["main_text"] = refined
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
    with col_s:
        if st.button("📝 AI 요약", use_container_width=True, help="3문장 이내로 요약"):
            if text_input.strip():
                with st.spinner("AI가 요약하고 있어요..."):
                    try:
                        summarized = refine_text_with_ai(text_input, "summarize")
                        st.session_state["main_text"] = summarized
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
    with col_t:
        if st.button("🌏 한국어 번역", use_container_width=True, help="한국어로 번역"):
            if text_input.strip():
                with st.spinner("AI가 번역하고 있어요..."):
                    try:
                        translated = refine_text_with_ai(text_input, "translate_ko")
                        st.session_state["main_text"] = translated
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    st.divider()
    st.subheader("3️⃣ 속도 / 피치 조절")

    col_sp, col_pt = st.columns(2)
    with col_sp:
        speed = st.slider(
            "🐇 속도",
            min_value=0.5, max_value=2.0, value=1.0, step=0.1,
            help="1.0 = 기본 속도 / 낮을수록 느리고 높을수록 빨라요",
            key="speed_t",
        )
        speed_label = {0.5: "매우 느림 🐢", 0.75: "느림", 1.0: "기본 ✅", 1.25: "빠름", 1.5: "매우 빠름 🐇", 2.0: "초고속 ⚡"}.get(speed, f"{speed}x")
        st.caption(f"현재: **{speed}x** {speed_label}")

    with col_pt:
        pitch = st.slider(
            "🎵 피치",
            min_value=-12, max_value=12, value=0, step=1,
            help="0 = 기본 / 음수 = 낮은 목소리 / 양수 = 높은 목소리",
            key="pitch_t",
        )
        if pitch == 0:
            pitch_label = "기본 ✅"
        elif pitch > 0:
            pitch_label = f"+{pitch} 반음 (높음 🔼)"
        else:
            pitch_label = f"{pitch} 반음 (낮음 🔽)"
        st.caption(f"현재: **{pitch_label}**")

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
                    audio, service = text_to_speech(text_input, v_id)
                    # 속도/피치 후처리
                    audio = adjust_audio(audio, speed=speed, pitch=pitch)
                    st.session_state.update({
                        "audio_data": audio, "audio_text": text_input,
                        "audio_voice": v_name, "audio_service": service,
                    })
                    add_history(v_name, text_input, service, audio)
                    if service == "gtts":
                        st.warning("⚠️ ElevenLabs 한도 초과! Google TTS로 자동 전환됐어요.")
                    else:
                        st.success("✅ 음성 완성!")
                except Exception as e:
                    st.error(str(e))

    # 결과 재생
    if "audio_data" in st.session_state:
        st.divider()
        st.subheader("5️⃣ 듣기 & 다운로드")
        svc = st.session_state["audio_service"]
        svc_label = {"elevenlabs": "🟢 ElevenLabs", "openai": "🔵 OpenAI", "gtts": "🟡 Google TTS"}.get(svc, svc)
        c1, c2 = st.columns([3, 1])
        c1.caption(f"목소리: **{st.session_state['audio_voice']}**")
        c2.caption(svc_label)
        st.audio(st.session_state["audio_data"], format="audio/mp3")
        st.download_button("⬇️ MP3 다운로드", data=st.session_state["audio_data"],
                           file_name="generated_voice.mp3", mime="audio/mpeg",
                           use_container_width=True)


# ════════════════════════════════════════════════════════════
# 탭 2: MP3로 변환
# ════════════════════════════════════════════════════════════
with main_tab2:
    st.subheader("🎵 MP3 파일 업로드")
    st.caption("MP3를 올리면 AI가 텍스트로 변환하고, 원하는 목소리로 다시 만들어줘요!")

    uploaded = st.file_uploader("MP3 파일을 올려주세요", type=["mp3", "wav", "m4a"])

    if uploaded:
        st.audio(uploaded, format="audio/mp3")
        audio_bytes = uploaded.read()

        # ── STT ──
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔤 텍스트로 변환 (STT)", use_container_width=True):
                with st.spinner("AI가 음성을 텍스트로 변환하고 있어요..."):
                    try:
                        extracted = speech_to_text(audio_bytes)
                        st.session_state["stt_text"] = extracted
                        st.success("✅ 변환 완료!")
                    except Exception as e:
                        st.error(str(e))

        # STT 결과 표시 + AI 다듬기
        if "stt_text" in st.session_state:
            st.divider()
            st.subheader("📄 변환된 텍스트")
            stt_text = st.text_area("텍스트를 수정할 수 있어요!", value=st.session_state["stt_text"],
                                    height=150, key="stt_edit")

            col_r2, col_s2, col_t2 = st.columns(3)
            with col_r2:
                if st.button("✏️ AI 교정", use_container_width=True, key="stt_refine"):
                    with st.spinner("AI가 교정하고 있어요..."):
                        try:
                            refined = refine_text_with_ai(stt_text, "refine")
                            st.session_state["stt_text"] = refined
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
            with col_s2:
                if st.button("📝 AI 요약", use_container_width=True, key="stt_sum"):
                    with st.spinner("AI가 요약하고 있어요..."):
                        try:
                            summarized = refine_text_with_ai(stt_text, "summarize")
                            st.session_state["stt_text"] = summarized
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))
            with col_t2:
                if st.button("🌏 한국어 번역", use_container_width=True, key="stt_trans"):
                    with st.spinner("AI가 번역하고 있어요..."):
                        try:
                            translated = refine_text_with_ai(stt_text, "translate_ko")
                            st.session_state["stt_text"] = translated
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))

            st.divider()
            st.subheader("🎙️ 새 목소리 선택")
            v_id2, v_name2 = voice_selector_widget("mp3")

            st.divider()
            st.subheader("🎚️ 속도 / 피치 조절")
            col_sp2, col_pt2 = st.columns(2)
            with col_sp2:
                speed2 = st.slider(
                    "🐇 속도",
                    min_value=0.5, max_value=2.0, value=1.0, step=0.1,
                    help="1.0 = 기본 속도",
                    key="speed_mp3",
                )
                speed_label2 = {0.5: "매우 느림 🐢", 1.0: "기본 ✅", 2.0: "초고속 ⚡"}.get(speed2, f"{speed2}x")
                st.caption(f"현재: **{speed2}x** {speed_label2}")
            with col_pt2:
                pitch2 = st.slider(
                    "🎵 피치",
                    min_value=-12, max_value=12, value=0, step=1,
                    help="0 = 기본 / 음수 = 낮음 / 양수 = 높음",
                    key="pitch_mp3",
                )
                if pitch2 == 0:
                    pitch_label2 = "기본 ✅"
                elif pitch2 > 0:
                    pitch_label2 = f"+{pitch2} 반음 (높음 🔼)"
                else:
                    pitch_label2 = f"{pitch2} 반음 (낮음 🔽)"
                st.caption(f"현재: **{pitch_label2}**")

            if st.button("🎵 새 목소리로 생성!", type="primary", use_container_width=True, key="gen_mp3"):
                if not stt_text.strip():
                    st.warning("⚠️ 텍스트가 없어요!")
                elif not v_id2:
                    st.warning("⚠️ 목소리를 선택해주세요!")
                else:
                    with st.spinner("🎙️ 새 목소리로 생성 중..."):
                        try:
                            audio, service = text_to_speech(stt_text, v_id2)
                            # 속도/피치 후처리
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
                svc = st.session_state["mp3_service"]
                svc_label = {"elevenlabs": "🟢 ElevenLabs", "openai": "🔵 OpenAI", "gtts": "🟡 Google TTS"}.get(svc, svc)
                c1, c2 = st.columns([3, 1])
                c1.caption(f"목소리: **{st.session_state['mp3_voice']}**")
                c2.caption(svc_label)
                st.audio(st.session_state["mp3_audio"], format="audio/mp3")
                st.download_button("⬇️ MP3 다운로드", data=st.session_state["mp3_audio"],
                                   file_name="converted_voice.mp3", mime="audio/mpeg",
                                   use_container_width=True, key="dl_mp3")


# ════════════════════════════════════════════════════════════
# 탭 3: 생성 기록
# ════════════════════════════════════════════════════════════
with main_tab3:
    st.subheader("📝 생성 기록")
    history = load_history()

    if not history:
        st.info("아직 생성 기록이 없어요! 음성을 만들면 여기에 쌓여요 😊")
    else:
        if st.button("🗑️ 기록 전체 삭제"):
            _save(HIST_FILE, [])
            st.rerun()

        for i, h in enumerate(history):
            svc_label = {"elevenlabs": "🟢 ElevenLabs", "openai": "🔵 OpenAI", "gtts": "🟡 Google TTS"}.get(h["service"], h["service"])
            with st.expander(f"**{h['time']}** | {h['voice']} | {svc_label} — {h['text']}"):
                audio_bytes = bytes(h["audio"])

                # 다시 듣기 & 다운로드
                st.audio(audio_bytes, format="audio/mp3")
                col_dl, col_reuse = st.columns(2)
                col_dl.download_button("⬇️ 다운로드", data=audio_bytes,
                                       file_name=f"voice_{i+1}.mp3", mime="audio/mpeg",
                                       key=f"dl_hist_{i}")

                # 🔄 목소리 바꾸기 버튼
                if col_reuse.button("🔄 목소리 바꿔서 재생성", key=f"reuse_{i}", use_container_width=True):
                    st.session_state["reuse_text"] = h.get("full_text", h["text"])
                    st.info("✅ 텍스트를 가져왔어요! '✍️ 텍스트로 생성' 탭으로 이동해서 목소리를 바꿔보세요!")
