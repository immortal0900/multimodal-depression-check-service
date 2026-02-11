import streamlit as st
import requests
import io
import tempfile
import time
import os

st.set_page_config(page_title="AI Depression", layout="wide")

# ================= 세션 상태 =================
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "uploaded_audio" not in st.session_state:
    st.session_state.uploaded_audio = None
if "upload_time" not in st.session_state:
    st.session_state.upload_time = None
if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "realtime_analysis_done" not in st.session_state:
    st.session_state.realtime_analysis_done = False

# ================= CSS =================
st.markdown("""
<style>
.stApp { font-family: "Neue Helvetica", Helvetica, Arial, sans-serif; }
[data-testid="stSidebar"] { background-color: white; color: black; font-family: "Neue Helvetica", Helvetica, Arial, sans-serif; }
[data-testid="stSidebar"] h1 { text-align: center; }
[data-testid="stSidebar"] img { max-width: 100%; display: block; margin: 0 auto; }
.stButton > button { color: white !important; background-color: #4A90E2; height: 60px; margin-top: 12px; font-family: "Neue Helvetica", Helvetica, Arial, sans-serif; font-size: 16px; border-radius: 8px; }
.result-card { background-color: #f9f9f9; border: 2px solid #ddd; border-radius: 15px; padding: 20px; margin-top: 20px; font-family: "Neue Helvetica", Helvetica, Arial, sans-serif; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
.result-title { font-size: 20px; font-weight: bold; color: #333; margin-bottom: 10px; }
.result-item { font-size: 16px; color: #555; margin-bottom: 5px; }
</style>
""", unsafe_allow_html=True)

# ================= 사이드바 =================
with st.sidebar:
    st.title("AI Depression")
    st.markdown("---")
    st.write("📊 멀티모달 우울증 예측 시스템")
    st.write("이미지, 음성, 텍스트를 결합하여 우울증을 예측합니다")

# ================= 탭 생성 =================
tab1, tab2 = st.tabs(["📂 파일 업로드 & AI 분석", "🎤 실시간 분석"])

# ================= TAB 1 =================
with tab1:
    st.header("파일 업로드 & AI 분석")
    st.write("이미지 파일과 음성 파일을 업로드하면 자동으로 분석을 시작합니다.")
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])

    # 왼쪽: 이미지 업로드
    with col1:
        st.subheader("📷 이미지 업로드")
        new_uploaded_image = st.file_uploader(
            "얼굴 이미지를 선택하세요",
            type=["png", "jpg", "jpeg"],
            key="image_uploader_tab1"
        )
        
        if new_uploaded_image is not None:
            if st.session_state.uploaded_image != new_uploaded_image:
                st.session_state.uploaded_image = new_uploaded_image
                st.session_state.upload_time = time.time()
                st.session_state.analysis_done = False
            st.image(new_uploaded_image, caption="업로드된 이미지", use_container_width=True)

    # 오른쪽: 오디오 업로드
    with col2:
        st.subheader("🎤 음성 업로드")
        new_uploaded_audio = st.file_uploader(
            "음성 파일을 선택하세요",
            type=["wav"],
            key="audio_uploader_tab1"
        )
        
        if new_uploaded_audio is not None:
            if st.session_state.uploaded_audio != new_uploaded_audio:
                st.session_state.uploaded_audio = new_uploaded_audio
                st.session_state.upload_time = time.time()
                st.session_state.analysis_done = False
            st.audio(new_uploaded_audio, format="audio/wav")

    response_placeholder = st.empty()

    # 자동 분석 (이미지와 오디오가 모두 있을 때)
    if (st.session_state.uploaded_image and st.session_state.uploaded_audio 
        and not st.session_state.analysis_done):
        
        elapsed = time.time() - st.session_state.upload_time
        if elapsed < 2:
            response_placeholder.info("자동으로 AI 분석을 시작합니다...")
        else:
            response_placeholder.info("📡 AI 분석 서버로 데이터 전송 중...")
            
            try:
                FASTAPI_URL = "http://localhost:8999/predict"
                
                # 업로드된 파일을 직접 전송
                files = {
                    "image": (st.session_state.uploaded_image.name, 
                             st.session_state.uploaded_image.getvalue(), 
                             "image/png"),
                    "audio": (st.session_state.uploaded_audio.name, 
                             st.session_state.uploaded_audio.getvalue(), 
                             "audio/wav")
                }
                
                response = requests.post(FASTAPI_URL, files=files)
                
                if response.status_code == 200:
                    result_json = response.json()
                    # 결과 카드 출력
                    items_html = f"""
                    <div class="result-item"><b>최종 진단:</b> {result_json['final_prediction']}</div>
                    <div class="result-item"><b>우울증 가능성:</b> {result_json['depression_percentage']}%</div>
                    <div class="result-item"><b>위험도:</b> {result_json['risk_level']}</div>
                    <div class="result-item"><b>개별 결과:</b></div>
                    <ul>
                        <li>이미지: {result_json['individual_results']['image']['percentage']}%</li>
                        <li>음성: {result_json['individual_results']['sound']['percentage']}%</li>
                        <li>텍스트: {result_json['individual_results']['text']['percentage']}%</li>
                    </ul>
                    """
                    st.markdown(f"<div class='result-card'>{items_html}</div>", unsafe_allow_html=True)
                    st.session_state.analysis_done = True
                    response_placeholder.success("✅ AI 분석 완료")
                else:
                    response_placeholder.error(f"❌ 분석 실패: {response.status_code} - {response.text}")
                    
            except Exception as e:
                response_placeholder.error(f"❌ 서버 연결 실패: {e}")

# ================= TAB 2 =================
with tab2:
    st.header("실시간 분석")
    st.write("📷 사진을 촬영하고 🎤 음성을 녹음한 뒤, 'AI 분석 시작' 버튼을 누르세요.")
    st.markdown("---")
    col1, col2 = st.columns([3, 2])

    # 왼쪽: 카메라 + 음성 입력
    with col1:
        picture_widget = st.camera_input("카메라 촬영", label_visibility="visible")
        audio_widget = st.audio_input("음성 녹음", label_visibility="visible")

    # 오른쪽: 버튼 + 결과
    with col2:
        status_placeholder = st.empty()

        if st.button("🧠 AI 분석 시작", key="analyze_button"):
            if not picture_widget:
                st.warning("📷 사진을 먼저 촬영해주세요.")
            if not audio_widget:
                st.warning("🎤 음성을 먼저 녹음해주세요.")

            if picture_widget and audio_widget:
                temp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
                temp_img.write(picture_widget.getvalue())
                temp_img.flush()
                st.image(picture_widget, caption="촬영된 사진")

                temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_audio.write(audio_widget.getvalue())
                temp_audio.flush()
                st.audio(audio_widget, format="audio/wav")

                status_placeholder.info("📡 AI 분석 서버로 데이터 전송 중...")

                try:
                    FASTAPI_URL = "http://localhost:8999/predict"
                    files_to_send = {
                        "image": ("realtime_image.png", open(temp_img.name, 'rb'), "image/png"),
                        "audio": ("realtime_audio.wav", open(temp_audio.name, 'rb'), "audio/wav")
                    }
                    response = requests.post(FASTAPI_URL, files=files_to_send)
                    
                    if response.status_code == 200:
                        result_json = response.json()
                        items_html = f"""
                        <div class="result-item"><b>최종 진단:</b> {result_json['final_prediction']}</div>
                        <div class="result-item"><b>우울증 가능성:</b> {result_json['depression_percentage']}%</div>
                        <div class="result-item"><b>위험도:</b> {result_json['risk_level']}</div>
                        <div class="result-item"><b>개별 결과:</b></div>
                        <ul>
                            <li>이미지: {result_json['individual_results']['image']['percentage']}%</li>
                            <li>음성: {result_json['individual_results']['sound']['percentage']}%</li>
                            <li>텍스트: {result_json['individual_results']['text']['percentage']}%</li>
                        </ul>
                        """
                        st.markdown(f"<div class='result-card'>{items_html}</div>", unsafe_allow_html=True)
                        st.session_state.realtime_analysis_done = True
                        status_placeholder.success("🧠 AI 실시간 분석 완료")
                    else:
                        status_placeholder.error(f"❌ 분석 실패: {response.status_code} - {response.text}")
                        
                except Exception as e:
                    status_placeholder.error(f"❌ 요청 중 오류 발생: {e}")
                finally:
                    # 임시 파일 정리
                    try:
                        if os.path.exists(temp_img.name):
                            os.unlink(temp_img.name)
                        if os.path.exists(temp_audio.name):
                            os.unlink(temp_audio.name)
                    except Exception:
                        pass
