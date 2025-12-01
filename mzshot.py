import time
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration,
    WebRtcMode,
)
import av

mp_face = mp.solutions.face_detection


def get_face_roll_angle(img_bgr):
    """BGR 이미지에서 첫 번째 얼굴의 roll angle(기울기) 계산."""
    h, w, _ = img_bgr.shape

    with mp_face.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.6
    ) as face_detector:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = face_detector.process(img_rgb)

        if not results.detections:
            return None, None

        detection = results.detections[0]
        keypoints = detection.location_data.relative_keypoints

        right_eye = keypoints[0]
        left_eye = keypoints[1]

        x1, y1 = right_eye.x * w, right_eye.y * h
        x2, y2 = left_eye.x * w, left_eye.y * h

        dx = x2 - x1
        dy = y2 - y1

        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)

        right_eye_pt = (int(x1), int(y1))
        left_eye_pt = (int(x2), int(y2))

        return angle_deg, (right_eye_pt, left_eye_pt)


def draw_angle_overlay(img_bgr, angle_deg, eye_pts, label=""):
    img = img_bgr.copy()
    h, w, _ = img.shape

    if angle_deg is not None and eye_pts is not None:
        (re, le) = eye_pts
        cv2.line(img, re, le, (0, 255, 0), 2)
        text = f"{label} roll: {angle_deg:.1f} deg"
    else:
        text = f"{label} No face"

    cv2.putText(
        img,
        text,
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return img


class PoseMatchProcessor(VideoProcessorBase):
    """웹캠 프레임을 받아서 타겟 각도와 비교 + 자동 캡처 + 유사도 바 표시"""

    def __init__(self):
        self.ref_angle = None
        self.tolerance = 5.0
        self.cooldown_sec = 3.0
        self.last_capture_time = 0.0

        self.current_angle = None
        self.similarity = None
        self.captured_images = []

        self.face_detector = mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.6
        )

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(img_rgb)

        text = "No face"
        sim_for_bar = 0.0

        if results and results.detections:
            detection = results.detections[0]
            keypoints = detection.location_data.relative_keypoints
            right_eye = keypoints[0]
            left_eye = keypoints[1]

            x1, y1 = right_eye.x * w, right_eye.y * h
            x2, y2 = left_eye.x * w, left_eye.y * h
            dx, dy = x2 - x1, y2 - y1

            angle_rad = np.arctan2(dy, dx)
            angle_deg = np.degrees(angle_rad)
            self.current_angle = angle_deg

            right_eye_pt = (int(x1), int(y1))
            left_eye_pt = (int(x2), int(y2))

            cv2.line(img, right_eye_pt, left_eye_pt, (0, 255, 0), 2)

            text = f"current: {angle_deg:.1f} deg"

            if self.ref_angle is not None:
                diff = abs(angle_deg - self.ref_angle)

                if diff >= self.tolerance:
                    sim = 0.0
                else:
                    sim = max(0.0, 100.0 * (1.0 - diff / self.tolerance))

                self.similarity = sim
                sim_for_bar = sim

                text += f" | diff: {diff:.1f} deg | sim: {sim:.0f}%"

                # 조건 만족하면 자동 캡처
                now = time.time()
                if sim >= 90.0 and now - self.last_capture_time > self.cooldown_sec:
                    self.last_capture_time = now
                    self.captured_images.append(img.copy())
                    if len(self.captured_images) > 10:
                        self.captured_images.pop(0)

                    cv2.putText(
                        img,
                        "CAPTURED!",
                        (30, h - 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 0, 255),
                        3,
                        cv2.LINE_AA,
                    )
        else:
            self.current_angle = None
            self.similarity = None

        # 상단 텍스트
        cv2.putText(
            img,
            text,
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # ----- 유사도 바(프로그레스바) 오버레이 -----
        bar_x1, bar_y1 = 30, h - 30
        bar_x2, bar_y2 = w - 30, h - 10

        # 바 배경 (회색)
        cv2.rectangle(img, (bar_x1, bar_y1), (bar_x2, bar_y2), (80, 80, 80), 1)

        if sim_for_bar > 0:
            ratio = max(0.0, min(1.0, sim_for_bar / 100.0))
            fill_x2 = bar_x1 + int((bar_x2 - bar_x1) * ratio)
            # 채워진 부분 (초록)
            cv2.rectangle(img, (bar_x1, bar_y1), (fill_x2, bar_y2), (0, 200, 0), -1)

        # 바 중앙에 텍스트
        sim_text = f"Similarity: {sim_for_bar:.0f}%"
        text_size, _ = cv2.getTextSize(sim_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        tx = bar_x1 + (bar_x2 - bar_x1 - text_size[0]) // 2
        ty = bar_y1 - 5
        cv2.putText(
            img,
            sim_text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return av.VideoFrame.from_ndarray(img, format="bgr24")


def main():
    st.title("타겟 포즈 유사도 기반 자동 촬영")

    st.markdown(
        """
        1. **타겟 사진**을 업로드하면 얼굴 기울기(roll angle)를 분석합니다.  
        2. 웹캠을 켜면 실시간으로 현재 각도와 비교해서,  
           **각도 차이가 설정값 이하**가 되면 자동으로 사진을 캡처합니다.  
        3. 유사도(각도 기준)는 웹캠 화면 아래의 **녹색 바**로 표시됩니다.
        """
    )

    # --- 타겟 사진 업로드 ---
    st.sidebar.header("① 타겟 사진 업로드")
    ref_file = st.sidebar.file_uploader(
        "타겟 포즈 사진 (jpg, png)",
        type=["jpg", "jpeg", "png"],
        key="ref_upload",
    )

    ref_angle = None
    ref_disp = None

    if ref_file is not None:
        data = ref_file.read()
        arr = np.frombuffer(data, np.uint8)
        ref_img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if ref_img is None:
            st.sidebar.error("타겟 사진을 읽지 못했습니다.")
        else:
            ref_angle, eye_pts = get_face_roll_angle(ref_img)
            if ref_angle is None:
                st.sidebar.error("타겟 사진에서 얼굴을 찾지 못했습니다.")
            else:
                st.sidebar.success(f"타겟 얼굴 각도: {ref_angle:.1f}°")
                ref_disp = draw_angle_overlay(ref_img, ref_angle, eye_pts, label="target")

    # --- 촬영 조건 ---
    st.sidebar.header("② 촬영 조건")
    tolerance = st.sidebar.slider(
        "허용 각도 차 (deg)",
        min_value=2.0,
        max_value=30.0,
        value=8.0,
        step=1.0,
    )
    cooldown_sec = st.sidebar.slider(
        "촬영 간 최소 간격 (초)",
        min_value=0.0,
        max_value=10.0,
        value=3.0,
        step=1.0,
    )

    if ref_disp is not None:
        st.subheader("타겟 사진 (각도 표시)")
        st.image(ref_disp, channels="BGR")

    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

    st.subheader("웹캠")

    webrtc_ctx = webrtc_streamer(
        key="pose-match-capture",
        mode=WebRtcMode.SENDRECV,   # ★ SENDRECV로 변경
        rtc_configuration=rtc_config,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=PoseMatchProcessor,
        async_processing=True,
    )

    if webrtc_ctx.video_processor:
        vp: PoseMatchProcessor = webrtc_ctx.video_processor

        vp.ref_angle = ref_angle
        vp.tolerance = tolerance
        vp.cooldown_sec = cooldown_sec

        st.subheader("현재 상태")

        if ref_angle is None:
            st.warning("타겟 사진을 먼저 업로드해야 유사도 계산이 가능합니다.")
        else:
            if vp.current_angle is not None:
                st.write(f"현재 각도: **{vp.current_angle:.1f}°**")
            else:
                st.write("현재 얼굴을 찾는 중입니다.")

            if vp.similarity is not None:
                st.write(f"유사도(각도 기준): **{vp.similarity:.0f}%**")
            else:
                st.write("유사도 계산 대기 중...")

        st.subheader("자동 촬영된 사진들")
        if vp.captured_images:
            for idx, img in enumerate(reversed(vp.captured_images), start=1):
                st.image(img, channels="BGR", caption=f"캡처 #{idx}")
        else:
            st.write("아직 캡처된 사진이 없습니다. 타겟 각도와 비슷하게 맞춰보세요.")


if __name__ == "__main__":
    main()