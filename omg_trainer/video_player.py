import mediapipe as mp
import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QPushButton, QFileDialog, QSlider
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import vision

def draw_landmarks_on_image(rgb_image, detection_result):
  if detection_result is None:
    return rgb_image
  
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        # BaseOptions = mp.tasks.BaseOptions
        # PoseLandmarker = mp.tasks.vision.PoseLandmarker
        # PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        # # PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
        # VisionRunningMode = mp.tasks.vision.RunningMode

        #
        self.og_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path='pose_landmarker_lite.task'),
            running_mode=VisionRunningMode.VIDEO)
        self.og_landmarker = PoseLandmarker.create_from_options(self.og_options)

        # 윈도우 설정
        self.setWindowTitle("Video Player")
        self.setGeometry(200, 200, 800, 600)

        # 비디오 레이블 초기화
        self.video_label = QLabel()

        # 비디오 재생을 위한 타이머 설정
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # 비디오 파일 선택 버튼 초기화
        self.select_button = QPushButton("Select Video")
        self.select_button.clicked.connect(self.select_video)

        # 재생 속도 슬라이더 초기화
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(10)
        self.speed_slider.setValue(5)
        self.speed_slider.valueChanged.connect(self.update_speed)

        # 레이아웃 초기화
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addWidget(self.select_button)
        layout.addWidget(self.speed_slider)

        # 위젯 설정
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # 선택한 비디오 파일 경로
        self.video_path = ""

        # 재생 속도
        self.playback_speed = 1.0

    def select_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            selected_files = file_dialog.selectedFiles()
            self.video_path = selected_files[0]

            # 선택한 비디오 파일 열기
            self.cap = cv2.VideoCapture(self.video_path)

            # 비디오 재생 시작
            self.timer.start(30)

    def update_speed(self):
        # 재생 속도 슬라이더 값에 따라 재생 속도 설정
        value = self.speed_slider.value()
        self.playback_speed = value / 5.0

    def update_frame(self):
        # 비디오 프레임 읽기
        timestamp = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, og_frame = self.cap.read()

        # 프레임 읽기에 실패하면 종료
        if not ret:
            self.timer.stop()
            return

        # 프레임을 QImage로 변환
        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        og_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=og_frame)
        og_landmarker_result = self.og_landmarker.detect_for_video(og_image, timestamp)
        frame_rgb = draw_landmarks_on_image(og_image.numpy_view(), og_landmarker_result)

        height, width, channels = frame_rgb.shape
        q_image = QImage(
            frame_rgb.data, width, height, width * channels, QImage.Format_RGB888
        )

        # QImage를 QPixmap으로 변환하여 비디오 레이블에 표시
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaledToWidth(800, Qt.SmoothTransformation))

        # 재생 속도 적용
        delay = int(1000 / (self.cap.get(cv2.CAP_PROP_FPS) * self.playback_speed))

        # 딜레이 후 타이머 재실행
        self.timer.start(delay)


if __name__ == "__main__":
    app = QApplication([])
    player = VideoPlayer()
    player.show()
    app.exec_()
