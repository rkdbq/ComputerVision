import cv2, pygame, time, threading
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton, QFileDialog, QSlider
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from preprocess import *

class VideoPlayer(QMainWindow):
    def __init__(self):
        super().__init__()

        # 윈도우 설정
        self.setWindowTitle("OMG Trainer")
        self.setGeometry(200, 200, 1400, 600)

        self.og_title_label = QLabel()
        self.webcam_title_label = QLabel()

        # 점수
        self.score_label = QLabel()
        self.score = 0
        self.total_score = np.zeros(0)

        # 비디오 레이블 초기화
        self.video_label = QLabel()
        self.webcam_label = QLabel()

        self.webcam = cv2.VideoCapture(0)
        self.webcam_timer = QTimer()
        self.webcam_timer.timeout.connect(self.update_webcam_frame)
        self.webcam_timer.start(30)
        self.webcam_timestamp = 0
        self.webcam_landmarks = []

        self.og_timestamp = 0

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
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.update_speed)

        og_title_layout = QHBoxLayout()
        og_title_layout.addWidget(self.og_title_label)
        og_title_layout.setAlignment(Qt.AlignHCenter)

        webcam_title_layout = QHBoxLayout()
        webcam_title_layout.addWidget(self.webcam_title_label)
        webcam_title_layout.setAlignment(Qt.AlignHCenter)

        title_layout = QHBoxLayout()
        title_layout.addLayout(og_title_layout)
        title_layout.addLayout(webcam_title_layout)


        # 레이아웃 초기화
        score_layout = QVBoxLayout()
        score_layout.addWidget(self.score_label)
        score_layout.setAlignment(Qt.AlignHCenter)  # 좌우 가운데 정렬
    
        
        og_control_layout = QVBoxLayout()
        og_control_layout.addWidget(self.select_button)
        og_control_layout.addWidget(self.speed_slider)
        og_control_layout.setContentsMargins(0, 0, 700, 0) 

        og_layout = QVBoxLayout()
        og_layout.addWidget(self.video_label)
        og_layout.setAlignment(Qt.AlignHCenter)

        webcam_layout = QVBoxLayout()
        webcam_layout.addWidget(self.webcam_label)
        webcam_layout.setAlignment(Qt.AlignHCenter)

        video_layout = QHBoxLayout()
        video_layout.addLayout(og_layout)
        video_layout.addLayout(webcam_layout)
        # video_layout.setAlignment(Qt.AlignHCenter)
        

        layout = QVBoxLayout()
        layout.addLayout(score_layout)
        layout.addLayout(title_layout)
        layout.addLayout(video_layout)
        layout.addLayout(og_control_layout)

        # 위젯 설정
        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        # 선택한 비디오 파일 경로
        self.video_path = ""
        self.audio_path = ""

        # 재생 속도
        self.playback_speed = 1.0

        self.update_score_label("댄스 동영상을 선택하고, 얼마나 비슷하게 출 수 있는지 점수를 측정해보세요.")
        self.og_title_label.setText("도전할 동영상")
        self.webcam_title_label.setText("ME")

    def restart_video(self):
        self.score = 0
        self.total_score = np.zeros(0)

    def select_video(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            self.restart_video()
            selected_files = file_dialog.selectedFiles()
            self.video_path = selected_files[0]
            self.audio_path = change_extension(self.video_path, ".mp3")
            convert_to_mp3(self.video_path, self.audio_path)
            
            # 선택한 비디오 파일 열기
            self.cap = cv2.VideoCapture(self.video_path)

            # 비디오 재생 시작
            self.timer.start(30)

    def update_speed(self):
        # 재생 속도 슬라이더 값에 따라 재생 속도 설정
        value = self.speed_slider.value()
        self.playback_speed = value / 5.0

    def update_webcam_frame(self):
        # 비디오 프레임 읽기
        self.webcam_timestamp += 1
        ret, cam_frame = self.webcam.read()

        # 프레임 읽기에 실패하면 종료
        if not ret:
            self.webcam_timer.stop()
            return
        
        # 프레임을 QImage로 변환
        cam_frame = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
        cam_frame = cv2.flip(cam_frame, 1)
        landmarked_frame, self.webcam_landmarks = webcam_frame_with_landmark(cam_frame, self.webcam_timestamp)
        
        height, width, channels = landmarked_frame.shape
        q_image = QImage(
            landmarked_frame.data, width, height, width * channels, QImage.Format_RGB888
        )

        # QImage를 QPixmap으로 변환하여 비디오 레이블에 표시
        pixmap = QPixmap.fromImage(q_image)
        self.webcam_label.setPixmap(pixmap.scaledToWidth(600, Qt.SmoothTransformation))

    def update_frame(self):
        # 비디오 프레임 읽기
        # timestamp = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        self.og_timestamp += 1
        ret, og_frame = self.cap.read()

        # 프레임 읽기에 실패하면 종료
        if not ret:
            self.timer.stop()
            self.update_score_label(f"Total Score: {round(np.mean(self.total_score), 1)}")
            return

        # 프레임을 QImage로 변환
        og_frame = cv2.cvtColor(og_frame, cv2.COLOR_BGR2RGB)
        landmarked_frame, og_landmarks = frame_with_landmark(og_frame, self.og_timestamp)
        self.score = (1 - calculate_landmark_loss(og_landmarks, self.webcam_landmarks)) * 100
        self.total_score = np.concatenate((self.total_score, np.array([self.score])), axis=0)
        self.update_score_label()
        # print(self.score)

        height, width, channels = landmarked_frame.shape
        q_image = QImage(
            landmarked_frame.data, width, height, width * channels, QImage.Format_RGB888
        )

        # QImage를 QPixmap으로 변환하여 비디오 레이블에 표시
        pixmap = QPixmap.fromImage(q_image)
        self.video_label.setPixmap(pixmap.scaledToWidth(600, Qt.SmoothTransformation))

        # 재생 속도 적용
        delay = int((1000 / (self.cap.get(cv2.CAP_PROP_FPS) * self.playback_speed)))
        if self.playback_speed == 2.0:
            delay = 0
        # print(self.playback_speed)
        # print(delay)

        # 딜레이 후 타이머 재실행
        self.timer.start(delay)

    def update_score_label(self, text=""):
        # score_label에 현재 score 값을 표시
        bad = "font-size: 20px; color: red;"
        good = "font-size: 20px; color: orange;"
        excellent = "font-size: 20px; color: green;"

        if text != "":
            self.score_label.setStyleSheet("font-size: 20px;")
            self.score_label.setText(text)
            return

        if self.score < 60:
            self.score_label.setStyleSheet(bad)
            self.score_label.setText(f"Try Hard ({round(self.score, 1)}% simular)") 
        elif self.score < 85:
            self.score_label.setStyleSheet(good)
            self.score_label.setText(f"Keep Going ({round(self.score, 1)}% simular)")
        else:
            self.score_label.setStyleSheet(excellent)
            self.score_label.setText(f"Perfect ({round(self.score, 1)}% simular)")   


if __name__ == "__main__":
    app = QApplication([])
    player = VideoPlayer()
    player.show()
    app.exec_()
