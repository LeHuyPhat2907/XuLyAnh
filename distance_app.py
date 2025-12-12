import tkinter as tk
import cv2
import numpy as np
import os

root = tk.Tk()

root.title("Ứng dụng đo khoảng cách!")
root.geometry("400x300")



def cammera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        cv2.imshow('Webcam', frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

def display_results(frame, distance):
    text = f"Khoang cach: {distance:.2f} cm"

    org = (50, 50)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    color = (0, 255, 0)
    thickness = 2

    cv2.putText(frame, text, org, font, font_scale, color, thickness, cv2.LINE_AA)

    return frame
def load_camera_params(matrix_path, dist_path):
    if not os.path.exists(matrix_path) or not os.path.exists(dist_path):
        print("Lỗi: Không tìm thấy file tham số camera. Đảm bảo Thành viên 1 đã chạy xong.")
        # Trả về các ma trận identity nếu không tìm thấy để tránh lỗi crash
        return np.eye(3), np.zeros((1, 5))

    # Tải ma trận camera (3x3)
    camera_matrix = np.load(matrix_path)
    # Tải hệ số méo hình (1x5)
    dist_coeffs = np.load(dist_path)

    return camera_matrix, dist_coeffs
camera_button = tk.Button(
    root,
    text="Mở Camera",
    command=cammera,
    font=("Arial", 14),
    bg="#4CAF50",
    fg="black",
)

camera_button.pack(pady=50)
root.mainloop()