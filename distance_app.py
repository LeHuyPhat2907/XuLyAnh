import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import pickle

# ==================== THAM SỐ VÀ HẰNG SỐ ====================
W_REAL = 5.398  # Chiều rộng thực tế (cm) - thẻ tín dụng
D_REF = 50.0    # Khoảng cách tham chiếu (cm)
W_PIXEL_REF = 160  # Kích thước pixel tham chiếu
K_REF = (W_PIXEL_REF * D_REF) / W_REAL  # Focal length constant

# ==================== HÀM TẢI THAM SỐ ====================

def load_camera_params():
    """Tải tham số camera từ file calibration_data.pkl"""
    calib_path = os.path.join(os.path.dirname(__file__), "step1_calibrate", "calibration_data.pkl")
    if not os.path.exists(calib_path):
        if os.path.exists("calibration_data.pkl"):
            calib_path = "calibration_data.pkl"
        else:
            return None, None

    try:
        with open(calib_path, "rb") as f:
            data = pickle.load(f)
        return data.get("camera_matrix"), data.get("dist_coeff")
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không tải được tham số: {e}")
        return None, None

# ==================== HÀM XỬ LÝ TÍCH HỢP ====================

def undistort_image(frame, mtx, dist, crop=True):
    """Khử méo với cơ chế bảo vệ nếu tham số calibration bị sai"""
    if mtx is None or dist is None:
        return frame

    h, w = frame.shape[:2]

    # Thay đổi alpha=1 để KHÔNG cắt bỏ bất kỳ pixel nào, giúp bạn nhìn thấy toàn cảnh bị lỗi
    new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)

    # Chỉ cắt ảnh nếu thông số ROI hợp lệ và không làm mất quá nhiều ảnh
    if crop:
        x, y, w_roi, h_roi = roi
        # Nếu diện tích vùng hợp lệ quá nhỏ (dưới 30% ảnh gốc), nghĩa là calibration sai
        if w_roi * h_roi > (w * h * 0.3):
            undistorted = undistorted[y:y + h_roi, x:x + w_roi]
        else:
            # Nếu calibration sai quá nặng, trả về ảnh gốc kèm cảnh báo
            cv2.putText(undistorted, "CANH BAO: FILE CALIBRATION LOI!", (10, h-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    return undistorted

def estimate_size_based_distance(W_pixel_detected, W_real=W_REAL, K_ref=K_REF):
    if W_pixel_detected <= 0:
        return None
    return (W_real * K_ref) / W_pixel_detected

def calculate_distance_with_overlay(frame, mtx, dist, W_pixel_detected=85):
    undistorted = undistort_image(frame, mtx, dist, crop=True)
    distance = estimate_size_based_distance(W_pixel_detected)

    if distance is not None:
        cv2.putText(undistorted, f"Khoang cach: {distance:.2f} cm", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(undistorted, f"Pixel: {W_pixel_detected}px", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        h, w = undistorted.shape[:2]
        center_x, center_y = w // 2, h // 2
        box_size = W_pixel_detected
        cv2.rectangle(undistorted,
                      (int(center_x - box_size / 2), int(center_y - box_size / 2)),
                      (int(center_x + box_size / 2), int(center_y + box_size / 2)),
                      (0, 255, 0), 2)

    return undistorted, distance

# ==================== GIAO DIỆN TKINTER ====================

def open_camera():
    mtx, dist = load_camera_params()
    if mtx is None:
        messagebox.showerror("Lỗi", "Chưa có file calibration_data.pkl! Chạy step1_calibrate trước.")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Lỗi", "Không mở được camera.")
        return

    messagebox.showinfo("Camera", "Nhấn ESC trong cửa sổ video để thoát.")

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Lỗi", "Không đọc được frame từ camera.")
            break

        processed_frame, distance = calculate_distance_with_overlay(frame, mtx, dist, W_pixel_detected=85)

        cv2.imshow('Distance Estimation - Live', processed_frame)
        if distance:
            print(f"Khoang cach: {distance:.2f} cm")  # Ghi ra console để tránh crash do displayStatusBar

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC để thoát
            break

    cap.release()
    cv2.destroyAllWindows()

def open_image():
    mtx, dist = load_camera_params()
    if mtx is None:
        messagebox.showerror("Lỗi", "Chưa có file calibration_data.pkl! Chạy step1_calibrate trước.")
        return

    # Tránh crash trên macOS: dùng tuple patterns thay vì chuỗi có dấu chấm phẩy
    file_path = filedialog.askopenfilename(
        title="Chọn ảnh",
        filetypes=[
            ("Image Files", ("*.jpg", "*.jpeg", "*.png", "*.bmp")),
            ("Tất cả", "*.*")
        ]
    )
    if not file_path:
        return

    img = cv2.imread(file_path)
    if img is None:
        messagebox.showerror("Lỗi", "Không đọc được ảnh đã chọn.")
        return

    processed_img, distance = calculate_distance_with_overlay(img, mtx, dist, W_pixel_detected=85)
    cv2.imshow('Distance Estimation - Image', processed_img)
    if distance:
        print(f"Khoang cach: {distance:.2f} cm")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==================== MAIN WINDOW ====================

root = tk.Tk()
root.title("App Đo Khoảng Cách - Tkinter")
root.geometry("420x320")

title_label = tk.Label(root, text="CHỌN CHẾ ĐỘ ĐO", font=("Arial", 14, "bold"))
title_label.pack(pady=20)

btn_cam = tk.Button(root, text="📹 Mở Camera (Realtime)", command=open_camera,
                    width=28, height=2, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
btn_cam.pack(pady=10)

btn_img = tk.Button(root, text="📁 Chọn Ảnh Từ Máy Tính", command=open_image,
                    width=28, height=2, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
btn_img.pack(pady=10)

footer = tk.Label(root, text="Thành viên 3 – App + Integration", font=("Arial", 10))
footer.pack(pady=20)

root.mainloop()