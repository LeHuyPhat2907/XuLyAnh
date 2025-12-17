import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import os
import pickle

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

MEASURE_MODE = "face"


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

def calculate_distance_with_overlay(frame, mtx, dist):
    undistorted = undistort_image(frame, mtx, dist)
    gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)

    distance = None

    # ================= MODE: FACE =================
    if MEASURE_MODE == "face":
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80)
        )

        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
            W_pixel_detected = w
            distance = estimate_size_based_distance(W_pixel_detected)

            cv2.rectangle(undistorted, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(undistorted, "MODE: FACE",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # ================= MODE: OBJECT =================
    elif MEASURE_MODE == "object":
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            c = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)

            if w > 50 and h > 50:
                W_pixel_detected = w
                distance = estimate_size_based_distance(W_pixel_detected)

                cv2.rectangle(undistorted, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(undistorted, "MODE: OBJECT",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    # ================= HIỂN THỊ =================
    if distance:
        cv2.putText(
            undistorted,
            f"Distance: {distance:.2f} cm",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

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

        processed_frame, distance = calculate_distance_with_overlay(frame, mtx, dist)

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
        messagebox.showerror("Lỗi", "Chưa có file calibration_data.pkl!")
        return

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
        messagebox.showerror("Lỗi", "Không đọc được ảnh.")
        return

    undistorted = undistort_image(img, mtx, dist)

    # === CHỌN ROI ===
    cv2.imshow("Chon vat can do", undistorted)
    x, y, w, h = cv2.selectROI(
        "Chon vat can do", undistorted, fromCenter=False, showCrosshair=True
    )
    cv2.destroyWindow("Chon vat can do")

    if w == 0 or h == 0:
        messagebox.showinfo("Thông báo", "Bạn chưa chọn vùng đo.")
        return

    # === TÍNH KHOẢNG CÁCH ===
    distance = estimate_size_based_distance(w)

    cv2.rectangle(undistorted, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(
        undistorted,
        f"Distance: {distance:.2f} cm",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Distance Estimation - Image", undistorted)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==================== MAIN WINDOW ====================




root = tk.Tk()

mode_var = tk.StringVar(value="face")

def set_mode():
    global MEASURE_MODE
    MEASURE_MODE = mode_var.get()
    print("Current mode:", MEASURE_MODE)
root.title("App Đo Khoảng Cách - Tkinter")
root.geometry("420x320")

title_label = tk.Label(root, text="CHỌN CHẾ ĐỘ ĐO", font=("Arial", 14, "bold"))
title_label.pack(pady=20)

tk.Radiobutton(
    root,
    text="👤 Đo khuôn mặt",
    variable=mode_var,
    value="face",
    command=set_mode,
    font=("Arial", 11)
).pack(anchor="w", padx=60)

tk.Radiobutton(
    root,
    text="📦 Đo vật thể",
    variable=mode_var,
    value="object",
    command=set_mode,
    font=("Arial", 11)
).pack(anchor="w", padx=60)


btn_cam = tk.Button(root, text="📹 Mở Camera (Realtime)", command=open_camera,
                    width=28, height=2, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
btn_cam.pack(pady=10)

btn_img = tk.Button(root, text="📁 Chọn Ảnh Từ Máy Tính", command=open_image,
                    width=28, height=2, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
btn_img.pack(pady=10)

footer = tk.Label(root, text="Thành viên 3 – App + Integration", font=("Arial", 10))
footer.pack(pady=20)

root.mainloop()