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
CROP_PADDING_RATIO = 1.0  # Tỷ lệ padding xung quanh vùng chọn (100% = tăng kích thước 2x)

# ==================== THAM SỐ VÀ HẰNG SỐ ====================
# Tham số cho vật thể (thẻ tín dụng)
W_REAL_OBJECT = 5.398  # Chiều rộng thực tế (cm) - thẻ tín dụng
D_REF_OBJECT = 50.0    # Khoảng cách tham chiếu (cm)
W_PIXEL_REF_OBJECT = 160  # Kích thước pixel tham chiếu
K_REF_OBJECT = (W_PIXEL_REF_OBJECT * D_REF_OBJECT) / W_REAL_OBJECT

# Tham số cho khuôn mặt (chính xác hơn)
W_REAL_FACE = 15.0  # Chiều rộng thực tế khuôn mặt trung bình (cm)
D_REF_FACE = 50.0   # Khoảng cách tham chiếu (cm)
W_PIXEL_REF_FACE = 200  # Kích thước pixel tham chiếu cho face
K_REF_FACE = (W_PIXEL_REF_FACE * D_REF_FACE) / W_REAL_FACE

# Tham số cho Homography (đo trên mặt phẳng)
# Tọa độ thực tế của tờ giấy A4 (cm)
P_WORLD_REAL = np.float32([
    [0, 0],
    [21.0, 0],    # Chiều rộng A4 = 21 cm
    [21.0, 29.7], # Chiều dài A4 = 29.7 cm
    [0, 29.7]
])

# ==================== HÀM TẢI THAM SỐ ====================

def load_camera_params():
    """Tải tham số camera từ file calibration_data.pkl"""
    calib_path = os.path.join(os.path.dirname(__file__), "step1_calibrate", "calibration_data.pkl")
    if not os.path.exists(calib_path):
        if os.path.exists("calibration_data.pkl"):
            calib_path = "calibration_data.pkl"
        else:
            print("Warning: No calibration file found. Will use original image without distortion correction.")
            return None, None

    try:
        with open(calib_path, "rb") as f:
            data = pickle.load(f)
        mtx = data.get("camera_matrix")
        dist = data.get("dist_coeff")
        if mtx is not None and dist is not None:
            print("Calibration loaded successfully.")
        return mtx, dist
    except Exception as e:
        print(f"Error loading calibration: {e}. Will use original image.")
        return None, None
# ==================== HÀM XỬ LÝ TÍCH HỢP ====================
def undistort_image(frame, mtx, dist, crop=False):
    """Khử méo ảnh - giữ nguyên toàn bộ ảnh, không crop"""
    if mtx is None or dist is None:
        return frame

    # Kiểm tra tham số calibration có hợp lệ không
    try:
        h, w = frame.shape[:2]

        # Kiểm tra kích thước ma trận camera
        if mtx.shape != (3, 3):
            print("Warning: Invalid camera matrix size. Using original image.")
            return frame

        # QUAN TRỌNG: Sử dụng alpha=1 để giữ TOÀN BỘ ảnh gốc, không crop
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        undistorted = cv2.undistort(frame, mtx, dist, None, new_mtx)

        # KHÔNG crop để giữ nguyên kích thước ảnh
        return undistorted
    except Exception as e:
        print(f"Error in undistort: {e}. Using original image.")
        return frame

def estimate_size_based_distance(W_pixel_detected, mode="object"):
    """Tính khoảng cách dựa trên kích thước pixel phát hiện được"""
    if W_pixel_detected <= 0:
        return None

    # Chọn tham số phù hợp với mode
    if mode == "face":
        W_real = W_REAL_FACE
        K_ref = K_REF_FACE
    else:
        W_real = W_REAL_OBJECT
        K_ref = K_REF_OBJECT

    distance = (W_real * K_ref) / W_pixel_detected
    return distance

def measure_homography_distance(img, ref_points_img, P_world_real):
    """Đo khoảng cách trên mặt phẳng bằng Homography"""
    try:
        # Tìm ma trận Homography
        H, mask = cv2.findHomography(ref_points_img, P_world_real, cv2.RANSAC, 5.0)

        if H is None:
            return None, None

        return H, mask
    except Exception as e:
        print(f"Error in homography: {e}")
        return None, None

def crop_with_padding(image, x, y, w, h, padding_ratio=CROP_PADDING_RATIO):
    """Cắt ảnh quanh vùng chọn với padding, giữ nguyên tỷ lệ"""
    img_h, img_w = image.shape[:2]

    # Tính kích thước padding
    pad_x = int(w * padding_ratio / 2)
    pad_y = int(h * padding_ratio / 2)

    # Tính tọa độ crop
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(img_w, x + w + pad_x)
    y2 = min(img_h, y + h + pad_y)

    # Cắt ảnh
    cropped = image[y1:y2, x1:x2].copy()

    # Lưu thông tin offset để ánh xạ lại tọa độ
    offset = (x1, y1)

    return cropped, offset, (x1, y1, x2, y2)

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
            distance = estimate_size_based_distance(W_pixel_detected, mode="face")

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
                distance = estimate_size_based_distance(W_pixel_detected, mode="object")

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

        frame = cv2.flip(frame, 1)

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
    # Không bắt buộc phải có calibration - có thể dùng ảnh gốc

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

    # Kiểm tra và áp dụng undistortion (nếu có calibration hợp lệ)
    if mtx is not None and dist is not None:
        working_img = undistort_image(img, mtx, dist)
        # Kiểm tra kết quả undistort có hợp lệ không
        if working_img is None or working_img.size == 0 or working_img.shape[0] == 0 or working_img.shape[1] == 0:
            print("Warning: Undistortion failed. Using original image.")
            working_img = img.copy()
    else:
        working_img = img.copy()
        print("Using original image (no calibration)")

    # Kiểm tra lần cuối trước khi hiển thị
    if working_img.shape[0] <= 0 or working_img.shape[1] <= 0:
        messagebox.showerror("Lỗi", "Ảnh không hợp lệ sau khi xử lý.")
        return

    # === BƯỚC 1: HIỂN THỊ ẢNH VÀ CHO PHÉP CHỌN VÙNG ===
    cv2.imshow("Chon vung vat the - Nhan Enter/Space de xac nhan", working_img)
    messagebox.showinfo("Hướng dẫn", "Hãy kéo chọn vùng chứa vật thể cần đo.\nNhấn Enter/Space để xác nhận hoặc ESC để hủy.")

    x, y, w, h = cv2.selectROI(
        "Chon vung vat the - Nhan Enter/Space de xac nhan",
        working_img,
        fromCenter=False,
        showCrosshair=True
    )
    cv2.destroyWindow("Chon vung vat the - Nhan Enter/Space de xac nhan")

    if w == 0 or h == 0:
        messagebox.showinfo("Thông báo", "Bạn chưa chọn vùng đo.")
        return

    # === BƯỚC 2: CẮT ẢNH QUANH VÙNG CHỌN VỚI PADDING ===
    cropped_img, offset, crop_bounds = crop_with_padding(working_img, x, y, w, h, CROP_PADDING_RATIO)
    offset_x, offset_y = offset

    # === BƯỚC 3: SỬ DỤNG VÙNG ĐÃ CHỌN ĐỂ TÍNH KHOẢNG CÁCH ===
    # Vì người dùng đã chọn chính xác vật thể, ta sử dụng trực tiếp kích thước vùng chọn
    W_pixel_detected = w  # Chiều rộng vùng chọn
    distance = estimate_size_based_distance(W_pixel_detected, mode=MEASURE_MODE)

    # Vùng phát hiện chính là vùng người dùng chọn
    detection_rect = (x, y, w, h)

    # === HIỂN THỊ ẢNH CẮT ĐỂ THAM KHẢO (không dùng để detect) ===
    # cv2.imshow("Cropped Region", cropped_img)

    # === BƯỚC 4: VẼ KẾT QUẢ LÊN CÙNG ẢNH ĐÃ CHỌN (KHÔNG BỊ LỆCH) ===
    result_img = working_img.copy()

    # Vẽ vùng chọn ban đầu (màu xanh lá)
    cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(
        result_img,
        "Selected & Detected",
        (x, max(10, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Vẽ khoảng cách (nếu tính được)
    if distance is not None:
        cv2.putText(
            result_img,
            f"Distance: {distance:.2f} cm",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )
        print(f"Khoang cach: {distance:.2f} cm")
    else:
        cv2.putText(
            result_img,
            "Khong phat hien doi tuong",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

    # Hiển thị mode
    mode_text = "MODE: FACE" if MEASURE_MODE == "face" else "MODE: OBJECT"
    cv2.putText(
        result_img,
        mode_text,
        (10, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # === BƯỚC 5: HIỂN THỊ KẾT QUẢ ===
    cv2.imshow("Distance Estimation - Results on Original Image", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def open_image_homography():
    """Đo khoảng cách trên mặt phẳng bằng Homography"""
    mtx, dist = load_camera_params()

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

    if mtx is not None and dist is not None:
        working_img = undistort_image(img, mtx, dist)
        if working_img is None or working_img.size == 0:
            working_img = img.copy()
    else:
        working_img = img.copy()

    # === BƯỚC 1: CHỌN 4 ĐIỂM THAM CHIẾU (VD: GÓC TỜ GIẤY A4) ===
    ref_points = []
    temp_img = working_img.copy()

    messagebox.showinfo("Hướng dẫn", "Bước 1: Click chọn 4 góc của vật tham chiếu (VD: tờ giấy A4)\nTheo thứ tự: Trên trái → Trên phải → Dưới phải → Dưới trái")

    def mouse_callback(event, x, y, flags, param):
        nonlocal ref_points, temp_img
        if event == cv2.EVENT_LBUTTONDOWN and len(ref_points) < 4:
            ref_points.append([x, y])
            cv2.circle(temp_img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(temp_img, str(len(ref_points)), (x+10, y+10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.imshow("Chon 4 goc tham chieu", temp_img)

            if len(ref_points) == 4:
                # Vẽ hình chữ nhật
                pts = np.array(ref_points, np.int32)
                cv2.polylines(temp_img, [pts], True, (0, 255, 0), 2)
                cv2.imshow("Chon 4 goc tham chieu", temp_img)

    cv2.imshow("Chon 4 goc tham chieu", temp_img)
    cv2.setMouseCallback("Chon 4 goc tham chieu", mouse_callback)
    cv2.waitKey(0)
    cv2.destroyWindow("Chon 4 goc tham chieu")

    if len(ref_points) != 4:
        messagebox.showinfo("Thông báo", "Cần chọn đủ 4 điểm tham chiếu.")
        return

    # === BƯỚC 2: TÍNH MA TRẬN HOMOGRAPHY ===
    ref_points_img = np.float32(ref_points)
    H, mask = measure_homography_distance(working_img, ref_points_img, P_WORLD_REAL)

    if H is None:
        messagebox.showerror("Lỗi", "Không tính được ma trận Homography.")
        return

    # === BƯỚC 3: CHỌN 2 ĐIỂM CẦN ĐO ===
    measure_points = []
    result_img = working_img.copy()

    # Vẽ vùng tham chiếu
    pts = np.array(ref_points, np.int32)
    cv2.polylines(result_img, [pts], True, (0, 255, 0), 2)

    messagebox.showinfo("Hướng dẫn", "Bước 2: Click chọn 2 điểm bất kỳ để đo khoảng cách giữa chúng")

    def mouse_callback_measure(event, x, y, flags, param):
        nonlocal measure_points, result_img
        if event == cv2.EVENT_LBUTTONDOWN and len(measure_points) < 2:
            measure_points.append([x, y])
            color = (255, 0, 0) if len(measure_points) == 1 else (0, 0, 255)
            cv2.circle(result_img, (x, y), 6, color, -1)
            label = "A" if len(measure_points) == 1 else "B"
            cv2.putText(result_img, label, (x+10, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.imshow("Chon 2 diem can do", result_img)

            if len(measure_points) == 2:
                cv2.line(result_img, tuple(measure_points[0]),
                        tuple(measure_points[1]), (0, 255, 255), 2)
                cv2.imshow("Chon 2 diem can do", result_img)

    cv2.imshow("Chon 2 diem can do", result_img)
    cv2.setMouseCallback("Chon 2 diem can do", mouse_callback_measure)
    cv2.waitKey(0)
    cv2.destroyWindow("Chon 2 diem can do")

    if len(measure_points) != 2:
        messagebox.showinfo("Thông báo", "Cần chọn đủ 2 điểm để đo.")
        return

    # === BƯỚC 4: TÍNH KHOẢNG CÁCH THỰC TẾ ===
    points_img = np.float32(measure_points).reshape(-1, 1, 2)
    points_world_homo = cv2.perspectiveTransform(points_img, H)

    P_A_world = points_world_homo[0, 0]
    P_B_world = points_world_homo[1, 0]

    distance = np.linalg.norm(P_A_world - P_B_world)

    # === HIỂN THỊ KẾT QUẢ ===
    cv2.putText(result_img, f"Distance: {distance:.2f} cm",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(result_img, "MODE: HOMOGRAPHY (Plane)",
               (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    print(f"Điểm A: ({P_A_world[0]:.2f}, {P_A_world[1]:.2f}) cm")
    print(f"Điểm B: ({P_B_world[0]:.2f}, {P_B_world[1]:.2f}) cm")
    print(f"Khoảng cách thực tế: {distance:.2f} cm")

    cv2.imshow("Homography Measurement Result", result_img)
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
root.geometry("450x400")

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

btn_homo = tk.Button(root, text="📐 Đo Trên Mặt Phẳng (Homography)", command=open_image_homography,
                     width=28, height=2, bg="#FF9800", fg="white", font=("Arial", 10, "bold"))
btn_homo.pack(pady=10)

footer = tk.Label(root, text="Thành viên 3 – App + Integration", font=("Arial", 10))
footer.pack(pady=20)

root.mainloop()