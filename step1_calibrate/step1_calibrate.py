import os
import numpy as np
import cv2
import glob
import pickle

def run_calibration():
    # 1. Cấu hình bàn cờ (Sửa lại theo bàn cờ thật của bạn)
    CHESSBOARD_SIZE = (5,8)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 2. Chuẩn bị tọa đồ 3D ảo
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1],3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    # 3. Đọc ảnh từ thư mục
    images = glob.glob('step1_calibrate/calibration_images/*.jpg')
    print(f"Đang xử lý {len(images)} ảnh...")

    if len(images) == 0:
        print("Lỗi: Không tìm thấy ảnh trong thư mục 'calibration_images'!")
        return

    gray = None
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Tìm góc
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        print(fname, " -> ret =", ret)

        if ret is True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # (Tuỳ chọn) Vẽ để kiểm tra
            # cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners2, ret)
            # cv2.imshow('Checking', img)
            # cv2.waitKey(50)
        else:
            print("  -> Không tìm thấy đủ góc bàn cờ, bỏ qua ảnh này.")

    cv2.destroyAllWindows()

    # 4. Calibrate
    if len(objpoints) == 0 or len(imgpoints) == 0 or gray is None:
        print("Lỗi: Không có dữ liệu góc bàn cờ hợp lệ. Kiểm tra lại ảnh và kích thước CHESSBOARD_SIZE (hiện là 6x9).")
        print("Gợi ý: đảm bảo ảnh thật sự là bàn cờ 6x9 ô bên trong, và ảnh đủ sáng/độ phân giải.")
        return

    print("Đang tính toán ma trận...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 5. Xuất dữ liệu (Theo yêu cầu: Xuất camera matrix + distortion)
    data = {
        "camera_matrix": mtx,
        "dist_coeff": dist,
        "error": ret # Lưu thêm sai số để báo cáo
    }

    with open("calibration_data.pkl", "wb") as f:
        pickle.dump(data, f)

    print(f"Hoàn tất! Đã lưu file 'calibration_data.pkl'. Sai số: {ret}")

if __name__ == "__main__":
    run_calibration()
