import cv2
import numpy as np

# --- 1. THIẾT LẬP THAM SỐ VÀ CÁC HẰNG SỐ ---

# **A. THAM SỐ CHO SIZE-BASED ESTIMATION (Đo khoảng cách tới camera)**
# Kích thước thực tế của vật chuẩn (Ví dụ: chiều rộng của một thẻ ID hoặc hộp)
W_REAL = 5.398  # Chiều rộng thực tế (cm) của vật chuẩn (ví dụ: thẻ tín dụng)
D_REF = 50.0    # Khoảng cách tham chiếu khi đo (cm)
# Kích thước pixel của vật chuẩn khi nó ở khoảng cách D_REF (CẦN ĐO THỦ CÔNG MỘT LẦN)
W_PIXEL_REF = 160 # Ví dụ: 160 pixels

# Tính toán Focal Length / Pixel Constant (K_ref)
K_REF = (W_PIXEL_REF * D_REF) / W_REAL


# **B. THAM SỐ CHO HOMOGRAPHY (Đo khoảng cách trên mặt phẳng)**
# Tọa độ thực tế trên mặt phẳng thế giới (cm) của 4 điểm chuẩn (ví dụ: một tờ giấy A4)
# (0, 0) là góc trên trái
P_WORLD_REAL = np.float32([
    [0, 0],
    [29.7, 0],   # Chiều rộng A4 ~ 29.7 cm
    [29.7, 21.0], # Chiều cao A4 ~ 21.0 cm
    [0, 21.0]
])

#Đo khoảng cách size-based
def estimate_size_based_distance(image_path, W_real, K_ref):
    """
    Ước lượng khoảng cách tới vật thể bằng phương pháp Size-Based.
    Giả định: Vật thể đã được phát hiện và kích thước pixel đã được đo.
    """
    print("\n--- Phương pháp 1: Size-Based Estimation ---")
    
    # 1. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh tại đường dẫn: {image_path}")
        return

    # **2. Phát hiện và Đo kích thước Pixel (Đây là phần phức tạp nhất trong thực tế, 
    #     ta sẽ giả định bằng cách cho một giá trị)**
    # Thay thế bằng code phát hiện đối tượng thực tế (ví dụ: dùng Haar Cascade hoặc Yolo)
    W_pixel_detected = 85 # GIÁ TRỊ GIẢ ĐỊNH cho ảnh hiện tại

    if W_pixel_detected > 0:
        # 3. Tính toán khoảng cách
        D_estimated = (W_real * K_ref) / W_pixel_detected
        
        # 4. Hiển thị kết quả (giả định vẽ một bounding box)
        # Chỉ để minh họa, bạn cần tọa độ Bbox thực tế
        cv2.putText(img, f"Khoang cach: {D_estimated:.2f} cm", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        print(f"Kích thước Pixel vật thể: {W_pixel_detected} pixels")
        print(f"Khoảng cách ước lượng tới Camera: {D_estimated:.2f} cm")
    else:
        print("Không tìm thấy vật thể trong ảnh.")

    cv2.imshow("Size-Based Result", img)
    return D_estimated


#Đo khoảng cách bằng homography
def measure_homography_distance(image_path, P_world_real):
    """
    Đo khoảng cách trên một mặt phẳng bằng Homography.
    Cần 4 điểm tương ứng giữa ảnh và thế giới.
    """
    print("\n--- Phương pháp 2: Homography Measurement ---")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Lỗi: Không thể đọc ảnh tại đường dẫn: {image_path}")
        return

    # **1. Tọa độ 4 điểm trên ảnh (CẦN XÁC ĐỊNH BẰNG TAY HOẶC PHÁT HIỆN GÓC)**
    # Giả định 4 điểm góc của tờ A4 trên ảnh
    P_IMAGE = np.float32([
        [150, 200],  # Góc trên trái
        [500, 220],  # Góc trên phải
        [650, 650],  # Góc dưới phải
        [100, 700]   # Góc dưới trái
    ])
    
    # 2. Tìm Ma trận Homography (H)
    H, mask = cv2.findHomography(P_IMAGE, P_world_real, cv2.RANSAC, 5.0)

    if H is None:
        print("Lỗi: Không tìm thấy Ma trận Homography.")
        return

    print("Ma trận Homography (H) đã được tính.")
    
    # **3. Đo Khoảng cách giữa 2 điểm bất kỳ (P_A và P_B) trên mặt phẳng này**
    P_A_img = [300, 350] # Điểm A trên ảnh
    P_B_img = [450, 500] # Điểm B trên ảnh

    points_img = np.float32([P_A_img, P_B_img]).reshape(-1, 1, 2)
    
    # Áp dụng Homography
    points_world_homo = cv2.perspectiveTransform(points_img, H)
    
    P_A_world = points_world_homo[0, 0]
    P_B_world = points_world_homo[1, 0]
    
    # Tính khoảng cách Euclidean
    distance = np.linalg.norm(P_A_world - P_B_world)

    # 4. Hiển thị kết quả
    cv2.circle(img, tuple(P_A_img), 5, (0, 0, 255), -1) # Đỏ: A
    cv2.circle(img, tuple(P_B_img), 5, (255, 0, 0), -1) # Xanh dương: B
    cv2.line(img, tuple(P_A_img), tuple(P_B_img), (0, 255, 255), 2)
    
    cv2.putText(img, f"Khoang cach thuc te: {distance:.2f} cm", (50, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
    
    print(f"Điểm A thế giới: ({P_A_world[0]:.2f}, {P_A_world[1]:.2f}) cm")
    print(f"Điểm B thế giới: ({P_B_world[0]:.2f}, {P_B_world[1]:.2f}) cm")
    print(f"Khoảng cách thực tế trên mặt phẳng giữa A và B: {distance:.2f} cm")

    cv2.imshow("Homography Result", img)
    return distance


if __name__ == "__main__":
    
    # ĐƯỜNG DẪN ẢNH CỦA BẠN:
    # Thay thế bằng đường dẫn ảnh thực tế bạn chụp
    IMAGE_SIZE_BASED = "path/to/your/size_based_test_image.jpg"
    IMAGE_HOMOGRAPHY = "path/to/your/homography_test_image.jpg"

    print(f"Focal Length Constant (K_REF) đã tính: {K_REF:.2f}")

    # --- CHẠY SIZE-BASED ESTIMATION ---
    estimated_depth = estimate_size_based_distance(IMAGE_SIZE_BASED, W_REAL, K_REF)
    
    print("-" * 50)
    
    # --- CHẠY HOMOGRAPHY MEASUREMENT ---
    measured_plane_distance = measure_homography_distance(IMAGE_HOMOGRAPHY, P_WORLD_REAL)
    
    # Chờ phím bấm và đóng cửa sổ
    cv2.waitKey(0)
    cv2.destroyAllWindows()