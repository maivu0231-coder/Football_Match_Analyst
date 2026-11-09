import numpy as np
import pandas as pd


class KalmanBallTracker:
    """Kalman Filter & RTS Smoother cho 2D Bounding Box:

    State vector X = [center_x, center_y, vx, vy, width, height]^T
    Measurement Z  = [center_x, center_y, width, height]^T
    """

    def __init__(self, dt: float = 1.0):
        self.dim_x = 6
        self.dim_z = 4

        # Ma trận chuyển trạng thái (Constant Velocity cho tọa độ tâm, kích thước giữ ổn định)
        self.F = np.eye(self.dim_x)
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # Ma trận đo lường
        self.H = np.zeros((self.dim_z, self.dim_x))
        self.H[0, 0] = 1.0  # cx
        self.H[1, 1] = 1.0  # cy
        self.H[2, 4] = 1.0  # w
        self.H[3, 5] = 1.0  # h

        # Hiệp phương sai nhiễu quá trình Q (Process Noise)
        # Cho phép vận tốc thay đổi linh hoạt khi bóng đổi hướng/nảy
        self.Q = np.eye(self.dim_x)
        self.Q[0:2, 0:2] *= 1.0  # Nhiễu vị trí
        self.Q[2:4, 2:4] *= 4.0  # Nhiễu vận tốc (tăng để thích ứng khi bóng đổi hướng)
        self.Q[4:6, 4:6] *= 0.5  # Nhiễu kích thước

        # Hiệp phương sai nhiễu đo lường R (Measurement Noise)
        self.R = np.eye(self.dim_z) * 5.0

    def smooth(
        self, measurements: np.ndarray, max_allowed_distance: float = 100.0
    ):
        """measurements: numpy array shape (N, 4) chứa [cx, cy, w, h], các frame

        không có detection là NaN.
        """
        n_frames = len(measurements)
        if n_frames == 0:
            return np.array([])

        # Tìm frame hợp lệ đầu tiên để khởi tạo trạng thái
        valid_indices = np.where(~np.isnan(measurements[:, 0]))[0]
        if len(valid_indices) == 0:
            return measurements

        first_valid_idx = valid_indices[0]

        # Mảng lưu trữ forward pass
        x_pred = np.zeros((n_frames, self.dim_x))
        P_pred = np.zeros((n_frames, self.dim_x, self.dim_x))
        x_filt = np.zeros((n_frames, self.dim_x))
        P_filt = np.zeros((n_frames, self.dim_x, self.dim_x))

        # Khởi tạo trạng thái ban đầu
        first_meas = measurements[first_valid_idx]
        current_x = np.array(
            [first_meas[0], first_meas[1], 0.0, 0.0, first_meas[2], first_meas[3]]
        )
        current_P = np.eye(self.dim_x) * 10.0

        for k in range(first_valid_idx, n_frames):
            # 1. Dự đoán (Predict)
            if k == first_valid_idx:
                xp = current_x
                Pp = current_P
            else:
                xp = self.F @ current_x
                Pp = self.F @ current_P @ self.F.T + self.Q

            x_pred[k] = xp
            P_pred[k] = Pp

            # 2. Cập nhật (Update) nếu có detection hợp lệ
            z = measurements[k]
            is_valid = not np.isnan(z[0])

            if is_valid:
                # Kiểm tra khoảng cách gating với vị trí dự đoán để lọc outlier
                pred_center = xp[:2]
                meas_center = z[:2]
                dist = np.linalg.norm(meas_center - pred_center)

                if dist > max_allowed_distance:
                    is_valid = False  # Xem như miss detection vì nhảy quá xa

            if is_valid:
                y = z - (self.H @ xp)  # Sai số đo lường (Residual)
                S = self.H @ Pp @ self.H.T + self.R
                K = Pp @ self.H.T @ np.linalg.inv(S)  # Kalman Gain

                current_x = xp + K @ y
                current_P = (np.eye(self.dim_x) - K @ self.H) @ Pp
            else:
                # Khi không có detection: giữ nguyên giá trị dự đoán
                current_x = xp
                current_P = Pp

            x_filt[k] = current_x
            P_filt[k] = current_P

        # Nếu các frame đầu video chưa có detection, gán bằng frame hợp lệ đầu tiên
        for k in range(first_valid_idx):
            x_filt[k] = x_filt[first_valid_idx]
            P_filt[k] = P_filt[first_valid_idx]
            x_pred[k] = x_filt[first_valid_idx]
            P_pred[k] = P_filt[first_valid_idx]

        # 3. RTS Backward Smoothing Pass
        x_smooth = np.copy(x_filt)
        for k in range(n_frames - 2, -1, -1):
            Pp_next = P_pred[k + 1]
            # Ma trận tăng ích làm mượt C
            C = P_filt[k] @ self.F.T @ np.linalg.pinv(Pp_next)
            x_smooth[k] = x_filt[k] + C @ (x_smooth[k + 1] - x_pred[k + 1])

        # Trích xuất lại [cx, cy, w, h] từ state
        smoothed_boxes = np.column_stack(
            [
                x_smooth[:, 0],  # cx
                x_smooth[:, 1],  # cy
                x_smooth[:, 4],  # w
                x_smooth[:, 5],  # h
            ]
        )
        return smoothed_boxes