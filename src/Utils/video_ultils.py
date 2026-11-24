import os
import cv2


def read_video(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Loi: Khong the mo video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    current_frame = 0

    print(f"Bat dau doc video ({total_frames} frames)...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        current_frame += 1

        if total_frames > 0 and (current_frame % 50 == 0 or current_frame == total_frames):
            percent = (current_frame / total_frames) * 100
            print(f"\rDang doc video: {percent:.1f}% ({current_frame}/{total_frames} frames)", end="", flush=True)

    print()
    cap.release()
    return frames


def read_video_stream(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Loi: Khong the mo video: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

    cap.release()



def save_video(frames, output_video_path: str, fps: float = 24.0, fourcc_code: str = "mp4v")-> None:
    if frames is None:
        print("Loi: frames la None, khong the luu video!")
        return

    frames_iter = iter(frames)
    try:
        first_frame = next(frames_iter)
    except StopIteration:
        print("Loi: Danh sach frames dang trong, khong the luu video!")
        return

    height, width, _ = first_frame.shape
    total_frames = len(frames) if hasattr(frames, "__len__") else None
    #create output directory if it doesn't exist
    output_dir = os.path.dirname(output_video_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    if not out.isOpened():
        out.release()
        raise RuntimeError(f"Khong the mo VideoWriter de ghi video: {output_video_path}")

    idx = 0
    try:
        # Write incrementally. Do not collect generated frames into a list.
        out.write(first_frame)
        idx = 1
        del first_frame

        print("Bat dau luu video...")
        for frame in frames_iter:
            if frame is None or frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame {idx} co kich thuoc khong hop le; mong doi {(width, height)}"
                )
            out.write(frame)
            idx += 1

            if total_frames:
                if idx % 50 == 0 or idx == total_frames:
                    percent = (idx / total_frames) * 100
                    print(f"\rTien do luu video: {percent:.1f}% ({idx}/{total_frames} frames)", end="", flush=True)
            elif idx % 50 == 0:
                print(f"\rTien do luu video: {idx} frames da ghi...", end="", flush=True)

        if total_frames:
            print(f"\rTien do luu video: 100.0% ({idx}/{total_frames} frames)")
        else:
            print(f"\rTien do luu video: Da hoan tat {idx} frames!")
    finally:
        out.release()
        close = getattr(frames_iter, "close", None)
        if close is not None:
            close()

    print(f"Da luu video thanh cong tai: {output_video_path}")
