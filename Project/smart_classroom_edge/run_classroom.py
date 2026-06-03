import argparse
import sys
from collections import deque

import cv2

from config import (
    HISTORY_LEN,
    ROOM_CAPACITY,
    resolve_model_xml,
    resolve_video_path,
)
from person_counter import PersonCounter


def parse_args():
    p = argparse.ArgumentParser(description="Smart classroom edge — person count demo")
    p.add_argument(
        "--video",
        type=str,
        default=None,
        help="视频路径（默认: 同目录 test.mp4，其次 assets/test.mp4；也可设 SMART_CLASSROOM_VIDEO）",
    )
    p.add_argument(
        "--model",
        type=str,
        default=None,
        help="OpenVINO IR 的 .xml 路径（默认: 仓库内 OMZ 模型，或由 SMART_CLASSROOM_MODEL_XML 指定）",
    )
    p.add_argument(
        "--device",
        type=str,
        default="CPU",
        help="OpenVINO 推理设备，如 CPU、GPU（依环境而定）",
    )
    return p.parse_args()


def main():
    args = parse_args()
    video_path = resolve_video_path(args.video)
    model_xml = resolve_model_xml(args.model)

    if not model_xml.is_file():
        print(
            "找不到模型文件:",
            model_xml,
            "\n请将 OMZ 仓库放在与 smart_classroom_edge 同级的 open_model_zoo-master，或设置 --model /",
            "SMART_CLASSROOM_MODEL_XML。",
            file=sys.stderr,
        )
        sys.exit(1)

    cap = None
    try:
        counter = PersonCounter(model_xml, device=args.device)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(
                "无法打开视频:",
                video_path,
                "\n请将 mp4 放到上述路径，或使用 --video <路径> / 设置环境变量 SMART_CLASSROOM_VIDEO。",
                file=sys.stderr,
            )
            sys.exit(1)

        history = deque(maxlen=HISTORY_LEN)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            raw_people_count, people_count, kept = counter.count_in_frame(frame)

            if raw_people_count > 0:
                for (x, y, bw, bh), score in kept:
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(frame_w - 1, x + bw)
                    y2 = min(frame_h - 1, y + bh)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"person {score:.2f}",
                        (x1, max(30, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

            history.append(people_count)

            avg_people = sum(history) / len(history)
            utilization = avg_people / ROOM_CAPACITY

            if avg_people == 0:
                decision = "Empty room"
            elif utilization < 0.2:
                decision = "Underutilized"
            elif utilization > 0.85:
                decision = "Crowded"
            else:
                decision = "Normal usage"

            cv2.putText(
                frame,
                f"People: {people_count}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Avg people: {avg_people:.1f}",
                (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Utilization: {utilization * 100:.1f}%",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Decision: {decision}",
                (30, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )

            cv2.imshow("Smart Classroom Edge AI", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
