"""
教室人数考勤会话：按时间记录人数，生成走势图与班级层面汇总。

说明：仅基于画面中检测到的人数，无法识别具体学生身份；
「疑似签离」为启发式提示（签到阶段峰值 vs 课中位数），不能作为对个人的处分依据。
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from config import resolve_model_xml, resolve_video_path
from person_counter import PersonCounter


@dataclass
class SessionRow:
    time_sec: float
    frame: int
    raw_count: int
    smoothed_count: int


def parse_args():
    p = argparse.ArgumentParser(
        description="教室打卡会话：记录人数变化并生成走势图与汇总（班级统计，非个人识别）"
    )
    p.add_argument("--video", type=str, default=None, help="视频路径（默认同 run_classroom）")
    p.add_argument("--model", type=str, default=None, help="OpenVINO IR .xml")
    p.add_argument("--device", type=str, default="CPU", help="推理设备")
    p.add_argument(
        "--expected",
        type=int,
        required=True,
        help="应到人数（花名册人数），用于估算缺课与对比峰值",
    )
    p.add_argument(
        "--signin-minutes",
        type=float,
        default=15.0,
        help="签到阶段时长（分钟），该时段内取人数峰值作为「到堂高峰」参考",
    )
    p.add_argument(
        "--max-minutes",
        type=float,
        default=None,
        help="摄像头模式下最长运行分钟数（到点自动结束）；不指定则按 ESC 结束",
    )
    p.add_argument(
        "--sample-every",
        type=int,
        default=1,
        help="每 N 帧记录一行（>1 可减小 CSV 与加速）",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认: smart_classroom_edge/attendance_logs/<时间戳>）",
    )
    p.add_argument(
        "--no-show",
        action="store_true",
        help="不弹窗预览（适合无显示器服务器）",
    )
    p.add_argument(
        "--no-plot",
        action="store_true",
        help="仅写 CSV/JSON，不生成 PNG（需 matplotlib）",
    )
    return p.parse_args()


def default_output_dir() -> Path:
    base = Path(__file__).resolve().parent / "attendance_logs"
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = base / f"session_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def compute_summary(
    rows: list[SessionRow],
    expected: int,
    signin_minutes: float,
    video_duration_sec: float | None,
) -> dict:
    if not rows:
        return {"error": "无采样数据"}

    times = np.array([r.time_sec for r in rows], dtype=np.float64)
    smooth = np.array([r.smoothed_count for r in rows], dtype=np.float64)
    total_sec = float(times[-1] - times[0]) if len(times) > 1 else 0.0
    if total_sec <= 0:
        total_sec = float(times[-1]) if len(times) else 0.0

    signin_sec = min(signin_minutes * 60.0, max(total_sec * 0.35, 30.0))
    if signin_sec > total_sec * 0.9:
        signin_sec = total_sec * 0.25

    mask_signin = times <= (times[0] + signin_sec)
    mask_class = ~mask_signin

    peak_signin = int(np.max(smooth[mask_signin])) if np.any(mask_signin) else 0
    if np.any(mask_class):
        class_vals = smooth[mask_class]
        median_class = float(np.median(class_vals))
        p90_class = float(np.percentile(class_vals, 90))
        mean_class = float(np.mean(class_vals))
    else:
        median_class = mean_class = p90_class = float(np.median(smooth))

    # 到课/缺课为粗估：以课中位数代表「大致在位人数」
    estimated_present = min(expected, int(round(median_class)))
    absent_estimate = max(0, expected - estimated_present)

    drop_abs = max(3, int(round(0.12 * max(expected, 1))))
    peak_ok = peak_signin >= min(5, max(3, int(0.35 * max(expected, 1))))
    delta_peak_median = peak_signin - median_class
    possible_early_leave_pattern = bool(
        peak_ok and delta_peak_median >= drop_abs and median_class < peak_signin * 0.75
    )

    return {
        "expected": expected,
        "signin_minutes_config": signin_minutes,
        "signin_window_sec_effective": round(signin_sec, 2),
        "session_span_sec": round(total_sec, 2),
        "video_duration_sec": video_duration_sec,
        "peak_during_signin": peak_signin,
        "median_during_class": round(median_class, 2),
        "mean_during_class": round(mean_class, 2),
        "p90_during_class": round(p90_class, 2),
        "estimated_present_by_median": estimated_present,
        "absent_estimate": absent_estimate,
        "possible_early_leave_pattern": possible_early_leave_pattern,
        "early_leave_hint": (
            "签到阶段人数峰值明显高于课中位数，可能存在「到堂后陆续离开」或检测波动；"
            "不能指向具体个人，建议结合点名、座位或考勤机复核。"
            if possible_early_leave_pattern
            else "未触发「峰值显著回落」启发式；若需更严判据可调阈值或拉长签到窗口。"
        ),
        "disclaimer": (
            "本结果仅依据画面中人体数量估计，受视角、遮挡、重叠与模型误差影响；"
            "不构成对单人的签到或缺勤法律/纪律认定。"
        ),
    }


def write_csv(path: Path, rows: list[SessionRow]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["time_sec", "frame", "raw_count", "smoothed_count"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def write_plot(path: Path, rows: list[SessionRow], signin_sec: float, summary: dict) -> None:
    import matplotlib.pyplot as plt

    try:
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    t0 = rows[0].time_sec
    x = [r.time_sec - t0 for r in rows]
    y_raw = [r.raw_count for r in rows]
    y_sm = [r.smoothed_count for r in rows]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot(x, y_raw, alpha=0.35, label="原始检测人数")
    ax.plot(x, y_sm, linewidth=1.8, label="平滑后人数（展示用）")
    ax.axvline(signin_sec, color="orange", linestyle="--", label="签到阶段结束（约）")
    ax.axhline(summary["expected"], color="gray", linestyle=":", label="应到人数")
    ax.set_xlabel("时间 (秒)")
    ax.set_ylabel("人数")
    ax.set_title("课堂人数随时间变化（班级统计）")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    model_xml = resolve_model_xml(args.model)
    video_path = resolve_video_path(args.video)

    if not model_xml.is_file():
        print("找不到模型文件:", model_xml, file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else default_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("无法打开视频:", video_path, file=sys.stderr)
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count_est = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_from_meta = None
    if fps > 1e-3 and frame_count_est > 0:
        duration_from_meta = frame_count_est / fps

    counter = PersonCounter(model_xml, device=args.device)
    rows: list[SessionRow] = []
    frame_idx = 0
    t_start = time.perf_counter()
    max_run_sec = (args.max_minutes * 60.0) if args.max_minutes is not None else None

    print("输出目录:", out_dir)
    print("应到人数:", args.expected, "| 签到阶段(配置):", args.signin_minutes, "分钟")
    print("按 ESC 提前结束。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if fps > 1e-3:
                time_sec = frame_idx / fps
            else:
                time_sec = time.perf_counter() - t_start

            if max_run_sec is not None and time_sec >= max_run_sec:
                break

            raw_c, smooth_c, kept = counter.count_in_frame(frame)

            if frame_idx % args.sample_every == 0:
                rows.append(
                    SessionRow(
                        time_sec=time_sec,
                        frame=frame_idx,
                        raw_count=raw_c,
                        smoothed_count=smooth_c,
                    )
                )

            if not args.no_show:
                vis = frame.copy()
                h, w = vis.shape[:2]
                for (x, y, bw, bh), sc in kept:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(w - 1, x + bw), min(h - 1, y + bh)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    vis,
                    f"smoothed={smooth_c} raw={raw_c}",
                    (20, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("Attendance session", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1
    finally:
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()

    csv_path = out_dir / "timeseries.csv"
    json_path = out_dir / "summary.json"
    plot_path = out_dir / "trend.png"

    write_csv(csv_path, rows)

    summary = compute_summary(
        rows,
        expected=args.expected,
        signin_minutes=args.signin_minutes,
        video_duration_sec=duration_from_meta,
    )
    # 供作图用的签到窗长（与 compute_summary 内逻辑一致，从 summary 取）
    signin_sec_eff = float(summary.get("signin_window_sec_effective", args.signin_minutes * 60))

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not args.no_plot and rows:
        try:
            write_plot(plot_path, rows, signin_sec_eff, summary)
        except ImportError:
            print("未安装 matplotlib，跳过 trend.png。可: pip install matplotlib", file=sys.stderr)

    print("已写入:", csv_path)
    print("已写入:", json_path)
    if plot_path.is_file():
        print("已写入:", plot_path)
    print("汇总:", json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
