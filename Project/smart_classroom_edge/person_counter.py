"""OpenVINO 人体检测与人数平滑逻辑，供演示与考勤会话共用。"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from openvino.runtime import Core
except ModuleNotFoundError:
    from openvino import Core

from config import CONF_THRESH, MAX_MISSING_FRAMES, NMS_THRESH


def decode_ssd_boxes(loc, conf, priors):
    loc = loc.reshape(-1, 4)
    conf = conf.reshape(-1, 2)
    prior_boxes = priors[0, 0].reshape(-1, 4)
    variances = priors[0, 1].reshape(-1, 4)

    scores = conf[:, 1]

    boxes = []
    valid_scores = []

    for i, score in enumerate(scores):
        if score < CONF_THRESH:
            continue

        pxmin, pymin, pxmax, pymax = prior_boxes[i]
        prior_w = pxmax - pxmin
        prior_h = pymax - pymin
        prior_cx = (pxmin + pxmax) / 2
        prior_cy = (pymin + pymax) / 2

        dx, dy, dw, dh = loc[i]
        vx, vy, vw, vh = variances[i]

        cx = vx * dx * prior_w + prior_cx
        cy = vy * dy * prior_h + prior_cy
        bw = np.exp(vw * dw) * prior_w
        bh = np.exp(vh * dh) * prior_h

        xmin = max(0.0, cx - bw / 2)
        ymin = max(0.0, cy - bh / 2)
        xmax = min(1.0, cx + bw / 2)
        ymax = min(1.0, cy + bh / 2)

        boxes.append([xmin, ymin, xmax, ymax])
        valid_scores.append(float(score))

    return boxes, valid_scores


def nms_box_indices(nms_boxes, scores):
    if not nms_boxes:
        return np.array([], dtype=np.int64)
    raw = cv2.dnn.NMSBoxes(nms_boxes, scores, CONF_THRESH, NMS_THRESH)
    if raw is None:
        return np.array([], dtype=np.int64)
    arr = np.asarray(raw).reshape(-1)
    return arr.astype(np.int64, copy=False)


class PersonCounter:
    """单帧检测 + 与 run_classroom 一致的短时丢检保持。"""

    def __init__(self, model_xml: Path | str, device: str = "CPU") -> None:
        path = Path(model_xml)
        core = Core()
        model = core.read_model(str(path))
        self._compiled = core.compile_model(model, device)
        self._input_layer = self._compiled.input(0)
        outputs = {o.any_name: o for o in self._compiled.outputs}
        self._loc_out = outputs["mbox_loc1/out/conv/flat"]
        self._conf_out = outputs["mbox_main_conf/out/conv/flat/softmax/flat"]
        self._prior_out = outputs["mbox/priorbox"]
        self._missing_frames = 0
        self._last_valid_count = 0

    def reset_smoothing(self) -> None:
        self._missing_frames = 0
        self._last_valid_count = 0

    def count_in_frame(
        self, frame: np.ndarray
    ) -> tuple[int, int, list[tuple[list[int], float]]]:
        """
        返回 (原始检测人数, 平滑后显示人数, [(x,y,w,h), score], ...)。
        """
        frame_h, frame_w = frame.shape[:2]
        _, _, input_h, input_w = self._input_layer.shape
        resized = cv2.resize(frame, (input_w, input_h))
        input_data = resized.transpose(2, 0, 1)[None, :, :, :].astype(np.float32)

        results = self._compiled([input_data])
        loc = results[self._loc_out]
        conf = results[self._conf_out]
        priors = results[self._prior_out]

        boxes, scores = decode_ssd_boxes(loc, conf, priors)

        nms_boxes: list[list[int]] = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            x = int(xmin * frame_w)
            y = int(ymin * frame_h)
            bw = int((xmax - xmin) * frame_w)
            bh = int((ymax - ymin) * frame_h)
            nms_boxes.append([x, y, bw, bh])

        indices = nms_box_indices(nms_boxes, scores)
        raw_people_count = int(len(indices))

        if raw_people_count > 0:
            smoothed = raw_people_count
            self._last_valid_count = raw_people_count
            self._missing_frames = 0
        else:
            self._missing_frames += 1
            if self._missing_frames <= MAX_MISSING_FRAMES:
                smoothed = self._last_valid_count
            else:
                smoothed = 0

        kept: list[tuple[list[int], float]] = []
        for idx in indices:
            i = int(idx)
            kept.append((nms_boxes[i], float(scores[i])))
        return raw_people_count, smoothed, kept
