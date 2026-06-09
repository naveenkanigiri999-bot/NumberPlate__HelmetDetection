
import os
import re
import subprocess
import sys
import time
import pathlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2 as cv
import easyocr
import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath


RIDER_CONFIDENCE = 0.2
RIDER_NMS = 0.3
HELMET_CONFIDENCE = 0.55
HELMET_NMS = 0.35
PLATE_CONFIDENCE = 0.35

PLATE_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


STATE_CODE_MAP_INDIA = {
    "AN": "Andaman and Nicobar",
    "AP": "Andhra Pradesh",
    "AR": "Arunachal Pradesh",
    "AS": "Assam",
    "BR": "Bihar",
    "CG": "Chhattisgarh",
    "CH": "Chandigarh",
    "DD": "Daman and Diu",
    "DL": "Delhi",
    "GA": "Goa",
    "GJ": "Gujarat",
    "HR": "Haryana",
    "HP": "Himachal Pradesh",
    "JH": "Jharkhand",
    "JK": "Jammu and Kashmir",
    "KA": "Karnataka",
    "KL": "Kerala",
    "LA": "Ladakh",
    "LD": "Lakshadweep",
    "MH": "Maharashtra",
    "ML": "Meghalaya",
    "MN": "Manipur",
    "MP": "Madhya Pradesh",
    "MZ": "Mizoram",
    "NL": "Nagaland",
    "OD": "Odisha",
    "PB": "Punjab",
    "PY": "Puducherry",
    "RJ": "Rajasthan",
    "SK": "Sikkim",
    "TN": "Tamil Nadu",
    "TR": "Tripura",
    "TS": "Telangana",
    "UK": "Uttarakhand",
    "UP": "Uttar Pradesh",
    "WB": "West Bengal",
}

LOCAL_VEHICLE_REGISTRY = {
    "MH12AB1234": {
        "owner_name": "Demo Owner",
        "vehicle_make": "Honda",
        "vehicle_model": "Activa",
        "fuel_type": "Petrol",
    }
}

USER_DATABASE = {
        "1": {"email": "user1@gmail.com"},
        "2": {"email": "user2@gmail.com"},
        "3": {"email": "user3@gmail.com"},
        "4": {"email": "user4@gmail.com"},
        "5": {"email": "user5@gmail.com"},
        "6": {"email": "user6@gmail.com"},
        "7": {"email": "naveenkanigiri999@gmail.com"},
        "8": {"email": "user8@gmail.com"},
        "9": {"email": "user9@gmail.com"},
        "10": {"email": "user10@gmail.com"},
        "11": {"email": "user11@gmail.com"},
        "12": {"email": "user12@gmail.com"},
        "13": {"email": "user13@gmail.com"},
        "14": {"email": "user14@gmail.com"},
        "15": {"email": "user15@gmail.com"},
        "16": {"email": "user16@gmail.com"},
        "17": {"email": "user17@gmail.com"},
        "18": {"email": "user18@gmail.com"},
        "19": {"email": "user19@gmail.com"},
        "20": {"email": "user20@gmail.com"},
        "21": {"email": "user21@gmail.com"},
        "22": {"email": "user22@gmail.com"},
        "23": {"email": "user23@gmail.com"},
        "24": {"email": "user24@gmail.com"},
        "25": {"email": "user25@gmail.com"},
        "26": {"email": "user26@gmail.com"},
        "27": {"email": "user27@gmail.com"},
        "28": {"email": "user28@gmail.com"},
        "29": {"email": "user29@gmail.com"},
        "30": {"email": "user30@gmail.com"}
}

CHALLAN_RULES = {
    "India": {
        "currency": "INR",
        "helmet_no_rider": 1000,
        "notes": "Default configuration; update per state law before production use.",
    },
    "United States": {
        "currency": "USD",
        "helmet_no_rider": 150,
        "notes": "Default configuration; update per state law before production use.",
    },
    "United Kingdom": {
        "currency": "GBP",
        "helmet_no_rider": 100,
        "notes": "Default configuration; update per latest DVLA/police guidance.",
    },
}


@dataclass
class Detection:
    label: str
    confidence: float
    box: Tuple[int, int, int, int]


def first_existing_path(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def clip_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def box_area(box: Tuple[int, int, int, int]) -> int:
    _, _, w, h = box
    return max(0, w) * max(0, h)


def intersection_area(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> int:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def box_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    inter = intersection_area(a, b)
    if inter == 0:
        return 0.0
    union = box_area(a) + box_area(b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def flatten_indices(indices) -> List[int]:
    if indices is None:
        return []
    try:
        arr = np.array(indices).flatten()
        return [int(i) for i in arr]
    except Exception:
        return []


def wrapped_lines(text: str, font_scale: float, thickness: int, max_width: int) -> List[str]:
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test_line = word if current == "" else current + " " + word
        (w, _), _ = cv.getTextSize(test_line, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        if w <= max_width:
            current = test_line
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def draw_text_label(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    text_color: Tuple[int, int, int],
    bg_color: Optional[Tuple[int, int, int]] = (18, 24, 38),
    font_scale: float = None,
    thickness: int = 1,
) -> None:
    h, w = frame.shape[:2]

    # 🔥 dynamic font scaling
    if font_scale is None:
        font_scale = max(0.4, min(0.8, w / 1200))

    # 🔥 max width constraint (VERY IMPORTANT)
    max_width = int(w * 0.35)

    # smart wrapping
    lines = wrapped_lines(text, font_scale, thickness, max_width)

    if not lines:
        return

    line_height = int(22 * font_scale)
    text_sizes = [cv.getTextSize(line, cv.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0] for line in lines]

    block_w = min(max(size[0] for size in text_sizes) + 10, max_width)
    block_h = (line_height * len(lines)) + 10

    x, y = origin

    # 🔥 keep inside frame (no overflow)
    x = max(4, min(int(x), w - block_w - 10))
    y = min(max(block_h + 4, int(y)), h - 10)

    top = y - block_h

    # 🔥 background box
    if bg_color is not None:
        overlay = frame.copy()
        cv.rectangle(overlay, (x, top), (x + block_w, y), bg_color, -1)
        cv.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv.rectangle(frame, (x, top), (x + block_w, y), (120, 130, 150), 1)

    # 🔥 draw text
    for i, line in enumerate(lines):
        y_pos = top + (i + 1) * line_height - 6
        cv.putText(
            frame,
            line,
            (x + 5, y_pos),
            cv.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_color,
            thickness,
            cv.LINE_AA,
        )


def sanitize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def coerce_by_pattern(raw: str, pattern: str) -> Tuple[str, int]:
    to_digit = {
        "O": "0",
        "Q": "0",
        "D": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
        "T": "7",
    }
    to_alpha = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "8": "B",
        "7": "T",
    }

    out = []
    edits = 0
    for ch, expected in zip(raw, pattern):
        original = ch
        if expected == "D":
            if not ch.isdigit():
                ch = to_digit.get(ch, ch)
            if not ch.isdigit():
                return "", 999
        else:
            if not ch.isalpha():
                ch = to_alpha.get(ch, ch)
            if not ch.isalpha():
                return "", 999
        if ch != original:
            edits += 1
        out.append(ch)
    return "".join(out), edits


def normalize_india_plate(raw: str) -> Tuple[str, bool, float]:
    value = sanitize_plate_text(raw)
    if len(value) < 8:
        return value, False, -0.5

    best_text = value
    best_score = -1.0
    best_valid = False

    for district_len in (1, 2):
        for series_len in (1, 2, 3):
            expected_len = 2 + district_len + series_len + 4
            if len(value) != expected_len:
                continue
            pattern = "A" * 2 + "D" * district_len + "A" * series_len + "D" * 4
            fixed, edits = coerce_by_pattern(value, pattern)
            if not fixed:
                continue
            regex = re.compile(rf"^[A-Z]{{2}}\d{{{district_len}}}[A-Z]{{{series_len}}}\d{{4}}$")
            is_valid = bool(regex.match(fixed))
            if not is_valid:
                continue
            score = 0.7 - (edits * 0.05)
            if score > best_score:
                best_score = score
                best_text = fixed
                best_valid = True

    if best_valid:
        return best_text, True, best_score

    return value, False, -0.2


def normalize_plate(raw: str, country: str) -> Tuple[str, bool, float]:
    value = sanitize_plate_text(raw)
    if not value:
        return "", False, -1.0

    if country == "India":
        return normalize_india_plate(value)

    if country == "United Kingdom":
        fixed, edits = coerce_by_pattern(value, "AADD AAA".replace(" ", "")) if len(value) == 7 else (value, 0)
        valid = bool(re.match(r"^[A-Z]{2}\d{2}[A-Z]{3}$", fixed))
        score = 0.6 - (edits * 0.05) if valid else -0.2
        return fixed, valid, score

    valid = 5 <= len(value) <= 8
    score = 0.35 if valid else -0.1
    return value, valid, score

def send_violation_email(to_email, subject, body, root=None):
    sender_email = "naveenkanigiri999@gmail.com"
    sender_password = "qkkh vbpr kgbs iobj"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()

        print(f"✅ Email sent to {to_email}")

        # ✅ POPUP SUCCESS
        if root:
            messagebox.showinfo(
                "Email Sent",
                f"Email successfully sent to:\n{to_email}"
            )

    except Exception as e:
        print("❌ Email failed:", e)

        # ❌ POPUP ERROR
        if root:
            messagebox.showerror(
                "Email Failed",
                f"Error:\n{str(e)}"
            )

def build_violation_message(result):
    msg = "🚨 Traffic Violation Detected\n\n"

    if result["no_helmet_count"] > 0 or not result["plate_detected"] or not result["plate_valid"]:
        msg += f"❌ No Helmet Riders: {result['no_helmet_count']}\n"

    if not result["plate_detected"]:
        msg += "❌ Number Plate Not Detected\n"

    if result["plate_detected"] and not result["plate_valid"]:
        msg += "⚠ Invalid Number Plate\n"

    msg += f"\nPlate: {result['plate_text']}"
    msg += f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}"

    return msg

def ensure_pkg_resources_available() -> None:
    try:
        import pkg_resources  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    cmd = [sys.executable, "-m", "pip", "install", "setuptools<81", "wheel"]
    subprocess.check_call(cmd)

    try:
        import pkg_resources  # noqa: F401
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "setuptools is required (pkg_resources missing). Install with: "
            f"\"{sys.executable} -m pip install setuptools<81 wheel\""
        ) from exc


class ViolationEngine:
    def __init__(self, app_dir: Path) -> None:
        self.app_dir = app_dir
        self.project_dir = app_dir.parent

        self.rider_net = None
        self.helmet_net = None
        self.plate_model = None
        self.reader = None
        self.rider_labels: List[str] = []
        self.helmet_labels: List[str] = []
        self.models_loaded = False

        self.last_loaded_paths: Dict[str, str] = {}

    def load_models(self) -> None:
        rider_cfg = self.app_dir / "yolov3model" / "yolov3.cfg"
        rider_weights = self.app_dir / "yolov3model" / "yolov3.weights"
        rider_labels = first_existing_path(
            [
                self.app_dir / "yolov3model" / "label_backup.txt",
                self.app_dir / "yolov3model" / "yolov3-labels",
            ]
        )

        helmet_cfg = self.app_dir / "Models" / "yolov3-obj.cfg"
        helmet_weights = self.app_dir / "Models" / "yolov3-obj_2400.weights"
        helmet_labels = self.app_dir / "Models" / "obj.names"

        plate_model_path = first_existing_path(
            [
                self.project_dir / "NumberPlate" / "model" / "best.pt",
                self.project_dir / "NumberPlate" / "NumberPlate" / "model" / "best.pt",
            ]
        )
        yolov5_repo = first_existing_path(
            [
                self.project_dir / "NumberPlate" / "yolov5",
                self.project_dir / "NumberPlate" / "NumberPlate" / "yolov5",
            ]
        )

        missing = []
        for path in [
            rider_cfg,
            rider_weights,
            rider_labels,
            helmet_cfg,
            helmet_weights,
            helmet_labels,
            plate_model_path,
            yolov5_repo,
        ]:
            if path is None or not Path(path).exists():
                missing.append(str(path))

        if missing:
            raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))

        self.rider_net = cv.dnn.readNetFromDarknet(str(rider_cfg), str(rider_weights))
        self.rider_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.rider_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        self.helmet_net = cv.dnn.readNetFromDarknet(str(helmet_cfg), str(helmet_weights))
        self.helmet_net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.helmet_net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

        with open(rider_labels, "r", encoding="utf-8") as handle:
            self.rider_labels = [line.strip() for line in handle if line.strip()]

        with open(helmet_labels, "r", encoding="utf-8") as handle:
            self.helmet_labels = [line.strip() for line in handle if line.strip()]

        ensure_pkg_resources_available()
        self.plate_model = torch.hub.load(
            str(yolov5_repo),
            "custom",
            path=str(plate_model_path),
            source="local",
            force_reload=False,
        )
        self.plate_model.conf = PLATE_CONFIDENCE
        self.plate_model.iou = 0.45

        self.reader = easyocr.Reader(["en"], gpu=False)
        self.models_loaded = True

        self.last_loaded_paths = {
            "rider_cfg": str(rider_cfg),
            "helmet_cfg": str(helmet_cfg),
            "plate_model": str(plate_model_path),
            "yolov5_repo": str(yolov5_repo),
        }

    def _output_layer_names(self, net) -> List[str]:
        layer_names = net.getLayerNames()
        return [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    def detect_with_dnn(
        self,
        frame: np.ndarray,
        net,
        labels: List[str],
        conf_threshold: float,
        nms_threshold: float,
        keep_labels: Optional[set] = None,
    ) -> List[Detection]:
        height, width = frame.shape[:2]
        blob = cv.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(self._output_layer_names(net))

        boxes = []
        confidences = []
        class_ids = []

        for out in outs:
            for det in out:
                scores = det[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < conf_threshold:
                    continue
                if class_id >= len(labels):
                    continue
                label = labels[class_id]
                if keep_labels and label not in keep_labels:
                    continue

                cx = int(det[0] * width)
                cy = int(det[1] * height)
                w = int(det[2] * width)
                h = int(det[3] * height)
                x = int(cx - (w / 2))
                y = int(cy - (h / 2))
                x, y, w, h = clip_box((x, y, w, h), width, height)

                boxes.append([x, y, w, h])
                confidences.append(confidence)
                class_ids.append(class_id)

        indices = flatten_indices(cv.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold))
        detections = []
        for idx in indices:
            cls_id = class_ids[idx]
            detections.append(Detection(labels[cls_id], confidences[idx], tuple(boxes[idx])))
        return detections

    def detect_rider_components(self, frame: np.ndarray) -> Tuple[List[Detection], List[Detection]]:
        detections = self.detect_with_dnn(
            frame,
            self.rider_net,
            self.rider_labels,
            RIDER_CONFIDENCE,
            RIDER_NMS,
            keep_labels=None,
        )
        persons = [d for d in detections if d.label == "person"]
        bikes = [d for d in detections if "bike" in d.label.lower()]
        return persons, bikes

    def match_riders(self, persons: List[Detection], bikes: List[Detection]) -> List[Dict[str, Detection]]:
        riders: List[Dict[str, Detection]] = []
        used_bikes = set()

        for person in persons:
            px, py, pw, ph = person.box
            person_bottom = py + ph
            person_center_x = px + (pw // 2)

            best_idx = -1
            best_score = 0.0
            for idx, bike in enumerate(bikes):
                if idx in used_bikes:
                    continue
                bx, by, bw, bh = bike.box
                bike_bottom = by + bh

                iou = box_iou(person.box, bike.box)
                center_in_bike = bx <= person_center_x <= (bx + bw)
                vertical_match = (by - int(0.25 * bh)) <= person_bottom <= (bike_bottom + int(0.35 * bh))

                score = iou
                if center_in_bike and vertical_match:
                    score += 0.2

                if score > best_score and (iou > 0.02 or (center_in_bike and vertical_match)):
                    best_score = score
                    best_idx = idx

            if best_idx >= 0:
                riders.append({"person": person, "bike": bikes[best_idx]})
                used_bikes.add(best_idx)

        # Fallback pairing: if strict geometric matching fails, pair by nearest centers.
        if not riders and persons and bikes:
            remaining_bikes = set(range(len(bikes)))
            for person in persons:
                if not remaining_bikes:
                    break
                px, py, pw, ph = person.box
                pcx = px + (pw / 2.0)
                pcy = py + (ph / 2.0)

                nearest_idx = -1
                nearest_dist = float("inf")
                for idx in remaining_bikes:
                    bx, by, bw, bh = bikes[idx].box
                    bcx = bx + (bw / 2.0)
                    bcy = by + (bh / 2.0)
                    dist = ((pcx - bcx) ** 2 + (pcy - bcy) ** 2) ** 0.5
                    if dist < nearest_dist:
                        nearest_dist = dist
                        nearest_idx = idx

                if nearest_idx >= 0:
                    riders.append({"person": person, "bike": bikes[nearest_idx]})
                    remaining_bikes.remove(nearest_idx)

        return riders

    def detect_helmets(self, frame: np.ndarray) -> List[Detection]:
        detections = self.detect_with_dnn(
            frame,
            self.helmet_net,
            self.helmet_labels,
            HELMET_CONFIDENCE,
            HELMET_NMS,
            keep_labels=None,
        )
        return detections

    def helmet_matches_rider(self, helmet: Detection, person: Detection) -> bool:
        px, py, pw, ph = person.box
        hx, hy, hw, hh = helmet.box

        head_box = (px, py, pw, max(1, int(ph * 0.45)))
        inter = intersection_area(helmet.box, head_box)
        if inter <= 0:
            return False

        helmet_cover_ratio = inter / float(max(1, box_area(helmet.box)))
        helmet_center_x = hx + (hw // 2)
        helmet_center_y = hy + (hh // 2)

        center_in_head = (
            head_box[0] <= helmet_center_x <= (head_box[0] + head_box[2])
            and head_box[1] <= helmet_center_y <= (head_box[1] + head_box[3])
        )

        return center_in_head and helmet_cover_ratio >= 0.25

    def riders_with_helmet(self, riders: List[Dict[str, Detection]], helmets: List[Detection]) -> set:
        matched = set()
        for rider_index, rider in enumerate(riders):
            person = rider["person"]
            for helmet in helmets:
                if self.helmet_matches_rider(helmet, person):
                    matched.add(rider_index)
                    break
        return matched

    def detect_plate(self, frame: np.ndarray) -> Optional[Detection]:
        results = self.plate_model(frame)
        out = results.pandas().xyxy[0]
        if out is None or len(out) == 0:
            return None

        out = out.sort_values("confidence", ascending=False)
        top = out.iloc[0]

        xmin = int(top["xmin"])
        ymin = int(top["ymin"])
        xmax = int(top["xmax"])
        ymax = int(top["ymax"])
        confidence = float(top["confidence"])
        if confidence < PLATE_CONFIDENCE:
            return None

        x, y, w, h = clip_box((xmin, ymin, xmax - xmin, ymax - ymin), frame.shape[1], frame.shape[0])
        return Detection("Plate", confidence, (x, y, w, h))

    def preprocess_plate_variants(self, plate_roi: np.ndarray) -> List[np.ndarray]:
        variants: List[np.ndarray] = [plate_roi]

        gray = cv.cvtColor(plate_roi, cv.COLOR_BGR2GRAY)
        gray = cv.bilateralFilter(gray, 9, 75, 75)
        variants.append(gray)

        adaptive = cv.adaptiveThreshold(
            gray,
            255,
            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv.THRESH_BINARY,
            31,
            9,
        )
        variants.append(adaptive)

        inverse = cv.bitwise_not(adaptive)
        kernel = np.ones((2, 2), np.uint8)
        morph = cv.morphologyEx(inverse, cv.MORPH_CLOSE, kernel, iterations=1)
        variants.append(morph)

        upscaled = cv.resize(morph, None, fx=2.0, fy=2.0, interpolation=cv.INTER_CUBIC)
        variants.append(upscaled)

        return variants

    def read_plate_text(self, plate_roi: np.ndarray, country: str) -> Tuple[str, float, bool, List[Tuple[str, float]]]:
        variants = self.preprocess_plate_variants(plate_roi)
        candidates: List[Tuple[str, float]] = []

        for variant in variants:
            try:
                result = self.reader.readtext(
                    variant,
                    detail=1,
                    paragraph=False,
                    allowlist=PLATE_ALLOWLIST,
                )
            except Exception:
                continue

            for _, text, confidence in result:
                cleaned = sanitize_plate_text(text)
                if len(cleaned) < 5:
                    continue
                candidates.append((cleaned, float(confidence)))

        if not candidates:
            return "", 0.0, False, []

        scored: List[Tuple[str, float, bool]] = []
        for raw, conf in candidates:
            normalized, valid, format_score = normalize_plate(raw, country)
            total_score = conf + format_score
            scored.append((normalized, total_score, valid))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_text, best_score, best_valid = scored[0]
        return best_text, best_score, best_valid, candidates

    def infer_vehicle_details(
        self,
        plate_text: str,
        country: str,
        rider_count: int,
        bike_count: int,
        valid_plate: bool,
    ) -> Dict[str, str]:
        details = {
            "country": country,
            "plate_number": plate_text if plate_text else "Not available",
            "vehicle_type": "Two-wheeler" if (rider_count > 0 or bike_count > 0) else "Unknown",
            "plate_valid": "Yes" if valid_plate else "No",
            "registration_region": "Unknown",
            "owner_name": "Not in local registry",
            "vehicle_make": "Unknown",
            "vehicle_model": "Unknown",
            "fuel_type": "Unknown",
        }

        if country == "India" and valid_plate and len(plate_text) >= 5:
            state = plate_text[:5]
            details["registration_region"] = STATE_CODE_MAP_INDIA.get(state, "Unknown state code")

        lookup_key = plate_text.upper().strip()
        if lookup_key in LOCAL_VEHICLE_REGISTRY:
            details.update(LOCAL_VEHICLE_REGISTRY[lookup_key])

        return details

    def build_challan(
        self,
        country: str,
        no_helmet_riders: int,
        plate_text: str,
    ) -> Optional[Dict[str, str]]:
        print(f"DEBUG build_challan: no_helmet_riders={no_helmet_riders}, country={country}")
        if no_helmet_riders <= 0:
            print(f"DEBUG: No violations found, returning None")
            return None

        print(f"DEBUG: Creating challan for {no_helmet_riders} no-helmet riders")
        rule = CHALLAN_RULES.get(country, CHALLAN_RULES["India"])
        unit_fine = rule["helmet_no_rider"]
        total_fine = unit_fine * no_helmet_riders

        challan = {
            "offense": "Riding without helmet",
            "offender_count": str(no_helmet_riders),
            "fine_unit": f"{rule['currency']} {unit_fine}",
            "fine_total": f"{rule['currency']} {total_fine}",
            "plate_number": plate_text if plate_text else "Not detected",
            "notes": rule["notes"],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        return challan

    def analyze(self, frame: np.ndarray, country: str) -> Dict[str, object]:
        if not self.models_loaded:
            self.load_models()

        annotated = frame.copy()

        persons, bikes = self.detect_rider_components(frame)
        riders = self.match_riders(persons, bikes)

        helmets = self.detect_helmets(frame)
        helmeted_riders = self.riders_with_helmet(riders, helmets)

        no_helmet_indices = [idx for idx in range(len(riders)) if idx not in helmeted_riders]
        
        # Debug output
        print(f"DEBUG: Persons detected: {len(persons)}")
        print(f"DEBUG: Bikes detected: {len(bikes)}")
        print(f"DEBUG: Riders matched: {len(riders)}")
        print(f"DEBUG: Helmets detected: {len(helmets)}")
        print(f"DEBUG: Helmeted riders: {helmeted_riders}")
        print(f"DEBUG: Riders without helmet indices: {no_helmet_indices}")
        print(f"DEBUG: Count of riders without helmet: {len(no_helmet_indices)}")

        plate_detection = self.detect_plate(frame)
        plate_text = ""
        plate_score = 0.0
        plate_valid = False
        ocr_candidates: List[Tuple[str, float]] = []

        if plate_detection:
            x, y, w, h = plate_detection.box
            margin_x = max(2, int(w * 0.08))
            margin_y = max(2, int(h * 0.2))
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(frame.shape[1], x + w + margin_x)
            y2 = min(frame.shape[0], y + h + margin_y)
            roi = frame[y1:y2, x1:x2]

            if roi.size > 0:
                plate_text, plate_score, plate_valid, ocr_candidates = self.read_plate_text(roi, country)

        vehicle_details = self.infer_vehicle_details(plate_text, country, len(riders), len(bikes), plate_valid)
        challan = self.build_challan(country, len(no_helmet_indices), plate_text)

        for person in persons:
            px, py, pw, ph = person.box
            cv.rectangle(annotated, (px, py), (px + pw, py + ph), (59, 130, 246), 1)

        for bike in bikes:
            bx, by, bw, bh = bike.box
            cv.rectangle(annotated, (bx, by), (bx + bw, by + bh), (245, 158, 11), 1)

        for i, rider in enumerate(riders, start=1):
            person = rider["person"]
            bike = rider["bike"]
            p_color = (0, 180, 0) if (i - 1) in helmeted_riders else (0, 0, 255)

            px, py, pw, ph = person.box
            bx, by, bw, bh = bike.box

            cv.rectangle(annotated, (px, py), (px + pw, py + ph), p_color, 2)
            cv.rectangle(annotated, (bx, by), (bx + bw, by + bh), (255, 200, 0), 2)

            tag = f"Rider {i}: {'Helmet' if (i - 1) in helmeted_riders else 'No Helmet'}"
            draw_text_label(annotated, tag, (px, max(22, py - 6)), p_color, bg_color=(15, 23, 42))

        for helmet in helmets:
            hx, hy, hw, hh = helmet.box
            cv.rectangle(annotated, (hx, hy), (hx + hw, hy + hh), (0, 255, 0), 2)
            draw_text_label(
                annotated,
                f"Helmet {helmet.confidence:.2f}",
                (hx, max(20, hy - 8)),
                (0, 255, 0),
                bg_color=(15, 23, 42),
            )

        if plate_detection:
            x, y, w, h = plate_detection.box
            cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 255), 2)
            plate_label = plate_text if plate_text else "Plate detected, OCR uncertain"
            draw_text_label(
                annotated,
                f"Plate: {plate_label}",
                (x, max(24, y - 8)),
                (255, 255, 255),
                bg_color=(15, 23, 42),
            )

        summary = f"P:{len(persons)}  B:{len(bikes)}  R:{len(riders)}  NoHelmet:{len(no_helmet_indices)}"
        draw_text_label(
            annotated,
            summary,
            (12, annotated.shape[0] - 10),
            (226, 232, 240),
            bg_color=(2, 6, 23),
            font_scale=0.55,
        )

        return {
            "annotated_frame": annotated,
            "persons": persons,
            "bikes": bikes,
            "riders": riders,
            "helmeted_riders": helmeted_riders,
            "no_helmet_count": len(no_helmet_indices),
            "plate_text": plate_text,
            "plate_score": plate_score,
            "plate_valid": plate_valid,
            "plate_detected": plate_detection is not None,
            "vehicle_details": vehicle_details,
            "challan": challan,
            "ocr_candidates": sorted(ocr_candidates, key=lambda x: x[1], reverse=True)[:5],
        }


class HelmetDetectionApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Helmet and Number Plate Enforcement")
        self.root.geometry("1320x900")
        self.root.minsize(1160, 780)
        self.root.configure(bg="#eaf0f8")

        self.app_dir = Path(__file__).resolve().parent
        self.engine = ViolationEngine(self.app_dir)

        self.selected_image: Optional[Path] = None
        self.preview_photo = None

        self.country_var = tk.StringVar(value="India")
        self.status_var = tk.StringVar(value="Ready")
        self.image_var = tk.StringVar(value="No image selected")

        self._build_styles()
        self._build_ui()

    def _build_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Header.TLabel", font=("Segoe UI", 23, "bold"), foreground="#0b1220", background="#eaf0f8")
        style.configure("SubHeader.TLabel", font=("Segoe UI", 10), foreground="#334155", background="#eaf0f8")
        style.configure("Card.TFrame", background="#ffffff", relief="flat", borderwidth=1)
        style.configure("CardTitle.TLabel", font=("Segoe UI", 12, "bold"), foreground="#111827", background="#ffffff")
        style.configure("CardBody.TLabel", font=("Segoe UI", 10), foreground="#1f2937", background="#ffffff")
        style.configure("TSeparator", background="#d1d9e6")
        style.configure("TCombobox", fieldbackground="#f8fafc", background="#f8fafc", arrowsize=16)
        style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            foreground="#ffffff",
            background="#0b5fff",
            borderwidth=0,
            focuscolor="#0b5fff",
            padding=(10, 8),
        )
        style.map(
            "Primary.TButton",
            background=[("active", "#004ed0"), ("pressed", "#003aa3")],
            foreground=[("disabled", "#cbd5e1"), ("!disabled", "#ffffff")],
        )

    def _build_ui(self) -> None:
        self.root.grid_rowconfigure(0, weight=0)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=1)

        self.root.grid_columnconfigure(0, weight=0)
        self.root.grid_columnconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="Card.TFrame", padding=(24, 18))
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(14, 8))
        header.grid_columnconfigure(0, weight=1)

        ttk.Label(header, text="Helmet Violation and Plate Intelligence", style="Header.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="rider-helmet validation, vehicle details, and country-based challan summary.",
            style="SubHeader.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        controls = ttk.Frame(self.root, style="Card.TFrame", padding=18)
        controls.grid(row=1, column=0, sticky="nsew", padx=(14, 8), pady=(0, 8))
        controls.grid_columnconfigure(0, weight=1)

        ttk.Label(controls, text="Controls", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))

        ttk.Label(controls, text="Country", style="CardBody.TLabel").grid(row=1, column=0, sticky="w")
        country_combo = ttk.Combobox(
            controls,
            textvariable=self.country_var,
            values=list(CHALLAN_RULES.keys()),
            state="readonly",
            width=26,
        )
        country_combo.grid(row=2, column=0, sticky="ew", pady=(4, 10))

        ttk.Button(controls, text="Load Models", command=self.load_models, style="Primary.TButton").grid(
            row=3, column=0, sticky="ew", pady=4
        )
        ttk.Button(controls, text="Select Image", command=self.select_image).grid(row=4, column=0, sticky="ew", pady=4)
        ttk.Button(controls, text="Analyze", command=self.analyze_image, style="Primary.TButton").grid(
            row=5, column=0, sticky="ew", pady=4
        )
        ttk.Button(controls, text="Clear", command=self.clear_results).grid(row=6, column=0, sticky="ew", pady=4)

        ttk.Separator(controls, orient="horizontal").grid(row=7, column=0, sticky="ew", pady=10)

        ttk.Label(controls, text="Selected Image", style="CardBody.TLabel").grid(row=8, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.image_var, style="CardBody.TLabel", wraplength=260).grid(
            row=9, column=0, sticky="w", pady=(4, 10)
        )

        ttk.Label(controls, text="Status", style="CardBody.TLabel").grid(row=10, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.status_var, style="CardBody.TLabel", wraplength=260).grid(
            row=11, column=0, sticky="w", pady=(4, 0)
        )

        preview_card = ttk.Frame(self.root, style="Card.TFrame", padding=18)
        preview_card.grid(row=1, column=1, sticky="nsew", padx=(8, 14), pady=(0, 8))
        preview_card.grid_rowconfigure(1, weight=1)
        preview_card.grid_columnconfigure(0, weight=1)

        ttk.Label(preview_card, text="Detection Preview", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.preview_label = tk.Label(
            preview_card,
            anchor="center",
            bg="#020617",   # darker clean bg
            fg="#94a3b8",
            text="No preview",
            font=("Segoe UI", 13, "bold"),
            bd=0,
            highlightthickness=0,
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        log_card = ttk.Frame(self.root, style="Card.TFrame", padding=18)
        log_card.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=14, pady=(0, 14))
        log_card.grid_rowconfigure(1, weight=1)
        log_card.grid_columnconfigure(0, weight=1)

        ttk.Label(log_card, text="Workflow Output", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.log_text = ScrolledText(
            log_card,
            height=12,
            font=("Consolas", 10),
            bg="#020617",
            fg="#e5e7eb",
            insertbackground="white",
            borderwidth=0,
        )
        self.log_text.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.log_text.configure(state="disabled")

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def log(self, text: str, clear: bool = False) -> None:
        self.log_text.configure(state="normal")
        if clear:
            self.log_text.delete("1.0", tk.END)
        self.log_text.insert(tk.END, text + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def load_models(self) -> None:
        try:
            self.set_status("Loading models. This may take a minute on first run...")
            start = time.time()
            self.engine.load_models()
            elapsed = time.time() - start
            self.set_status(f"Models loaded in {elapsed:.1f}s")
            self.log("Models loaded successfully.", clear=False)
            for key, value in self.engine.last_loaded_paths.items():
                self.log(f"  {key}: {value}")
            print("RIDER LABELS:", self.engine.rider_labels)
            print("HELMET LABELS:", self.engine.helmet_labels)
        except Exception as exc:
            self.set_status("Model loading failed")
            messagebox.showerror("Model Load Error", str(exc))
        


    def select_image(self) -> None:
        start_dir = self.app_dir / "bikes"
        path = filedialog.askopenfilename(
            initialdir=str(start_dir if start_dir.exists() else self.app_dir),
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")],
        )
        if not path:
            return

        self.selected_image = Path(path)
        self.image_var.set(str(self.selected_image))
        self.set_status("Image selected")

        frame = cv.imread(str(self.selected_image))
        if frame is not None:
            self.show_preview(frame)
            self.log(f"Selected image: {self.selected_image}", clear=False)

    def show_preview(self, frame: np.ndarray) -> None:
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        self.preview_label.update_idletasks()

        container_w = self.preview_label.winfo_width()
        container_h = self.preview_label.winfo_height()

        if container_w < 100 or container_h < 100:
            container_w, container_h = 900, 500

        # Maintain aspect ratio properly
        image.thumbnail((container_w - 20, container_h - 20), Image.Resampling.LANCZOS)

        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_photo, text="")

    def analyze_image(self) -> None:
        if self.selected_image is None:
            messagebox.showwarning("Input Required", "Select an image first.")
            return

        frame = cv.imread(str(self.selected_image))
        if frame is None:
            messagebox.showerror("Input Error", "Unable to read selected image.")
            return

        try:
            self.set_status("Running detection workflow...")
            start = time.time()
            result = self.engine.analyze(frame, self.country_var.get())
            # 🔥 identify image name
            image_name = self.selected_image.stem  # e.g., bikes1 → IMG_1

            # map image → email
            user = USER_DATABASE.get(image_name)

            if user:
                # check violation
                if result["no_helmet_count"] > 0 or not result["plate_detected"]:
                    message = build_violation_message(result)

                    send_violation_email(
                        user["email"],
                        "Traffic Violation Alert 🚨",
                        message,
                        self.root
                    )
            elapsed = time.time() - start

            self.show_preview(result["annotated_frame"])
            self.render_report(result, elapsed)
            self.set_status(f"Analysis complete in {elapsed:.2f}s")
        except Exception as exc:
            self.set_status("Analysis failed")
            messagebox.showerror("Analysis Error", str(exc))

    def render_report(self, result: Dict[str, object], elapsed: float) -> None:
        self.log("", clear=True)
        self.log(f"Image: {self.selected_image}")
        self.log(f"Country: {self.country_var.get()}")
        self.log(f"Execution time: {elapsed:.2f} sec")
        self.log("-" * 70)

        rider_count = len(result["riders"])
        helmeted_count = len(result["helmeted_riders"])
        no_helmet_count = result["no_helmet_count"]
        person_count = len(result["persons"])
        bike_count = len(result["bikes"])

        self.log(f"Persons detected: {person_count}")
        self.log(f"Bikes detected: {bike_count}")
        self.log(f"Riders detected: {rider_count}")
        self.log(f"Riders with helmet: {helmeted_count}")
        self.log(f"No-helmet violations: {no_helmet_count}")
        if rider_count == 0 and (person_count > 0 or bike_count > 0):
            self.log("Note: Person/bike detected but rider pairing was weak in this frame.")

        if result["plate_detected"]:
            plate_line = result["plate_text"] if result["plate_text"] else "Detected, OCR uncertain"
            self.log(f"Plate detection: {plate_line}")
            self.log(f"Plate confidence score: {result['plate_score']:.2f}")
            self.log(f"Plate format valid: {'Yes' if result['plate_valid'] else 'No'}")
        else:
            self.log("Plate detection: Not found")

        self.log("-" * 70)
        self.log("Vehicle Details")
        details = result["vehicle_details"]
        for key in [
            "country",
            "plate_number",
            "vehicle_type",
            "plate_valid",
            "registration_region",
            "owner_name",
            "vehicle_make",
            "vehicle_model",
            "fuel_type",
        ]:
            self.log(f"  {key}: {details.get(key, 'Unknown')}")

        self.log("-" * 70)
        challan = result["challan"]
        if challan:
            self.log("Challan")
            self.log(f"  offense: {challan['offense']}")
            self.log(f"  offender_count: {challan['offender_count']}")
            self.log(f"  fine_unit: {challan['fine_unit']}")
            self.log(f"  fine_total: {challan['fine_total']}")
            self.log(f"  plate_number: {challan['plate_number']}")
            self.log(f"  timestamp: {challan['timestamp']}")
            self.log(f"  notes: {challan['notes']}")
        else:
            self.log("Challan: Not generated (no no-helmet violation)")

        candidates = result["ocr_candidates"]
        if candidates:
            self.log("-" * 70)
            self.log("Top OCR candidates")
            for text, conf in candidates:
                self.log(f"  {text}: {conf:.2f}")

    def clear_results(self) -> None:
        self.selected_image = None
        self.image_var.set("No image selected")
        self.status_var.set("Ready")
        self.preview_label.configure(image="", text="No preview")
        self.preview_photo = None
        self.log("", clear=True)


def main() -> None:
    root = tk.Tk()
    app = HelmetDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
