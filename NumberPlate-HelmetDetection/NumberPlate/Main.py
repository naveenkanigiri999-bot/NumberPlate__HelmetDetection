
import os
import re
import subprocess
import sys
import time
import pathlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import cv2
import easyocr
import numpy as np
import torch
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

if os.name == "nt":
    pathlib.PosixPath = pathlib.WindowsPath

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
    },
    "DL8CAF5031": {
        "owner_name": "Demo Owner 2",
        "vehicle_make": "Bajaj",
        "vehicle_model": "Pulsar",
        "fuel_type": "Petrol",
    },
}

USER_DATABASE = {
        "0": {"email": "user0@gmail.com"},
        "1": {"email": "user1@gmail.com"},
        "2": {"email": "user2@gmail.com"},
        "3": {"email": "user3@gmail.com"},
        "4": {"email": "user4@gmail.com"},
        "5": {"email": "user5@gmail.com"},
        "6": {"email": "user6@gmail.com"},
        "7": {"email": "user7@gmail.com"},
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
        "violations": {
            "No Helmet": 1000,
            "Red Light Jump": 1000,
            "No Insurance": 2000,
        },
        "notes": "Default values. Update as per latest local/state law before production use.",
    },
    "United States": {
        "currency": "USD",
        "violations": {
            "No Helmet": 150,
            "Red Light Jump": 100,
            "No Insurance": 500,
        },
        "notes": "Default values. Update per state law before production use.",
    },
    "United Kingdom": {
        "currency": "GBP",
        "violations": {
            "No Helmet": 100,
            "Red Light Jump": 100,
            "No Insurance": 300,
        },
        "notes": "Default values. Update per latest guidance before production use.",
    },
}


def first_existing_path(paths: List[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


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
            if not regex.match(fixed):
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

def send_plate_email(to_email, subject, body, root=None):
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

        if root:
            messagebox.showinfo("Email Sent", f"Sent to {to_email}")

    except Exception as e:
        print("❌ Email failed:", e)

        if root:
            messagebox.showerror("Email Failed", str(e))

class NumberPlateApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Number Plate Intelligence")
        self.root.geometry("1220x860")
        self.root.minsize(1040, 740)
        self.root.configure(bg="#f1f5f9")

        self.base_dir = Path(__file__).resolve().parent
        self.model = None
        self.reader = easyocr.Reader(["en"], gpu=False)
        self.selected_image: Optional[Path] = None
        self.preview_photo = None

        self.country_var = tk.StringVar(value="India")
        self.violation_var = tk.StringVar(value="No Helmet")
        self.status_var = tk.StringVar(value="Ready")
        self.image_var = tk.StringVar(value="No image selected")

        self._build_styles()
        self._build_ui()

    def _build_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")

        style.configure("Header.TLabel", font=("Segoe UI", 20, "bold"), foreground="#0f172a", background="#f1f5f9")
        style.configure("SubHeader.TLabel", font=("Segoe UI", 10), foreground="#475569", background="#f1f5f9")
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("CardTitle.TLabel", font=("Segoe UI", 11, "bold"), foreground="#1e293b", background="#ffffff")
        style.configure("CardBody.TLabel", font=("Segoe UI", 10), foreground="#334155", background="#ffffff")

    def _build_ui(self) -> None:
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        header = ttk.Frame(self.root, style="Card.TFrame", padding=(18, 14))
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=14, pady=(14, 8))
        header.grid_columnconfigure(0, weight=1)

        ttk.Label(header, text="Number Plate Detection and Details", style="Header.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="vehicle details extraction, and country-based challan calculation.",
            style="SubHeader.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        controls = ttk.Frame(self.root, style="Card.TFrame", padding=14)
        controls.grid(row=1, column=0, sticky="nsew", padx=(16, 10), pady=(0, 8))
        controls.grid_columnconfigure(0, weight=1)

        ttk.Label(controls, text="Controls", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))

        ttk.Label(controls, text="Country", style="CardBody.TLabel").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.country_var,
            values=list(CHALLAN_RULES.keys()),
            state="readonly",
        ).grid(row=2, column=0, sticky="ew", pady=(4, 8))


        ttk.Button(controls, text="Load Model", command=self.load_model).grid(row=3, column=0, sticky="ew", pady=4)
        ttk.Button(controls, text="Select Image", command=self.select_image).grid(row=4, column=0, sticky="ew", pady=4)
        ttk.Button(controls, text="Detect Plate", command=self.detect_plate).grid(row=5, column=0, sticky="ew", pady=4)
        ttk.Button(controls, text="Clear", command=self.clear).grid(row=6, column=0, sticky="ew", pady=4)
        ttk.Separator(controls, orient="horizontal").grid(row=9, column=0, sticky="ew", pady=10)

        ttk.Label(controls, text="Selected Image", style="CardBody.TLabel").grid(row=10, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.image_var, style="CardBody.TLabel", wraplength=260).grid(
            row=11, column=0, sticky="w", pady=(4, 10)
        )

        ttk.Label(controls, text="Status", style="CardBody.TLabel").grid(row=12, column=0, sticky="w")
        ttk.Label(controls, textvariable=self.status_var, style="CardBody.TLabel", wraplength=260).grid(
            row=13, column=0, sticky="w", pady=(4, 0)
        )

        preview_card = ttk.Frame(self.root, style="Card.TFrame", padding=14)
        preview_card.grid(row=1, column=1, sticky="nsew", padx=(10, 16), pady=(0, 8))
        preview_card.grid_rowconfigure(1, weight=1)
        preview_card.grid_columnconfigure(0, weight=1)

        ttk.Label(preview_card, text="Preview", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.preview_label = tk.Label(
            preview_card,
            bg="#020617",
            fg="#94a3b8",
            text="Preview",
            font=("Segoe UI", 12),
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew", pady=(10, 0))

        log_card = ttk.Frame(self.root, style="Card.TFrame", padding=14)
        log_card.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=14, pady=(0, 14))
        log_card.grid_rowconfigure(1, weight=1)
        log_card.grid_columnconfigure(0, weight=1)

        ttk.Label(log_card, text="Output", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w")
        self.log_text = ScrolledText(log_card, height=12, font=("Consolas", 10), bg="#020617", fg="#e2e8f0")
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

    def load_model(self) -> None:
        try:
            model_path = first_existing_path(
                [
                    self.base_dir / "model" / "best.pt",
                    self.base_dir / "NumberPlate" / "model" / "best.pt",
                ]
            )
            yolo_repo = first_existing_path(
                [
                    self.base_dir / "yolov5",
                    self.base_dir / "NumberPlate" / "yolov5",
                ]
            )

            if model_path is None or yolo_repo is None:
                raise FileNotFoundError("Could not find local YOLOv5 repository or model/best.pt")

            self.set_status("Loading model...")
            ensure_pkg_resources_available()
            self.model = torch.hub.load(str(yolo_repo), "custom", path=str(model_path), source="local", force_reload=False)
            self.model.conf = PLATE_CONFIDENCE
            self.model.iou = 0.45
            self.set_status("Model loaded")
            self.log(f"Model loaded: {model_path}")
        except Exception as exc:
            self.set_status("Model load failed")
            messagebox.showerror("Model Load Error", str(exc))

    def select_image(self) -> None:
        path = filedialog.askopenfilename(
            initialdir=str(self.base_dir / "testImages"),
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp"), ("All Files", "*.*")],
        )
        if not path:
            return

        self.selected_image = Path(path)
        self.image_var.set(str(self.selected_image))
        self.set_status("Image selected")

        frame = cv2.imread(str(self.selected_image))
        if frame is not None:
            self.show_preview(frame)
            self.log(f"Selected image: {self.selected_image}")

    def show_preview(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        max_w, max_h = 840, 420
        scale = min(max_w / image.width, max_h / image.height, 1.0)
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)

        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_photo)

    def detect_best_plate_box(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int, float]]:
        if self.model is None:
            return None

        results = self.model(frame)
        out = results.pandas().xyxy[0]
        if out is None or len(out) == 0:
            return None

        out = out.sort_values("confidence", ascending=False)
        top = out.iloc[0]
        conf = float(top["confidence"])
        if conf < PLATE_CONFIDENCE:
            return None

        xmin = max(0, int(top["xmin"]))
        ymin = max(0, int(top["ymin"]))
        xmax = min(frame.shape[1], int(top["xmax"]))
        ymax = min(frame.shape[0], int(top["ymax"]))

        if xmax <= xmin or ymax <= ymin:
            return None
        return xmin, ymin, xmax, ymax, conf

    def preprocess_plate_variants(self, roi: np.ndarray) -> List[np.ndarray]:
        variants = [roi]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        variants.append(gray)

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )
        variants.append(adaptive)

        inverse = cv2.bitwise_not(adaptive)
        morph = cv2.morphologyEx(inverse, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8), iterations=1)
        variants.append(morph)
        variants.append(cv2.resize(morph, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC))
        return variants

    def read_plate(self, roi: np.ndarray, country: str) -> Tuple[str, float, bool, List[Tuple[str, float]]]:
        candidates: List[Tuple[str, float]] = []
        for variant in self.preprocess_plate_variants(roi):
            try:
                result = self.reader.readtext(
                    variant,
                    detail=1,
                    paragraph=False,
                    allowlist=PLATE_ALLOWLIST,
                )
            except Exception:
                continue

            for _, text, conf in result:
                cleaned = sanitize_plate_text(text)
                if len(cleaned) < 5:
                    continue
                candidates.append((cleaned, float(conf)))

        if not candidates:
            return "", 0.0, False, []

        scored = []
        for raw, conf in candidates:
            normalized, valid, format_score = normalize_plate(raw, country)
            scored.append((normalized, conf + format_score, valid))

        scored.sort(key=lambda x: x[1], reverse=True)
        best_text, best_score, best_valid = scored[0]
        return best_text, best_score, best_valid, sorted(candidates, key=lambda x: x[1], reverse=True)[:5]

    def infer_vehicle_details(self, plate: str, country: str, valid: bool) -> Dict[str, str]:
        details = {
            "country": country,
            "plate_number": plate if plate else "Not available",
            "plate_valid": "Yes" if valid else "No",
            "vehicle_type": "Unknown",
            "registration_region": "Unknown",
            "owner_name": "Not in local registry",
            "vehicle_make": "Unknown",
            "vehicle_model": "Unknown",
            "fuel_type": "Unknown",
        }

        if country == "India" and valid and len(plate) >= 2:
            details["registration_region"] = STATE_CODE_MAP_INDIA.get(plate[:2], "Unknown state code")

        lookup = plate.upper().strip()
        if lookup in LOCAL_VEHICLE_REGISTRY:
            details.update(LOCAL_VEHICLE_REGISTRY[lookup])

        return details

    def build_challan(self, country: str, violation: str, plate: str) -> Dict[str, str]:
        rule = CHALLAN_RULES.get(country, CHALLAN_RULES["India"])
        amount = rule["violations"].get("No Helmet", 1000)

        offender_count = 1
        fine_unit = f"{rule['currency']} {amount}"
        fine_total = f"{rule['currency']} {amount * offender_count}"

        return {
            "offense": violation,
            "offender_count": offender_count,
            "fine_unit": fine_unit,
            "fine_total": fine_total,
            "plate_number": plate if plate else "Not detected",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "notes": rule["notes"],
        }

    def detect_plate(self) -> None:
        if self.model is None:
            self.load_model()
            if self.model is None:
                return

        if self.selected_image is None:
            messagebox.showwarning("Input Required", "Select an image first.")
            return

        frame = cv2.imread(str(self.selected_image))
        if frame is None:
            messagebox.showerror("Input Error", "Unable to read selected image.")
            return

        self.set_status("Detecting number plate...")
        start = time.time()

        box = self.detect_best_plate_box(frame)

        # ❌ NO PLATE FOUND
        if box is None:
            self.show_preview(frame)

            elapsed = time.time() - start
            self.log("")
            self.log(f"Image: {self.selected_image}")
            self.log(f"Execution time: {elapsed:.2f}s")
            self.log("-" * 40)
            self.log("❌ No Number Plate Detected")
            self.log("🚨 Alert No Number Plate Vehicle is Passing")
            self.log("-" * 40)

            # 🔥 EMAIL
            image_name = self.selected_image.stem
            user = USER_DATABASE.get(image_name)

            if user:
                challan = self.build_challan(
                    self.country_var.get(),
                    "No Number Plate",
                    "Not detected"
                )

                message = f"""
    🚨 Traffic Violation Alert

    Violation: No Number Plate
    Plate: {plate_text if plate_text else "Not detected"}
    Valid: {"Yes" if valid_plate else "No"}

    Fine: {challan['fine_total']}
    Time: {challan['timestamp']}
    """

                send_plate_email(
                    user["email"],
                    "No Number Plate Violation 🚨",
                    message,
                    self.root
                )

            self.set_status("No plate detected")
            return

        xmin, ymin, xmax, ymax, det_conf = box

        margin_x = max(2, int((xmax - xmin) * 0.08))
        margin_y = max(2, int((ymax - ymin) * 0.2))

        x1 = max(0, xmin - margin_x)
        y1 = max(0, ymin - margin_y)
        x2 = min(frame.shape[1], xmax + margin_x)
        y2 = min(frame.shape[0], ymax + margin_y)

        roi = frame[y1:y2, x1:x2]

        plate_text, plate_score, valid_plate, _ = self.read_plate(
            roi, self.country_var.get()
        )

        # 🔥 EMAIL + CHALLAN ONLY IF INVALID
        image_name = self.selected_image.stem
        user = USER_DATABASE.get(image_name)

        if user and (not plate_text or not valid_plate):

            challan = self.build_challan(
                self.country_var.get(),
                "No Number Plate",
                plate_text if plate_text else "Not detected"
            )

            self.log("🚨 Challan")
            self.log(f"  offense: {challan['offense']}")
            self.log(f"  offender_count: {challan['offender_count']}")
            self.log(f"  fine_unit: {challan['fine_unit']}")
            self.log(f"  fine_total: {challan['fine_total']}")
            self.log(f"  plate_number: {challan['plate_number']}")
            self.log(f"  timestamp: {challan['timestamp']}")
            self.log(f"  notes: {challan['notes']}")

            plate_text = "Not detected"
            valid_plate = False

            message = f"""
            🚨 Traffic Violation Alert

            Violation: No Number Plate
            Plate: {plate_text}
            Valid: {"Yes" if valid_plate else "No"}

            Fine: {challan['fine_total']}
            Time: {challan['timestamp']}
            """

            send_plate_email(
                user["email"],
                "Plate Violation 🚨",
                message,
                self.root
            )

        # 🔥 DRAW RESULT
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 255), 2)

        if plate_text:
            label = f"Plate: {plate_text}"
        else:
            label = "No Plate Detected"

        cv2.putText(
            frame,
            label,
            (xmin, max(20, ymin - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        self.show_preview(frame)

        # 🔥 CLEAN OUTPUT ONLY
        elapsed = time.time() - start

        self.log("")
        self.log(f"Image: {self.selected_image}")
        self.log(f"Execution time: {elapsed:.2f}s")
        self.log("-" * 40)

        if plate_text and valid_plate:
            self.log("✅ Number Plate Detected")
            self.log(f"Plate: {plate_text}")
        else:
            self.log("❌ No Valid Number Plate Detected")
            self.log("🚨 Challan Generated")

            # 🔥 ADD CHALLAN PRICE
            challan = self.build_challan(
                self.country_var.get(),
                "No Number Plate",
                plate_text if plate_text else "Not detected"
            )
            self.log(f"🕒 Time: {challan['timestamp']}")




    def clear(self) -> None:
        self.selected_image = None
        self.image_var.set("No image selected")
        self.status_var.set("Ready")
        self.preview_label.configure(image="")
        self.preview_photo = None
        self.log("")


def main() -> None:
    root = tk.Tk()
    app = NumberPlateApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
