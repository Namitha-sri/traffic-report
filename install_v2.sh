#!/bin/bash
# Installer for Smart Video Analytics v2
# Usage: bash install_v2.sh
#
# Run this from inside ~/traffic-report-v2/ on the VM.
# It will back up existing .py files and replace them with the v2 versions.

set -e

if [ ! -f "Dockerfile" ]; then
    echo "ERROR: Dockerfile not found. Run this from ~/traffic-report-v2/"
    exit 1
fi

BACKUP_DIR="backup-$(date +%Y%m%d-%H%M%S)"
echo "Creating backup in $BACKUP_DIR/"
mkdir -p "$BACKUP_DIR"
cp *.py "$BACKUP_DIR/" 2>/dev/null || true

echo "Writing v2 files..."


echo '  -> config.py'
cat > config.py << 'CLAUDE_FILE_EOF'
import os

# API Configuration
YOLO_ENDPOINT = os.getenv("YOLO_ENDPOINT", "").rstrip('/') + '/predict'
YOLO_API_KEY = os.getenv("YOLO_API_KEY", "")

QWEN_ENDPOINT = os.getenv("QWEN_ENDPOINT", "")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")

# Database Configuration
DB_URL = os.getenv("DB_URL", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "")

# Full COCO class set that YOLOv8 can detect (80 classes)
# The use case picks which of these to filter on.
COCO_CLASSES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard",
    37: "surfboard", 38: "tennis racket", 39: "bottle", 40: "wine glass",
    41: "cup", 42: "fork", 43: "knife", 44: "spoon", 45: "bowl",
    46: "banana", 47: "apple", 48: "sandwich", 49: "orange", 50: "broccoli",
    51: "carrot", 52: "hot dog", 53: "pizza", 54: "donut", 55: "cake",
    56: "chair", 57: "couch", 58: "potted plant", 59: "bed",
    60: "dining table", 61: "toilet", 62: "tv", 63: "laptop", 64: "mouse",
    65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush"
}

# Reverse lookup — name to ID — useful when the UI sends class names
COCO_NAME_TO_ID = {name: cid for cid, name in COCO_CLASSES.items()}


# Built-in use cases that ship with the app.
# Each use case specifies:
#   - key:       unique identifier used in DB and dropdown
#   - name:      what the user sees in the dropdown
#   - icon:      emoji shown in the dropdown
#   - description: short text shown in the preset card
#   - classes:   list of COCO class IDs to keep (drops everything else)
#   - prompt:    Qwen prompt template, must contain {detections}
BUILT_IN_USE_CASES = {
    "traffic": {
        "key": "traffic",
        "name": "Traffic Monitoring",
        "icon": "🚦",
        "description": "Monitor road traffic, detect vehicles, analyze flow and congestion",
        "classes": [0, 1, 2, 3, 5, 7],  # person, bicycle, car, motorcycle, bus, truck
        "prompt": """
Analyze this traffic scene image with the following detected objects: {detections}

Please provide a concise analysis in this format:

**Traffic Status:** [NORMAL/ABNORMAL]
**Traffic Flow:** [Light/Moderate/Heavy]
**Incident Analysis:** [If abnormal, describe any accidents, delays, or unusual situations and specify how many objects/vehicles are involved]
**Safety Assessment:** [Any safety concerns or violations observed]
**Recommendations:** [Brief optimization suggestions]

Keep each section to 1-2 sentences maximum.
"""
    },
    "hospital": {
        "key": "hospital",
        "name": "Hospital Emergency Gate",
        "icon": "🏥",
        "description": "Monitor ambulance access, emergency response readiness, patient flow at hospital entrances",
        "classes": [0, 2, 7],  # person, car, truck (ambulances usually classify as truck/car)
        "prompt": """
Analyze this hospital emergency entrance scene with the following detected objects: {detections}

Please provide a concise analysis in this format:

**Traffic Status:** [NORMAL/URGENT] (URGENT if ambulance or high patient volume)
**Traffic Flow:** [Light/Moderate/Heavy]
**Incident Analysis:** [Describe what is happening — ambulance arriving, patients being received, staff responding, or normal operation. Count people and vehicles involved.]
**Safety Assessment:** [Is the emergency entrance clear? Any blockages? Are medical staff visibly responding?]
**Recommendations:** [Suggestions to improve emergency response readiness or entrance accessibility]

Keep each section to 1-2 sentences maximum.
"""
    },
    "workshop": {
        "key": "workshop",
        "name": "Workshop / Factory Floor",
        "icon": "🏭",
        "description": "Monitor worker safety, PPE compliance, equipment activity on factory floors",
        "classes": [0],  # person — YOLOv8 doesn't know robots or CNC machines
        "prompt": """
Analyze this workshop or factory floor scene with the following detected objects: {detections}

Note: YOLOv8 detects people well but may not identify specialized industrial equipment (robots, CNC machines, conveyor belts) — use your own visual understanding to identify equipment in the scene.

Please provide a concise analysis in this format:

**Traffic Status:** [SAFE/UNSAFE]
**Traffic Flow:** [Low activity / Moderate activity / High activity]
**Incident Analysis:** [Describe what work is happening, what equipment is visible, and how many workers are present. Identify robots, machines, or tools you can see.]
**Safety Assessment:** [Are workers wearing visible PPE (helmets, vests, gloves)? Is there safe spacing between workers and equipment? Any visible hazards?]
**Recommendations:** [Safety or efficiency improvements]

Keep each section to 1-2 sentences maximum.
"""
    }
}


# Default use case key used at startup
DEFAULT_USE_CASE = "traffic"

CLAUDE_FILE_EOF

echo '  -> use_case_service.py'
cat > use_case_service.py << 'CLAUDE_FILE_EOF'
import os
import psycopg2
import json
from config import BUILT_IN_USE_CASES, COCO_CLASSES


class UseCaseService:
    """
    Manages use cases: built-in presets plus custom ones stored in Postgres.

    A use case is just a named bundle of:
        - which COCO class IDs YOLOv8 should filter on
        - what prompt Qwen should be asked

    Built-ins live in config.py (read-only). Custom ones live in a
    `use_cases` table in Postgres so they survive container rebuilds.
    """

    def __init__(self):
        self.db_url = os.getenv("DB_URL", "")
        self.db_user = os.getenv("DB_USER", "")
        self.db_password = os.getenv("DB_PASSWORD", "")
        self.db_name = os.getenv("DB_NAME", "")

    def _connect(self):
        """Open a fresh Postgres connection. Caller is responsible for closing."""
        return psycopg2.connect(
            host=self.db_url,
            database=self.db_name,
            user=self.db_user,
            password=self.db_password
        )

    def create_table_if_not_exists(self):
        """Create the use_cases table if it's not already there."""
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS use_cases (
                    id SERIAL PRIMARY KEY,
                    key VARCHAR(100) UNIQUE NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    icon VARCHAR(20),
                    description TEXT,
                    classes JSON NOT NULL,
                    prompt TEXT NOT NULL,
                    is_builtin BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
            cursor.close()
            conn.close()
            return True, "use_cases table ready"
        except Exception as e:
            return False, f"Failed to create use_cases table: {str(e)}"

    def get_all_use_cases(self):
        """
        Return every use case available to the user: built-ins first,
        then custom ones from the DB.

        Returns a list of dicts with keys: key, name, icon, description,
        classes, prompt, is_builtin.
        """
        all_cases = []

        # Built-ins from config.py
        for uc in BUILT_IN_USE_CASES.values():
            all_cases.append({**uc, "is_builtin": True})

        # Custom ones from the database
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT key, name, icon, description, classes, prompt
                FROM use_cases
                WHERE is_builtin = FALSE
                ORDER BY created_at ASC
            """)
            for row in cursor.fetchall():
                key, name, icon, description, classes, prompt = row
                all_cases.append({
                    "key": key,
                    "name": name,
                    "icon": icon or "⚙️",
                    "description": description or "",
                    # `classes` comes back from Postgres JSON column as a list
                    "classes": classes if isinstance(classes, list) else json.loads(classes),
                    "prompt": prompt,
                    "is_builtin": False
                })
            cursor.close()
            conn.close()
        except Exception as e:
            # If DB is unreachable, return just the built-ins and log the issue.
            print(f"UseCaseService: could not fetch custom use cases: {e}")

        return all_cases

    def get_use_case(self, key):
        """Fetch a single use case by its key. Returns None if not found."""
        for uc in self.get_all_use_cases():
            if uc["key"] == key:
                return uc
        return None

    def save_custom_use_case(self, key, name, icon, description, classes, prompt):
        """
        Save or update a user-defined use case.

        Validation rules:
          - key must be lowercase, no spaces (use underscores)
          - key cannot collide with a built-in
          - classes must be a non-empty list of valid COCO IDs
          - prompt must contain {detections} so we can inject detection summary
        """
        # Guard rails
        if key in BUILT_IN_USE_CASES:
            return False, f"'{key}' is a built-in use case and cannot be overwritten"

        if not key or not key.replace("_", "").isalnum() or key != key.lower():
            return False, "Key must be lowercase letters, numbers, and underscores only"

        if not classes or not isinstance(classes, list):
            return False, "You must select at least one object class"

        for c in classes:
            if c not in COCO_CLASSES:
                return False, f"Invalid class ID: {c}"

        if "{detections}" not in prompt:
            return False, "Prompt must include {detections} where the detected objects list should appear"

        try:
            conn = self._connect()
            cursor = conn.cursor()
            # Upsert: if the key already exists (custom), update it
            cursor.execute("""
                INSERT INTO use_cases (key, name, icon, description, classes, prompt, is_builtin)
                VALUES (%s, %s, %s, %s, %s, %s, FALSE)
                ON CONFLICT (key) DO UPDATE SET
                    name = EXCLUDED.name,
                    icon = EXCLUDED.icon,
                    description = EXCLUDED.description,
                    classes = EXCLUDED.classes,
                    prompt = EXCLUDED.prompt
            """, (key, name, icon, description, json.dumps(classes), prompt))
            conn.commit()
            cursor.close()
            conn.close()
            return True, f"Use case '{name}' saved successfully"
        except Exception as e:
            return False, f"Failed to save use case: {str(e)}"

    def delete_custom_use_case(self, key):
        """Delete a custom use case. Built-ins cannot be deleted."""
        if key in BUILT_IN_USE_CASES:
            return False, "Built-in use cases cannot be deleted"
        try:
            conn = self._connect()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM use_cases WHERE key = %s AND is_builtin = FALSE", (key,))
            rowcount = cursor.rowcount
            conn.commit()
            cursor.close()
            conn.close()
            if rowcount == 0:
                return False, f"No custom use case found with key '{key}'"
            return True, f"Use case '{key}' deleted"
        except Exception as e:
            return False, f"Failed to delete use case: {str(e)}"

CLAUDE_FILE_EOF

echo '  -> detection_service.py'
cat > detection_service.py << 'CLAUDE_FILE_EOF'
import requests
import base64
import json
import cv2
import numpy as np
from PIL import Image
import io
import os
from config import COCO_CLASSES


class DetectionService:
    def __init__(self):
        yolo_base = os.getenv("YOLO_ENDPOINT", "")
        self.endpoint = yolo_base.rstrip('/') + '/predict' if yolo_base else ""
        self.api_key = os.getenv("YOLO_API_KEY", "")

    def validate_endpoint(self):
        """Check if the endpoint is reachable"""
        if not self.endpoint:
            return False, "No YOLO endpoint configured"
        try:
            response = requests.head(self.endpoint.replace('/predict', ''), timeout=5)
            return True, "Endpoint reachable"
        except requests.exceptions.Timeout:
            return False, "Endpoint timeout - service may be slow or unreachable"
        except requests.exceptions.ConnectionError:
            return False, "Connection failed - check if endpoint URL is correct"
        except Exception as e:
            return False, f"Endpoint validation failed: {str(e)}"

    def encode_image(self, image):
        """Convert image to base64 string"""
        if isinstance(image, str):
            with open(image, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        elif isinstance(image, np.ndarray):
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        elif isinstance(image, Image.Image):
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def detect_objects(self, image, classes_filter=None):
        """
        Send image to YOLO endpoint and get detections.

        classes_filter: list of COCO class IDs to keep. If None, keeps all 80.
        """
        try:
            # Get image bytes
            if isinstance(image, str):
                with open(image, "rb") as f:
                    img_bytes = f.read()
                temp_img = Image.open(image)
                img_width, img_height = temp_img.size
            elif isinstance(image, np.ndarray):
                img_height, img_width = image.shape[:2]
                _, buffer = cv2.imencode('.jpg', image)
                img_bytes = buffer.tobytes()
            elif isinstance(image, Image.Image):
                img_width, img_height = image.size
                buffered = io.BytesIO()
                if image.mode in ('RGBA', 'LA', 'P'):
                    image = image.convert('RGB')
                image.save(buffered, format="JPEG")
                img_bytes = buffered.getvalue()
            else:
                raise ValueError("Unsupported image format")

            headers = {"Authorization": f"Bearer {self.api_key}"}
            files = {"images": ("image.jpg", img_bytes, "image/jpeg")}

            timeout = (10, 30)
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.endpoint, headers=headers, files=files, timeout=timeout
                    )
                    response.raise_for_status()
                    break
                except requests.exceptions.Timeout:
                    print(f"Timeout on attempt {attempt + 1}/{max_retries}")
                    if attempt == max_retries - 1:
                        raise
                except requests.exceptions.ConnectionError:
                    print(f"Connection error on attempt {attempt + 1}/{max_retries}")
                    if attempt == max_retries - 1:
                        raise
                except requests.exceptions.RequestException as e:
                    print(f"Request error on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt == max_retries - 1:
                        raise

            result = response.json()
            return self.parse_detections(result, classes_filter)

        except Exception as e:
            print(f"Detection API error: {e}")
            return []

    def parse_detections(self, result, classes_filter=None):
        """Parse YOLO response and filter to the requested classes."""
        detections = []

        if isinstance(result, dict):
            if "predictions" in result:
                predictions = result["predictions"]
            elif "results" in result:
                predictions = result["results"]
            elif "detections" in result:
                predictions = result["detections"]
            else:
                predictions = []
        elif isinstance(result, list):
            predictions = result
        else:
            predictions = []

        for prediction in predictions:
            if isinstance(prediction, list):
                for detection in prediction:
                    detections.extend(self._parse_single_detection(detection, classes_filter))
            else:
                detections.extend(self._parse_single_detection(prediction, classes_filter))

        return detections

    def _parse_single_detection(self, detection, classes_filter=None):
        """Parse a single detection object, applying the per-use-case class filter."""
        if not isinstance(detection, dict):
            return []

        class_id = detection.get("class", detection.get("class_id", detection.get("label", -1)))
        confidence = detection.get("confidence", detection.get("score", 0.0))

        bbox = []
        if "box" in detection:
            box = detection["box"]
            if isinstance(box, dict):
                if all(k in box for k in ['x1', 'y1', 'x2', 'y2']):
                    bbox = [box['x1'], box['y1'], box['x2'], box['y2']]
                elif all(k in box for k in ['x', 'y', 'w', 'h']):
                    x, y, w, h = box['x'], box['y'], box['w'], box['h']
                    bbox = [x, y, x + w, y + h]
            elif isinstance(box, list) and len(box) == 4:
                bbox = box
        elif "bbox" in detection:
            bbox = detection["bbox"]
        elif "bounding_box" in detection:
            bbox = detection["bounding_box"]

        # Must be a known COCO class
        if class_id not in COCO_CLASSES:
            return []

        # If the use case specifies a filter, enforce it
        if classes_filter is not None and class_id not in classes_filter:
            return []

        # Basic validity checks
        if confidence <= 0.3 or len(bbox) != 4:
            return []
        if not all(isinstance(coord, (int, float)) for coord in bbox):
            return []

        return [{
            "class": COCO_CLASSES[class_id],
            "confidence": confidence,
            "bbox": bbox
        }]

    def draw_detections(self, image, detections):
        """Draw bounding boxes on image"""
        if isinstance(image, str):
            pil_image = Image.open(image)
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        elif isinstance(image, Image.Image):
            pil_image = image.copy()
        else:
            raise ValueError("Unsupported image format")

        img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        img_height, img_width = img.shape[:2]

        base_size = 1000
        scale_factor = min(img_width, img_height) / base_size
        LINE_THICKNESS = max(1, int(3 * scale_factor))
        FONT_SCALE = max(0.3, 0.6 * scale_factor)
        FONT_THICKNESS = max(1, int(2 * scale_factor))

        for detection in detections:
            bbox = detection["bbox"]
            if len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(int(x1), img_width - 1))
                y1 = max(0, min(int(y1), img_height - 1))
                x2 = max(0, min(int(x2), img_width - 1))
                y2 = max(0, min(int(y2), img_height - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), LINE_THICKNESS)
                label = f"{detection['class']}: {detection['confidence']:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)[0]
                cv2.rectangle(img, (x1, y1 - label_size[1] - 10),
                              (x1 + label_size[0], y1), (0, 255, 0), -1)
                cv2.putText(img, label, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)

        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

CLAUDE_FILE_EOF

echo '  -> analysis_service.py'
cat > analysis_service.py << 'CLAUDE_FILE_EOF'
import requests
import base64
import cv2
import numpy as np
from PIL import Image
import io
import os


class AnalysisService:
    def __init__(self):
        self.endpoint = os.getenv("QWEN_ENDPOINT", "")
        self.api_key = os.getenv("QWEN_API_KEY", "")

    def encode_image(self, image):
        """Convert image to base64 string"""
        if isinstance(image, str):
            with open(image, "rb") as img_file:
                return base64.b64encode(img_file.read()).decode('utf-8')
        elif isinstance(image, np.ndarray):
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        elif isinstance(image, Image.Image):
            buffered = io.BytesIO()
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            image.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def analyze_scene(self, image, detections, prompt_template):
        """
        Send image + detections to Qwen with the use-case's prompt template.

        prompt_template must contain {detections} where the formatted
        detection summary will be substituted.
        """
        try:
            img_b64 = self.encode_image(image)
            detection_summary = self.format_detections(detections)
            prompt = prompt_template.format(detections=detection_summary)

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": "Qwen/Qwen2.5-VL-7B-Instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                            }
                        ]
                    }
                ],
                "max_tokens": 500,
                "temperature": 0.7
            }

            timeout = (10, 60)
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.endpoint, headers=headers, json=payload, timeout=timeout
                    )
                    response.raise_for_status()
                    break
                except requests.exceptions.Timeout:
                    print(f"Analysis timeout on attempt {attempt + 1}/{max_retries}")
                    if attempt == max_retries - 1:
                        raise
                except requests.exceptions.ConnectionError:
                    print(f"Analysis connection error on attempt {attempt + 1}/{max_retries}")
                    if attempt == max_retries - 1:
                        raise
                except requests.exceptions.RequestException as e:
                    print(f"Analysis request error on attempt {attempt + 1}/{max_retries}: {e}")
                    if attempt == max_retries - 1:
                        raise

            result = response.json()
            return self.parse_analysis(result)

        except Exception as e:
            print(f"Analysis error: {e}")
            return self.generate_fallback_analysis(detections)

    # Keep backward-compatible alias
    def analyze_traffic_scene(self, image, detections, prompt_template=None):
        from config import BUILT_IN_USE_CASES
        if prompt_template is None:
            prompt_template = BUILT_IN_USE_CASES["traffic"]["prompt"]
        return self.analyze_scene(image, detections, prompt_template)

    def format_detections(self, detections):
        """Format detections into a bullet list for the prompt"""
        if not detections:
            return "No objects detected"

        object_counts = {}
        for detection in detections:
            obj_type = detection['class']
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1

        summary = []
        for obj_type, count in object_counts.items():
            summary.append(f"- {obj_type.title()}: {count}")

        return "\n".join(summary)

    def parse_analysis(self, result):
        """Parse Qwen response into structured format"""
        try:
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
            else:
                content = str(result)
            return {"content": content}
        except Exception as e:
            print(f"Parse error: {e}")
            return self.generate_fallback_analysis([])

    def generate_fallback_analysis(self, detections):
        """Fallback if Qwen is unreachable — works for any use case."""
        if not detections:
            return {
                "content": "**Traffic Status:** UNKNOWN\n**Traffic Flow:** Unable to determine\n**Incident Analysis:** No objects detected for analysis\n**Safety Assessment:** Cannot assess without detection data\n**Recommendations:** Check connectivity or adjust use case class filter"
            }

        object_counts = {}
        total = 0
        for d in detections:
            object_counts[d['class']] = object_counts.get(d['class'], 0) + 1
            total += 1

        summary = ", ".join(f"{c} {n}" for n, c in object_counts.items())
        status = "NORMAL" if total <= 15 else "ABNORMAL"
        flow = "Light" if total <= 5 else "Moderate" if total <= 15 else "Heavy"

        return {
            "content": f"""
**Traffic Status:** {status}
**Traffic Flow:** {flow}
**Incident Analysis:** {total} object(s) detected: {summary}. AI analysis service unreachable — showing summary from object detection only.
**Safety Assessment:** Unable to perform detailed safety assessment without VLM analysis.
**Recommendations:** Restore Qwen connectivity for full analysis.
"""
        }

CLAUDE_FILE_EOF

echo '  -> database_service.py'
cat > database_service.py << 'CLAUDE_FILE_EOF'
import os
import psycopg2
from psycopg2 import sql
import pandas as pd


class DatabaseService:
    def __init__(self):
        self.db_url = os.getenv("DB_URL", "")
        self.db_user = os.getenv("DB_USER", "")
        self.db_password = os.getenv("DB_PASSWORD", "")
        self.db_name = os.getenv("DB_NAME", "")
        self.connection = None

    def connect(self):
        """Establish connection to PostgreSQL database"""
        try:
            try:
                self.connection = psycopg2.connect(
                    host=self.db_url,
                    database=self.db_name,
                    user=self.db_user,
                    password=self.db_password
                )
                return True, "Connected to database successfully"
            except psycopg2.OperationalError as e:
                if "does not exist" in str(e):
                    return self.create_database()
                else:
                    raise e
        except Exception as e:
            print(f"Database connection error: {e}")
            return False, f"Failed to connect to database: {str(e)}"

    def create_database(self):
        """Create the database if it doesn't exist"""
        try:
            conn = psycopg2.connect(
                host=self.db_url,
                database="postgres",
                user=self.db_user,
                password=self.db_password
            )
            conn.autocommit = True
            cursor = conn.cursor()
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (self.db_name,))
            exists = cursor.fetchone()
            if not exists:
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.db_name)))
            cursor.close()
            conn.close()

            self.connection = psycopg2.connect(
                host=self.db_url,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            return True, f"Database {self.db_name} created and connected successfully"
        except Exception as e:
            return False, f"Failed to create database: {str(e)}"

    def validate_connection(self):
        """Check if database connection is valid"""
        if not self.db_url or not self.db_user or not self.db_password or not self.db_name:
            return False, "Database connection parameters are not configured"
        try:
            success, msg = self.connect()
            if not success:
                return False, msg
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return True, "Database connection successful"
        except Exception as e:
            return False, f"Database connection failed: {str(e)}"

    def create_table_if_not_exists(self):
        """
        Create the horizontal video_analysis table.

        Note: this is a NEW table, separate from the original traffic_analysis
        table, so v1 and v2 can coexist in the same database.
        """
        try:
            if not self.connection:
                success, msg = self.connect()
                if not success:
                    return False, msg
            cursor = self.connection.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS video_analysis (
                    id SERIAL PRIMARY KEY,
                    video_name VARCHAR(255),
                    use_case VARCHAR(100),
                    frame_id VARCHAR(50),
                    status_field VARCHAR(100),
                    flow_field VARCHAR(100),
                    incident_analysis TEXT,
                    safety_assessment TEXT,
                    recommendations TEXT,
                    raw_analysis TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self.connection.commit()
            cursor.close()
            return True, "video_analysis table created or already exists"
        except Exception as e:
            return False, f"Failed to create table: {str(e)}"

    def save_analysis_results(self, video_name, use_case_key, analysis_results):
        """Save analysis results to the video_analysis table with use_case tag"""
        try:
            if not self.connection:
                success, msg = self.connect()
                if not success:
                    return False, msg

            table_success, table_msg = self.create_table_if_not_exists()
            if not table_success:
                return False, table_msg

            cursor = self.connection.cursor()
            for result in analysis_results:
                frame_id = f"frame_{result['frame']}"
                analysis = result.get('analysis', {})
                content = analysis.get('content', '')

                # Generic extraction: the first two "header" lines are used as
                # status_field / flow_field regardless of what the use case
                # calls them. The full text is always stored in raw_analysis.
                status = self._extract_field(content, "Traffic Status")
                flow = self._extract_field(content, "Traffic Flow")
                incident = self._extract_field(content, "Incident Analysis")
                safety = self._extract_field(content, "Safety Assessment")
                recs = self._extract_field(content, "Recommendations")

                cursor.execute("""
                    INSERT INTO video_analysis
                    (video_name, use_case, frame_id, status_field, flow_field,
                     incident_analysis, safety_assessment, recommendations, raw_analysis)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    video_name, use_case_key, frame_id,
                    status, flow, incident, safety, recs, content
                ))
            self.connection.commit()
            cursor.close()
            return True, f"Successfully saved {len(analysis_results)} analysis results to database"
        except Exception as e:
            return False, f"Failed to save analysis results: {str(e)}"

    def _extract_field(self, content, field_name):
        """Extract a field value from the Qwen response."""
        try:
            if not content:
                return ""
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if field_name.lower() in line.lower():
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if not next_line or ":" in next_line:
                            parts = line.split(':', 1)
                            return parts[1].strip() if len(parts) > 1 else ""
                        else:
                            result = []
                            j = i + 1
                            while j < len(lines) and not any(
                                h in lines[j].lower() for h in [
                                    "traffic status", "traffic flow", "incident analysis",
                                    "safety assessment", "recommendations"
                                ]
                            ):
                                if lines[j].strip():
                                    result.append(lines[j].strip())
                                j += 1
                            return " ".join(result)
            for line in lines:
                if field_name.lower() in line.lower():
                    parts = line.split(':', 1)
                    return parts[1].strip() if len(parts) > 1 else ""
            return ""
        except Exception:
            return ""

    def get_analysis_results(self, video_name=None, use_case=None, limit=100):
        """Get analysis results, optionally filtered by video name or use case"""
        try:
            if not self.connection:
                success, msg = self.connect()
                if not success:
                    return False, msg, None
            cursor = self.connection.cursor()

            if video_name and use_case:
                cursor.execute("""
                    SELECT * FROM video_analysis
                    WHERE video_name = %s AND use_case = %s
                    ORDER BY id DESC LIMIT %s
                """, (video_name, use_case, limit))
            elif video_name:
                cursor.execute("""
                    SELECT * FROM video_analysis
                    WHERE video_name = %s
                    ORDER BY id DESC LIMIT %s
                """, (video_name, limit))
            elif use_case:
                cursor.execute("""
                    SELECT * FROM video_analysis
                    WHERE use_case = %s
                    ORDER BY id DESC LIMIT %s
                """, (use_case, limit))
            else:
                cursor.execute("""
                    SELECT * FROM video_analysis
                    ORDER BY id DESC LIMIT %s
                """, (limit,))

            columns = [desc[0] for desc in cursor.description]
            results = cursor.fetchall()
            cursor.close()
            df = pd.DataFrame(results, columns=columns)
            return True, f"Retrieved {len(results)} records", df
        except Exception as e:
            return False, f"Failed to retrieve analysis results: {str(e)}", None

    def close(self):
        if self.connection:
            self.connection.close()
            self.connection = None

CLAUDE_FILE_EOF

echo '  -> app.py'
cat > app.py << 'CLAUDE_FILE_EOF'
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time

from detection_service import DetectionService
from analysis_service import AnalysisService
from database_service import DatabaseService
from use_case_service import UseCaseService
from config import COCO_CLASSES, COCO_NAME_TO_ID, BUILT_IN_USE_CASES, DEFAULT_USE_CASE

# ------------------------------------------------------------------
# Settings storage (same pattern as v1)
# ------------------------------------------------------------------
settings = {
    "yolo_endpoint": os.getenv("YOLO_ENDPOINT", "https://your-yolo-endpoint.com/predict"),
    "yolo_api_key": os.getenv("YOLO_API_KEY", "your-yolo-api-key-here"),
    "qwen_endpoint": os.getenv("QWEN_ENDPOINT", "https://your-qwen-endpoint.com/v1/chat/completions"),
    "qwen_api_key": os.getenv("QWEN_API_KEY", "your-qwen-api-key-here"),
    "db_url": os.getenv("DB_URL", "localhost"),
    "db_user": os.getenv("DB_USER", "postgres"),
    "db_password": os.getenv("DB_PASSWORD", ""),
    "db_name": os.getenv("DB_NAME", "traffic_analysis"),
}


def update_settings(yolo_endpoint, yolo_api_key, qwen_endpoint, qwen_api_key,
                    db_url=None, db_user=None, db_password=None, db_name=None):
    settings["yolo_endpoint"] = yolo_endpoint
    settings["yolo_api_key"] = yolo_api_key
    settings["qwen_endpoint"] = qwen_endpoint
    settings["qwen_api_key"] = qwen_api_key

    if db_url is not None:
        settings["db_url"] = db_url
    if db_user is not None:
        settings["db_user"] = db_user
    if db_password is not None:
        settings["db_password"] = db_password
    if db_name is not None:
        settings["db_name"] = db_name

    os.environ["YOLO_ENDPOINT"] = yolo_endpoint.replace("/predict", "") if yolo_endpoint else ""
    os.environ["YOLO_API_KEY"] = yolo_api_key if yolo_api_key else ""
    os.environ["QWEN_ENDPOINT"] = qwen_endpoint if qwen_endpoint else ""
    os.environ["QWEN_API_KEY"] = qwen_api_key if qwen_api_key else ""
    os.environ["DB_URL"] = settings["db_url"]
    os.environ["DB_USER"] = settings["db_user"]
    os.environ["DB_PASSWORD"] = settings["db_password"]
    os.environ["DB_NAME"] = settings["db_name"]

    for s in ("detector", "analyzer", "db_service", "use_case_service"):
        if s in st.session_state:
            del st.session_state[s]

    return "Settings updated successfully!"


# ------------------------------------------------------------------
# Lazy service getters
# ------------------------------------------------------------------
def get_detector():
    if 'detector' not in st.session_state:
        st.session_state.detector = DetectionService()
    return st.session_state.detector


def get_analyzer():
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = AnalysisService()
    return st.session_state.analyzer


def get_db_service():
    if 'db_service' not in st.session_state:
        st.session_state.db_service = DatabaseService()
    return st.session_state.db_service


def get_use_case_service():
    if 'use_case_service' not in st.session_state:
        st.session_state.use_case_service = UseCaseService()
    return st.session_state.use_case_service


def get_active_use_case():
    """Return the currently selected use case dict."""
    ucs = get_use_case_service()
    key = st.session_state.get("active_use_case_key", DEFAULT_USE_CASE)
    uc = ucs.get_use_case(key)
    if uc is None:
        # Fall back to default if the chosen one was deleted
        uc = ucs.get_use_case(DEFAULT_USE_CASE)
        st.session_state.active_use_case_key = DEFAULT_USE_CASE
    return uc


# ------------------------------------------------------------------
# Image + video processing (use-case aware)
# ------------------------------------------------------------------
def process_image(image, use_case):
    try:
        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
        else:
            pil_image = image

        detector = get_detector()
        analyzer = get_analyzer()

        yolo_valid, yolo_msg = detector.validate_endpoint()
        detections = detector.detect_objects(pil_image, classes_filter=use_case["classes"])
        annotated_image = detector.draw_detections(pil_image, detections) if detections else pil_image
        analysis = analyzer.analyze_scene(pil_image, detections, use_case["prompt"])
        results_text = format_analysis_results(detections, analysis, use_case, yolo_valid, yolo_msg)
        return annotated_image, results_text
    except Exception as e:
        return image, f"## Error Processing Image\n\n**Error:** {str(e)}"


def process_video(video_path, use_case, interval_seconds=3):
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(fps * interval_seconds)

        results = []
        frame_count = 0
        detector = get_detector()
        analyzer = get_analyzer()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_frame = Image.fromarray(rgb)
                detections = detector.detect_objects(pil_frame, classes_filter=use_case["classes"])
                analysis = analyzer.analyze_scene(pil_frame, detections, use_case["prompt"])
                annotated = detector.draw_detections(pil_frame, detections) if detections else pil_frame
                results.append({
                    "frame": frame_count,
                    "timestamp": frame_count / fps,
                    "detections": detections,
                    "analysis": analysis,
                    "annotated_frame": annotated
                })
                if len(results) >= 10:
                    break
            frame_count += 1

        cap.release()

        annotated_frames = [
            {"frame": r["annotated_frame"],
             "timestamp": r["timestamp"],
             "frame_number": r["frame"]}
            for r in results
        ]
        return annotated_frames, f"**Analysis Summary:** Processed {len(results)} frames at {interval_seconds}-second intervals using use case: {use_case['name']}\n\n", results
    except Exception as e:
        return None, f"Error processing video: {str(e)}", []


def format_analysis_results(detections, analysis, use_case, yolo_valid=True, yolo_msg=""):
    """Format results for display. Detection labels adapt to the use case."""
    text = f"### {use_case['icon']} {use_case['name']}\n\n"

    if not yolo_valid:
        text += f"⚠️ **YOLO Detection API:** {yolo_msg}\n\n"

    text += "### Detected Objects\n"
    if detections:
        counts = {}
        for d in detections:
            counts[d['class'].title()] = counts.get(d['class'].title(), 0) + 1
        for obj, c in counts.items():
            text += f"- **{obj}:** {c}\n"
    else:
        text += "- No relevant objects detected for this use case\n"

    text += "\n### AI Report\n"
    content = analysis.get('content', 'Analysis not available')
    text += content if isinstance(content, str) else str(content)
    return text


def format_single_frame_results(frame_result, frame_index, use_case):
    if not frame_result:
        return "No analysis results available"
    ts = frame_result['timestamp']
    text = f"## {use_case['icon']} Frame {frame_index} - {ts:.1f}s\n\n### Detected Objects\n"
    detections = frame_result['detections']
    if detections:
        counts = {}
        for d in detections:
            counts[d['class']] = counts.get(d['class'], 0) + 1
        for c, n in counts.items():
            text += f"- **{c.title()}:** {n}\n"
    else:
        text += "- No objects detected\n"
    analysis = frame_result['analysis']
    content = analysis.get('content', '')
    if isinstance(content, str):
        text += "\n### AI Analysis Report\n" + content
    return text


def export_to_database(video_name, use_case_key, analysis_results):
    try:
        db = get_db_service()
        return db.save_analysis_results(video_name, use_case_key, analysis_results)
    except Exception as e:
        return False, f"Error exporting to database: {str(e)}"


# ==================================================================
# UI
# ==================================================================
def create_interface():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True

    st.set_page_config(
        page_title="Smart Video Analytics",
        page_icon="🎥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🎥 Smart Video Analytics")
    st.markdown("*Configurable video analysis for any use case — traffic, hospital, workshop, or your own.*")

    # --------------------------------------------------------------
    # Active use case selector (persistent across tabs)
    # --------------------------------------------------------------
    ucs = get_use_case_service()

    # Make sure the use_cases table exists (DB might not be configured yet)
    try:
        ucs.create_table_if_not_exists()
    except Exception:
        pass

    all_use_cases = ucs.get_all_use_cases()
    uc_labels = [f"{uc['icon']} {uc['name']}" + (" (built-in)" if uc['is_builtin'] else " (custom)") for uc in all_use_cases]
    uc_keys = [uc['key'] for uc in all_use_cases]

    if "active_use_case_key" not in st.session_state:
        st.session_state.active_use_case_key = DEFAULT_USE_CASE

    try:
        current_index = uc_keys.index(st.session_state.active_use_case_key)
    except ValueError:
        current_index = 0
        st.session_state.active_use_case_key = uc_keys[0]

    col_uc1, col_uc2 = st.columns([3, 1])
    with col_uc1:
        selected_label = st.selectbox("**Active Use Case**", uc_labels, index=current_index,
                                      help="Pick the scenario you're analyzing — this controls what YOLO filters for and what Qwen is asked.")
        st.session_state.active_use_case_key = uc_keys[uc_labels.index(selected_label)]
    with col_uc2:
        if st.button("🔍 Check API & DB Status"):
            with st.spinner("Checking..."):
                detector = get_detector()
                yolo_valid, yolo_msg = detector.validate_endpoint()
                st.success(f"✅ YOLO: {yolo_msg}") if yolo_valid else st.error(f"❌ YOLO: {yolo_msg}")
                db = get_db_service()
                db_valid, db_msg = db.validate_connection()
                st.success(f"✅ Database: {db_msg}") if db_valid else st.error(f"❌ Database: {db_msg}")

    active_uc = get_active_use_case()
    st.info(f"{active_uc['icon']} **Current:** {active_uc['name']} — {active_uc['description']}")
    st.markdown("---")

    # --------------------------------------------------------------
    # Tabs
    # --------------------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["🖼️ Image Analysis", "🎬 Video Analysis", "⚙️ Settings"])

    # --------------- Tab 1: Image Analysis ---------------
    with tab1:
        st.header("Image Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Upload Image")
            image_input = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'], key="image_upload")
            if image_input and st.button("Analyze Image", type="primary"):
                with st.spinner(f"Analyzing as {active_uc['name']}..."):
                    try:
                        pil_image = Image.open(image_input)
                        annotated, results_text = process_image(pil_image, active_uc)
                        st.session_state.image_results = (annotated, results_text)
                    except Exception as e:
                        st.error(f"Error processing image: {e}")
        with col2:
            st.subheader("Results")
            if hasattr(st.session_state, 'image_results') and st.session_state.image_results:
                annotated, results_text = st.session_state.image_results
                if annotated:
                    st.image(annotated, caption="Detection Results", use_column_width=True)
                if results_text:
                    st.markdown(results_text)

    # --------------- Tab 2: Video Analysis ---------------
    with tab2:
        st.header("Video Analysis")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Upload Video")
            video_input = st.file_uploader("Choose a video...", type=['mp4', 'avi', 'mov', 'mkv'], key="video_upload")

            video_duration = None
            if video_input:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                    tmp.write(video_input.getvalue())
                    tmp_path = tmp.name
                try:
                    cap = cv2.VideoCapture(tmp_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    video_duration = int(total / fps) if fps > 0 else 30
                    cap.release()
                finally:
                    os.unlink(tmp_path)

            if video_duration:
                interval_seconds = st.number_input(
                    "Analysis Interval (seconds)",
                    min_value=1, max_value=video_duration, value=3, step=1,
                    help=f"Extract frames every N seconds. Video duration: {video_duration}s"
                )
            else:
                interval_seconds = st.number_input("Analysis Interval (seconds)",
                                                   min_value=1, max_value=300, value=3, step=1)

            if video_input and st.button("Analyze Video", type="primary"):
                with st.spinner(f"Analyzing as {active_uc['name']}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
                        tmp.write(video_input.read())
                        tmp_path = tmp.name
                    try:
                        annotated_frames, video_results, frame_results = process_video(
                            tmp_path, active_uc, interval_seconds
                        )
                        st.session_state.video_results = (annotated_frames, video_results)
                        st.session_state.video_analysis_results = frame_results
                        st.session_state.video_name = video_input.name
                        st.session_state.video_use_case_key = active_uc['key']
                    finally:
                        os.unlink(tmp_path)

            # Export to DB
            if hasattr(st.session_state, 'video_analysis_results') and st.session_state.video_analysis_results:
                st.markdown("---")
                st.subheader("💾 Database Export")
                custom_name = st.text_input("Video Name (for database)",
                                            value=getattr(st.session_state, 'video_name', "video"))
                if st.button("Export to Database", type="primary"):
                    with st.spinner("Exporting..."):
                        db = get_db_service()
                        ok, msg = db.validate_connection()
                        if ok:
                            success, message = export_to_database(
                                custom_name,
                                st.session_state.video_use_case_key,
                                st.session_state.video_analysis_results
                            )
                            st.success(f"✅ {message}") if success else st.error(f"❌ {message}")
                        else:
                            st.error(f"❌ Database connection failed: {msg}")

        with col2:
            st.subheader("Results")
            if hasattr(st.session_state, 'video_results') and st.session_state.video_results:
                annotated_frames, video_results = st.session_state.video_results
                if annotated_frames:
                    frame_options = [f"Frame {i+1} ({f['timestamp']:.1f}s)" for i, f in enumerate(annotated_frames)]
                    idx = st.selectbox("Choose a frame to analyze:", range(len(frame_options)),
                                       format_func=lambda x: frame_options[x], key="frame_selector")
                    if idx is not None and idx < len(annotated_frames):
                        fd = annotated_frames[idx]
                        st.image(fd['frame'], caption=f"Frame at {fd['timestamp']:.1f}s",
                                 use_column_width=True)
                        if hasattr(st.session_state, 'video_analysis_results'):
                            fr = st.session_state.video_analysis_results[idx]
                            st.markdown("### Frame Analysis Report")
                            st.markdown(format_single_frame_results(fr, idx + 1, active_uc))

    # --------------- Tab 3: Settings ---------------
    with tab3:
        st.header("Configuration")
        s_tab1, s_tab2, s_tab3 = st.tabs(["🔌 Model APIs", "🗄️ Database", "🎯 Use Cases"])

        # ----- Model APIs -----
        with s_tab1:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("YOLO Detection Settings")
                yolo_endpoint_input = st.text_input("YOLO Endpoint", value=settings["yolo_endpoint"])
                yolo_key_input = st.text_input("YOLO API Key", value=settings["yolo_api_key"], type="password")
            with col2:
                st.subheader("Qwen2.5-VL Analysis Settings")
                qwen_endpoint_input = st.text_input("Qwen Endpoint", value=settings["qwen_endpoint"])
                qwen_key_input = st.text_input("Qwen API Key", value=settings["qwen_api_key"], type="password")
            if st.button("Save API Settings", type="primary"):
                update_settings(yolo_endpoint_input, yolo_key_input, qwen_endpoint_input, qwen_key_input)
                st.success("API settings saved!")

        # ----- Database -----
        with s_tab2:
            col1, col2 = st.columns(2)
            with col1:
                db_url_input = st.text_input("Database Host", value=settings["db_url"])
                db_name_input = st.text_input("Database Name", value=settings["db_name"])
            with col2:
                db_user_input = st.text_input("Database User", value=settings["db_user"])
                db_password_input = st.text_input("Database Password", value=settings["db_password"], type="password")
            if st.button("Save Database Settings", type="primary"):
                update_settings(settings["yolo_endpoint"], settings["yolo_api_key"],
                                settings["qwen_endpoint"], settings["qwen_api_key"],
                                db_url_input, db_user_input, db_password_input, db_name_input)
                with st.spinner("Testing connection..."):
                    db = get_db_service()
                    ok, msg = db.validate_connection()
                    st.success(f"✅ {msg}") if ok else st.error(f"❌ {msg}")
                    if ok:
                        ok2, msg2 = db.create_table_if_not_exists()
                        st.success(f"✅ {msg2}") if ok2 else st.error(f"❌ {msg2}")
                        ok3, msg3 = get_use_case_service().create_table_if_not_exists()
                        st.success(f"✅ {msg3}") if ok3 else st.error(f"❌ {msg3}")

        # ----- Use Cases -----
        with s_tab3:
            st.subheader("🎯 Use Cases")
            st.markdown("Use cases define *what* YOLO filters for and *what question* Qwen answers. Built-ins ship with the app; you can also create your own below.")

            st.markdown("### Available Use Cases")
            for uc in ucs.get_all_use_cases():
                with st.expander(f"{uc['icon']} {uc['name']} {'(built-in)' if uc['is_builtin'] else '(custom)'}", expanded=False):
                    st.markdown(f"**Description:** {uc['description']}")
                    class_names = [COCO_CLASSES[c] for c in uc['classes'] if c in COCO_CLASSES]
                    st.markdown(f"**Detects:** {', '.join(class_names) if class_names else 'none'}")
                    st.markdown("**Prompt:**")
                    st.code(uc['prompt'], language="markdown")
                    if not uc['is_builtin']:
                        if st.button(f"🗑️ Delete", key=f"del_{uc['key']}"):
                            ok, msg = ucs.delete_custom_use_case(uc['key'])
                            st.success(msg) if ok else st.error(msg)
                            st.rerun()

            st.markdown("---")
            st.markdown("### ➕ Create a Custom Use Case")
            with st.form("new_use_case_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    new_name = st.text_input("Name*", placeholder="e.g., School Drop-off Zone")
                    new_key = st.text_input("Key* (lowercase, underscores)", placeholder="e.g., school_dropoff",
                                            help="Short identifier. No spaces. Used in the DB and dropdown.")
                with col_b:
                    new_icon = st.text_input("Icon (emoji)", value="⚙️", max_chars=4)
                    new_desc = st.text_input("Short Description", placeholder="What does this scenario monitor?")

                st.markdown("**Which objects should YOLOv8 detect?**")
                st.caption("Tick every object relevant to your scenario. YOLOv8 knows 80 object types.")
                selected_classes = []
                # Render checkboxes in a 4-column grid
                cols = st.columns(4)
                class_items = sorted(COCO_CLASSES.items(), key=lambda x: x[1])
                for i, (cid, cname) in enumerate(class_items):
                    with cols[i % 4]:
                        if st.checkbox(cname, key=f"cls_{cid}"):
                            selected_classes.append(cid)

                st.markdown("**Qwen Analysis Prompt***")
                st.caption("Write what you want Qwen to analyze. Must include `{detections}` where the detected objects list should appear.")
                new_prompt = st.text_area(
                    "Prompt",
                    height=250,
                    value="""Analyze this scene with the following detected objects: {detections}

Please provide a concise analysis in this format:

**Traffic Status:** [status here]
**Traffic Flow:** [flow level here]
**Incident Analysis:** [what's happening]
**Safety Assessment:** [safety notes]
**Recommendations:** [suggestions]

Keep each section to 1-2 sentences maximum.
"""
                )

                submitted = st.form_submit_button("💾 Save Use Case", type="primary")
                if submitted:
                    ok, msg = ucs.save_custom_use_case(
                        key=new_key.strip(),
                        name=new_name.strip(),
                        icon=new_icon.strip() or "⚙️",
                        description=new_desc.strip(),
                        classes=selected_classes,
                        prompt=new_prompt
                    )
                    if ok:
                        st.success(f"✅ {msg}")
                        st.info("Reload the page to see it in the Active Use Case dropdown at the top.")
                    else:
                        st.error(f"❌ {msg}")

    # Footer
    st.markdown("---")
    st.markdown("*Powered by YOLOv8 for object detection and Qwen2.5-VL for scene analysis. Horizontal by design.*")


if not os.getenv("YOLO_API_KEY") or not os.getenv("QWEN_API_KEY"):
    print("Warning: API keys not set. Configure them via env vars or the Settings tab.")

create_interface()

CLAUDE_FILE_EOF

echo ""
echo "✅ All 6 files written successfully."
echo ""
echo "Next steps:"
echo "  1. docker build -t traffic-report-v2:latest ."
echo "  2. docker rm -f traffic-report-v2"
echo "  3. docker run -d -p 8505:8505 --name traffic-report-v2 --restart=unless-stopped \\"
echo "       -e DB_URL=10.79.252.63 -e DB_USER=postgres -e DB_PASSWORD=admin \\"
echo "       -e DB_NAME=traffic_analysis \\"
echo "       -e YOLO_ENDPOINT=http://10.79.252.45:8008 -e YOLO_API_KEY=test \\"
echo "       -e QWEN_ENDPOINT=http://10.79.252.45:8007/v1/chat/completions -e QWEN_API_KEY=test \\"
echo "       traffic-report-v2:latest"
echo "  4. Open http://10.79.252.63:8505"
