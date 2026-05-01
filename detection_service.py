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

