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

