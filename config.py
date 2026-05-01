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

