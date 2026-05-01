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
                    extra_class_names JSON,
                    prompt TEXT NOT NULL,
                    is_builtin BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # If upgrading from an older version that didn't have this column,
            # add it in-place. ALTER is idempotent via IF NOT EXISTS.
            cursor.execute("""
                ALTER TABLE use_cases
                ADD COLUMN IF NOT EXISTS extra_class_names JSON
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
                SELECT key, name, icon, description, classes, extra_class_names, prompt
                FROM use_cases
                WHERE is_builtin = FALSE
                ORDER BY created_at ASC
            """)
            for row in cursor.fetchall():
                key, name, icon, description, classes, extra_names, prompt = row
                all_cases.append({
                    "key": key,
                    "name": name,
                    "icon": icon or "⚙️",
                    "description": description or "",
                    # `classes` comes back from Postgres JSON column as a list
                    "classes": classes if isinstance(classes, list) else json.loads(classes) if classes else [],
                    "extra_class_names": extra_names if isinstance(extra_names, list) else (json.loads(extra_names) if extra_names else []),
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

    def save_custom_use_case(self, key, name, icon, description, classes, prompt,
                              extra_class_names=None):
        """
        Save or update a user-defined use case.

        classes:           list of valid COCO IDs (YOLO will detect + draw boxes)
        extra_class_names: list of object names NOT in COCO that the user typed
                           anyway (e.g. "robot"). YOLO can't find them, but
                           they'll be mentioned in the Qwen prompt so the VLM
                           knows what to look for.

        Validation rules:
          - key must be lowercase, no spaces (use underscores)
          - key cannot collide with a built-in
          - at least one class OR extra_class_name must be provided
          - prompt must contain {detections}
        """
        extra_class_names = extra_class_names or []

        # Guard rails
        if key in BUILT_IN_USE_CASES:
            return False, f"'{key}' is a built-in use case and cannot be overwritten"

        if not key or not key.replace("_", "").isalnum() or key != key.lower():
            return False, "Key must be lowercase letters, numbers, and underscores only"

        if not classes and not extra_class_names:
            return False, "You must enter at least one object to detect"

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
                INSERT INTO use_cases (key, name, icon, description, classes, extra_class_names, prompt, is_builtin)
                VALUES (%s, %s, %s, %s, %s, %s, %s, FALSE)
                ON CONFLICT (key) DO UPDATE SET
                    name = EXCLUDED.name,
                    icon = EXCLUDED.icon,
                    description = EXCLUDED.description,
                    classes = EXCLUDED.classes,
                    extra_class_names = EXCLUDED.extra_class_names,
                    prompt = EXCLUDED.prompt
            """, (key, name, icon, description,
                  json.dumps(classes), json.dumps(extra_class_names),
                  prompt))
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
