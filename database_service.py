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
                    status_field VARCHAR(500),
                    flow_field VARCHAR(500),
                    incident_analysis TEXT,
                    safety_assessment TEXT,
                    recommendations TEXT,
                    raw_analysis TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # If the table was created by an older build with 100-char columns,
            # widen them in-place. ALTER is safe to run even if already wider.
            cursor.execute("ALTER TABLE video_analysis ALTER COLUMN status_field TYPE VARCHAR(500)")
            cursor.execute("ALTER TABLE video_analysis ALTER COLUMN flow_field TYPE VARCHAR(500)")
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

                # Try multiple label variants for each field so non-traffic use
                # cases (hospital, workshop, school, etc.) populate cleanly too.
                # The raw_analysis column always stores the full text as backup.
                status = self._extract_any(content, [
                    "Traffic Status", "Safety Status", "Status", "Patient Status"
                ])
                flow = self._extract_any(content, [
                    "Traffic Flow", "Patient Flow", "Activity Level",
                    "Activity", "Flow"
                ])
                incident = self._extract_any(content, ["Incident Analysis", "Scene Analysis", "Analysis"])
                safety = self._extract_any(content, ["Safety Assessment", "Assessment"])
                recs = self._extract_any(content, ["Recommendations", "Recommendation"])

                # Truncate to column limits as a safety net (columns are 500,
                # but Qwen sometimes returns long values)
                status = (status or "")[:500]
                flow = (flow or "")[:500]

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

    def _extract_any(self, content, field_names):
        """Try several label names, return the first non-empty match."""
        for name in field_names:
            val = self._extract_field(content, name)
            if val:
                return val
        return ""

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
