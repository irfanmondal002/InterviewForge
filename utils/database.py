import json
import sqlite3

from contextlib import contextmanager
from datetime import datetime

from utils.config import DB_PATH


@contextmanager
def get_connection():

    connection = sqlite3.connect(DB_PATH)

    connection.row_factory = sqlite3.Row

    try:

        connection.execute("PRAGMA foreign_keys = ON")

        yield connection

        connection.commit()

    except Exception:

        connection.rollback()

        raise

    finally:

        connection.close()


def init_db():

    with get_connection() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS applications (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                job_title TEXT NOT NULL,

                resume_filename TEXT NOT NULL,

                jd_filename TEXT NOT NULL,

                resume_text TEXT NOT NULL,

                jd_text TEXT NOT NULL,

                analysis_json TEXT,

                questions_json TEXT,

                study_plan_json TEXT,

                created_at TEXT NOT NULL

            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                application_id INTEGER NOT NULL,

                role TEXT NOT NULL,

                content TEXT NOT NULL,

                created_at TEXT NOT NULL,

                FOREIGN KEY(application_id)
                    REFERENCES applications(id)
                    ON DELETE CASCADE

            )
        """)


def create_application(
    job_title,
    resume_filename,
    jd_filename,
    resume_text,
    jd_text
):

    with get_connection() as connection:

        cursor = connection.execute(
            """
            INSERT INTO applications (

                job_title,
                resume_filename,
                jd_filename,
                resume_text,
                jd_text,
                created_at

            )

            VALUES (?, ?, ?, ?, ?, ?)
            """,

            (
                job_title,
                resume_filename,
                jd_filename,
                resume_text,
                jd_text,
                datetime.now().isoformat(timespec="seconds")
            )
        )

        return cursor.lastrowid


def save_analysis(application_id, analysis):

    analysis_json = json.dumps(
        analysis,
        ensure_ascii=False
    )

    with get_connection() as connection:

        connection.execute(
            """
            UPDATE applications

            SET analysis_json = ?

            WHERE id = ?
            """,

            (
                analysis_json,
                application_id
            )
        )


def save_generated_result(
    application_id,
    result,
    result_type
):

    if result_type not in (
        "questions_json",
        "study_plan_json"
    ):

        raise ValueError("Invalid result type.")

    result_json = json.dumps(
        result,
        ensure_ascii=False
    )

    with get_connection() as connection:

        query = f"""
            UPDATE applications

            SET {result_type} = ?

            WHERE id = ?
        """

        connection.execute(
            query,
            (
                result_json,
                application_id
            )
        )


def get_application(application_id):

    if application_id is None:
        return None

    with get_connection() as connection:

        row = connection.execute(
            """
            SELECT *

            FROM applications

            WHERE id = ?
            """,

            (application_id,)
        ).fetchone()

        return dict(row) if row else None


def list_applications():

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT
                id,
                job_title,
                created_at

            FROM applications

            ORDER BY id DESC
            """
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]


def save_chat_message(
    application_id,
    role,
    content
):

    with get_connection() as connection:

        connection.execute(
            """
            INSERT INTO chat_messages (

                application_id,
                role,
                content,
                created_at

            )

            VALUES (?, ?, ?, ?)
            """,

            (
                application_id,
                role,
                content,
                datetime.now().isoformat(timespec="seconds")
            )
        )


def get_chat_history(
    application_id,
    limit=50
):

    with get_connection() as connection:

        rows = connection.execute(
            """
            SELECT
                role,
                content,
                created_at

            FROM chat_messages

            WHERE application_id = ?

            ORDER BY id DESC

            LIMIT ?
            """,

            (
                application_id,
                limit
            )
        ).fetchall()

        # Reverse to show the conversation chronologically.
        return [
            dict(row)
            for row in reversed(rows)
        ]