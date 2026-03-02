"""SQLite-based append-only conversation log.

Stores all conversation turns for batch processing during sleep cycles.
Uses WAL mode for concurrent read/write safety.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_turns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    processed INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_conv_id ON conversation_turns(conversation_id);
CREATE INDEX IF NOT EXISTS idx_processed ON conversation_turns(processed);
CREATE INDEX IF NOT EXISTS idx_timestamp ON conversation_turns(timestamp);
"""


class ConversationLog:
    """Append-only conversation log backed by SQLite.

    All conversation turns are recorded for later batch processing
    by the sleep cycle memory extraction pipeline.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def append_turn(
        self,
        conversation_id: str,
        turn_index: int,
        role: str,
        content: str,
    ) -> int:
        """Append a conversation turn. Returns the row id."""
        now = datetime.now().isoformat()
        cursor = self._conn.execute(
            "INSERT INTO conversation_turns (conversation_id, turn_index, role, content, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, turn_index, role, content, now),
        )
        self._conn.commit()
        return cursor.lastrowid or 0

    def get_unprocessed_turns(self, limit: int = 500) -> list[dict]:
        """Get unprocessed turns ordered by conversation and turn index.

        Returns list of dicts with keys: id, conversation_id, turn_index, role, content, timestamp.
        """
        cursor = self._conn.execute(
            "SELECT id, conversation_id, turn_index, role, content, timestamp "
            "FROM conversation_turns "
            "WHERE processed = 0 "
            "ORDER BY conversation_id, turn_index "
            "LIMIT ?",
            (limit,),
        )
        return [
            {
                "id": row[0],
                "conversation_id": row[1],
                "turn_index": row[2],
                "role": row[3],
                "content": row[4],
                "timestamp": row[5],
            }
            for row in cursor.fetchall()
        ]

    def get_conversation(self, conversation_id: str) -> list[dict]:
        """Get all turns for a specific conversation, ordered by turn index."""
        cursor = self._conn.execute(
            "SELECT id, conversation_id, turn_index, role, content, timestamp, processed "
            "FROM conversation_turns "
            "WHERE conversation_id = ? "
            "ORDER BY turn_index",
            (conversation_id,),
        )
        return [
            {
                "id": row[0],
                "conversation_id": row[1],
                "turn_index": row[2],
                "role": row[3],
                "content": row[4],
                "timestamp": row[5],
                "processed": bool(row[6]),
            }
            for row in cursor.fetchall()
        ]

    def mark_processed(self, turn_ids: list[int]) -> int:
        """Mark turns as processed. Returns number of rows updated."""
        if not turn_ids:
            return 0
        placeholders = ",".join("?" for _ in turn_ids)
        cursor = self._conn.execute(
            f"UPDATE conversation_turns SET processed = 1 WHERE id IN ({placeholders})",
            turn_ids,
        )
        self._conn.commit()
        return cursor.rowcount

    def cleanup_old(self, days: int = 30) -> int:
        """Delete processed turns older than `days`. Returns number of rows deleted."""
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        cursor = self._conn.execute(
            "DELETE FROM conversation_turns WHERE processed = 1 AND timestamp < ?",
            (cutoff,),
        )
        self._conn.commit()
        deleted = cursor.rowcount
        if deleted > 0:
            logger.info("Cleaned up %d old conversation turns (older than %d days)", deleted, days)
        return deleted

    def count(self, processed: bool | None = None) -> int:
        """Count turns, optionally filtered by processed status."""
        if processed is None:
            cursor = self._conn.execute("SELECT COUNT(*) FROM conversation_turns")
        else:
            cursor = self._conn.execute(
                "SELECT COUNT(*) FROM conversation_turns WHERE processed = ?",
                (1 if processed else 0,),
            )
        return cursor.fetchone()[0]

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()
