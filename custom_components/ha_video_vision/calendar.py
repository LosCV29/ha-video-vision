"""Calendar platform for HA Video Vision - Timeline of camera analysis events."""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import uuid

from homeassistant.components.calendar import CalendarEntity, CalendarEvent
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.util import dt as dt_util

from .const import (
    DOMAIN,
    CONF_TIMELINE_ENABLED,
    CONF_TIMELINE_RETENTION_DAYS,
    DEFAULT_TIMELINE_RETENTION_DAYS,
)

_LOGGER = logging.getLogger(__name__)

DATABASE_NAME = "timeline.db"


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up HA Video Vision calendar from a config entry."""
    config = {**entry.data, **entry.options}

    # Only set up if timeline is enabled
    if not config.get(CONF_TIMELINE_ENABLED, True):
        _LOGGER.debug("Timeline disabled, skipping calendar setup")
        return

    timeline = VideoVisionTimeline(hass, entry)
    async_add_entities([timeline])

    # Store reference for service calls
    hass.data[DOMAIN][entry.entry_id]["timeline"] = timeline
    _LOGGER.info("HA Video Vision Timeline calendar created")


class VideoVisionTimeline(CalendarEntity):
    """Calendar entity that stores camera analysis events."""

    _attr_has_entity_name = True
    _attr_name = "Timeline"
    _attr_icon = "mdi:cctv"

    def __init__(self, hass: HomeAssistant, entry: ConfigEntry) -> None:
        """Initialize the timeline calendar."""
        self.hass = hass
        self._entry = entry
        self._attr_unique_id = f"{DOMAIN}_timeline"

        # Database path in HA storage directory
        self._db_path = Path(hass.config.path(".storage")) / DOMAIN / DATABASE_NAME
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Current/next event for calendar display
        self._event: CalendarEvent | None = None

        # Initialize database
        self._init_database()

    @property
    def device_info(self) -> dict[str, Any]:
        """Return device info."""
        return {
            "identifiers": {(DOMAIN, self._entry.entry_id)},
            "name": "HA Video Vision",
            "manufacturer": "HA Video Vision",
            "model": "AI Camera Analysis",
        }

    @property
    def event(self) -> CalendarEvent | None:
        """Return the current or next upcoming event."""
        return self._event

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return recent events as attributes for easy access."""
        events = self._get_recent_events(limit=10)
        return {
            "recent_events": [
                {
                    "camera": e["camera_name"],
                    "summary": e["summary"],
                    "time": e["start"],
                    "snapshot": e["snapshot_path"],
                }
                for e in events
            ],
            "total_events": self._get_event_count(),
        }

    def _init_database(self) -> None:
        """Initialize SQLite database."""
        conn = sqlite3.connect(self._db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    uid TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    description TEXT,
                    start TEXT NOT NULL,
                    end TEXT NOT NULL,
                    camera_entity TEXT,
                    camera_name TEXT,
                    snapshot_path TEXT,
                    person_detected INTEGER DEFAULT 0,
                    provider TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_start ON events(start)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_camera ON events(camera_entity)")
            conn.commit()
        finally:
            conn.close()

    def _get_retention_days(self) -> int:
        """Get retention period from config."""
        config = {**self._entry.data, **self._entry.options}
        return config.get(CONF_TIMELINE_RETENTION_DAYS, DEFAULT_TIMELINE_RETENTION_DAYS)

    def _cleanup_old_events(self) -> None:
        """Remove events older than retention period."""
        retention_days = self._get_retention_days()
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()

        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM events WHERE start < ?", (cutoff,)
            )
            if cursor.rowcount > 0:
                _LOGGER.debug("Cleaned up %d old timeline events", cursor.rowcount)
            conn.commit()
        finally:
            conn.close()

    async def async_add_event(
        self,
        camera_entity: str,
        camera_name: str,
        description: str,
        snapshot_path: str | None = None,
        person_detected: bool = False,
        provider: str | None = None,
    ) -> None:
        """Add a new event to the timeline."""
        now = dt_util.now()
        event_uid = str(uuid.uuid4())

        # Event duration: 1 minute (for calendar display)
        start = now.isoformat()
        end = (now + timedelta(minutes=1)).isoformat()

        # Truncate description for summary (calendar title)
        summary = description[:100] + "..." if len(description) > 100 else description

        def _insert():
            conn = sqlite3.connect(self._db_path)
            try:
                conn.execute(
                    """
                    INSERT INTO events
                    (uid, summary, description, start, end, camera_entity, camera_name,
                     snapshot_path, person_detected, provider)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event_uid,
                        summary,
                        description,
                        start,
                        end,
                        camera_entity,
                        camera_name,
                        snapshot_path,
                        1 if person_detected else 0,
                        provider,
                    ),
                )
                conn.commit()

                # Cleanup old events periodically
                self._cleanup_old_events()
            finally:
                conn.close()

        await self.hass.async_add_executor_job(_insert)

        # Update the calendar entity state
        self.async_schedule_update_ha_state(True)

        _LOGGER.debug(
            "Added timeline event for %s: %s", camera_name, summary[:50]
        )

    def _get_recent_events(self, limit: int = 10) -> list[dict]:
        """Get most recent events."""
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(
                "SELECT * FROM events ORDER BY start DESC LIMIT ?", (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            conn.close()

    def _get_event_count(self) -> int:
        """Get total event count."""
        conn = sqlite3.connect(self._db_path)
        try:
            cursor = conn.execute("SELECT COUNT(*) FROM events")
            return cursor.fetchone()[0]
        finally:
            conn.close()

    async def async_get_events(
        self,
        hass: HomeAssistant,
        start_date: datetime,
        end_date: datetime,
    ) -> list[CalendarEvent]:
        """Return calendar events within a datetime range."""
        def _fetch():
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                cursor = conn.execute(
                    """
                    SELECT * FROM events
                    WHERE start >= ? AND start <= ?
                    ORDER BY start DESC
                    """,
                    (start_date.isoformat(), end_date.isoformat()),
                )
                return [dict(row) for row in cursor.fetchall()]
            finally:
                conn.close()

        events = await self.hass.async_add_executor_job(_fetch)

        return [
            CalendarEvent(
                start=datetime.fromisoformat(e["start"]),
                end=datetime.fromisoformat(e["end"]),
                summary=e["summary"],
                description=e["description"],
                uid=e["uid"],
                location=e["snapshot_path"],  # Store snapshot path in location field
            )
            for e in events
        ]

    async def async_update(self) -> None:
        """Update the calendar entity with the most recent event."""
        def _get_current():
            now = datetime.now().isoformat()
            conn = sqlite3.connect(self._db_path)
            conn.row_factory = sqlite3.Row
            try:
                # Get the most recent event
                cursor = conn.execute(
                    "SELECT * FROM events ORDER BY start DESC LIMIT 1"
                )
                row = cursor.fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

        event_data = await self.hass.async_add_executor_job(_get_current)

        if event_data:
            self._event = CalendarEvent(
                start=datetime.fromisoformat(event_data["start"]),
                end=datetime.fromisoformat(event_data["end"]),
                summary=event_data["summary"],
                description=event_data["description"],
                uid=event_data["uid"],
            )
        else:
            self._event = None
