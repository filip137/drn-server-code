"""Streaming SQLite artifact for pulse-resolved characterization."""

from __future__ import annotations

import json
from pathlib import Path
import sqlite3
from typing import Iterable, Mapping, Sequence


TRAJECTORY_COLUMNS = (
    "trajectory_id", "preset", "corrupt_population", "controller",
    "start_protocol", "tolerance_ratio", "tolerance", "target_index",
    "target", "device_id", "repeat_id", "partition_name", "construction_seed",
    "repeat_seed", "conditioning_seed", "pulse_seed", "controller_seed",
    "corrupt", "conditioning_success",
    "conditioning_pulses", "conditioned_apparent", "conditioned_persistent",
    "sampled_lower_persistent", "sampled_upper_persistent",
    "accepted", "initialization_failed", "nonfinite", "budget_exhausted",
    "saturated", "endpoint_apparent", "endpoint_persistent",
    "residual_apparent", "residual_persistent", "set_count", "reset_count",
    "total_pulses", "verify_count", "reversals",
)
STORAGE_SCHEMA_VERSION = 2
VERIFY_EVENT_COLUMNS = (
    "trajectory_id", "verify_index", "apparent_before", "apparent_after",
    "persistent_after", "direction", "pulse_count", "signed_batch",
    "total_pulses",
)


class TrajectoryStore:
    """Own a transaction-safe machine-readable trajectory database."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self.connection.executescript(
            """
            CREATE TABLE metadata (
                key TEXT PRIMARY KEY,
                json_value TEXT NOT NULL
            );
            CREATE TABLE trajectories (
                trajectory_id INTEGER PRIMARY KEY,
                preset TEXT NOT NULL,
                corrupt_population INTEGER NOT NULL CHECK (corrupt_population IN (0, 1)),
                controller TEXT NOT NULL,
                start_protocol TEXT NOT NULL,
                tolerance_ratio REAL NOT NULL,
                tolerance REAL NOT NULL,
                target_index INTEGER NOT NULL,
                target REAL NOT NULL,
                device_id INTEGER NOT NULL,
                repeat_id INTEGER NOT NULL,
                partition_name TEXT NOT NULL,
                construction_seed INTEGER NOT NULL,
                repeat_seed INTEGER NOT NULL,
                conditioning_seed INTEGER NOT NULL,
                pulse_seed INTEGER NOT NULL,
                controller_seed INTEGER,
                corrupt INTEGER NOT NULL CHECK (corrupt IN (0, 1)),
                conditioning_success INTEGER NOT NULL CHECK (conditioning_success IN (0, 1)),
                conditioning_pulses INTEGER NOT NULL,
                conditioned_apparent REAL NOT NULL,
                conditioned_persistent REAL NOT NULL,
                sampled_lower_persistent REAL NOT NULL,
                sampled_upper_persistent REAL NOT NULL,
                accepted INTEGER CHECK (accepted IN (0, 1)),
                initialization_failed INTEGER CHECK (initialization_failed IN (0, 1)),
                nonfinite INTEGER CHECK (nonfinite IN (0, 1)),
                budget_exhausted INTEGER CHECK (budget_exhausted IN (0, 1)),
                saturated INTEGER CHECK (saturated IN (0, 1)),
                endpoint_apparent REAL,
                endpoint_persistent REAL,
                residual_apparent REAL,
                residual_persistent REAL,
                set_count INTEGER,
                reset_count INTEGER,
                total_pulses INTEGER,
                verify_count INTEGER,
                reversals INTEGER,
                UNIQUE (
                    preset, corrupt_population, controller, start_protocol,
                    tolerance_ratio, target_index, device_id, repeat_id
                )
            );
            CREATE TABLE verify_events (
                trajectory_id INTEGER NOT NULL REFERENCES trajectories(trajectory_id),
                verify_index INTEGER NOT NULL,
                apparent_before REAL,
                apparent_after REAL,
                persistent_after REAL,
                direction INTEGER NOT NULL CHECK (direction IN (-1, 0, 1)),
                pulse_count INTEGER NOT NULL,
                signed_batch INTEGER NOT NULL,
                total_pulses INTEGER NOT NULL,
                PRIMARY KEY (trajectory_id, verify_index)
            );
            CREATE INDEX trajectory_condition_idx ON trajectories (
                controller, start_protocol, tolerance_ratio, target_index,
                partition_name, accepted, corrupt
            );
            CREATE INDEX verify_trajectory_idx ON verify_events (trajectory_id);
            """
        )
        self.connection.execute(
            "INSERT INTO metadata(key, json_value) VALUES (?, ?)",
            (
                "storage_schema",
                json.dumps(
                    {
                        "schema": "ebl.ibm_reram.trajectory_database",
                        "schema_version": STORAGE_SCHEMA_VERSION,
                        "trajectory_columns": list(TRAJECTORY_COLUMNS),
                        "verify_event_columns": list(VERIFY_EVENT_COLUMNS),
                    },
                    allow_nan=False,
                    sort_keys=True,
                ),
            ),
        )
        self.connection.commit()
        self._next_trajectory_id = 1

    def close(self) -> None:
        self.connection.commit()
        self.connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.connection.close()

    def __enter__(self) -> "TrajectoryStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc is None:
            self.close()
        else:
            self.connection.rollback()
            self.connection.close()

    def insert_metadata(self, records: Mapping[str, str]) -> None:
        self.connection.executemany(
            "INSERT INTO metadata(key, json_value) VALUES (?, ?)",
            tuple((key, value) for key, value in records.items()),
        )
        self.connection.commit()

    def begin_trajectories(self, rows: Sequence[Mapping[str, object]]) -> list[int]:
        ids = list(
            range(self._next_trajectory_id, self._next_trajectory_id + len(rows))
        )
        self._next_trajectory_id += len(rows)
        columns = (
            "trajectory_id", "preset", "corrupt_population", "controller",
            "start_protocol", "tolerance_ratio", "tolerance", "target_index",
            "target", "device_id", "repeat_id", "partition_name",
            "construction_seed", "repeat_seed", "conditioning_seed", "pulse_seed",
            "controller_seed", "corrupt",
            "conditioning_success", "conditioning_pulses", "conditioned_apparent",
            "conditioned_persistent", "sampled_lower_persistent",
            "sampled_upper_persistent",
        )
        placeholders = ",".join("?" for _ in columns)
        self.connection.executemany(
            f"INSERT INTO trajectories ({','.join(columns)}) VALUES ({placeholders})",
            [
                tuple(
                    trajectory_id if column == "trajectory_id" else row[column]
                    for column in columns
                )
                for trajectory_id, row in zip(ids, rows)
            ],
        )
        return ids

    def insert_events(self, rows: Iterable[Sequence[object]]) -> None:
        self.connection.executemany(
            """
            INSERT INTO verify_events (
                trajectory_id, verify_index, apparent_before, apparent_after,
                persistent_after, direction, pulse_count, signed_batch,
                total_pulses
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def finish_trajectories(
        self,
        rows: Iterable[Mapping[str, object]],
    ) -> None:
        columns = (
            "accepted", "initialization_failed", "nonfinite", "budget_exhausted",
            "saturated", "endpoint_apparent", "endpoint_persistent",
            "residual_apparent", "residual_persistent", "set_count", "reset_count",
            "total_pulses", "verify_count", "reversals", "trajectory_id",
        )
        self.connection.executemany(
            """
            UPDATE trajectories SET
                accepted=?, initialization_failed=?, nonfinite=?, budget_exhausted=?,
                saturated=?, endpoint_apparent=?, endpoint_persistent=?,
                residual_apparent=?, residual_persistent=?, set_count=?, reset_count=?,
                total_pulses=?, verify_count=?, reversals=?
            WHERE trajectory_id=?
            """,
            [tuple(row[column] for column in columns) for row in rows],
        )
        self.connection.commit()


__all__ = [
    "STORAGE_SCHEMA_VERSION",
    "TRAJECTORY_COLUMNS",
    "TrajectoryStore",
    "VERIFY_EVENT_COLUMNS",
]
