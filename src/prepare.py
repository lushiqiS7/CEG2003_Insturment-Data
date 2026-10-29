"""Prepare pseudonymous event data, aggregates, and a prospective driver cohort."""

import hashlib
import json
import uuid
from collections import Counter
from datetime import datetime

import numpy as np
import openpyxl
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .common import ROOT, config, logger, save_json
from .excel_stream import EXPECTED, alarm_rows, normalize_key, parse_date, sha256
from .features import attach_target, build_features

LOG = logger("prepare")
SCHEMA = pa.schema(
    [
        ("captain_id", pa.string()),
        ("route", pa.string()),
        ("alarm_type", pa.string()),
        ("speed", pa.float32()),
        ("start", pa.timestamp("us")),
        ("end", pa.timestamp("us")),
        ("duration", pa.float32()),
        ("day", pa.date32()),
        ("night", pa.bool_()),
    ]
)


def read_incidents(path, mapping, settings):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    sheet = book["Database"]
    values = iter(sheet.values)
    headers = next(values)
    if list(headers[:6]) != [
        "BOCC Report",
        "Duplicate / Different Acc/ WSH / Near Miss / Need Not Report",
        "Time of Accident",
        "Date of Accident",
        "Service No",
        "Employee No",
    ]:
        raise ValueError("Unexpected accident schema")
    incidents = []
    for row in values:
        if not any(v not in (None, "") for v in row[:6]):
            continue
        date = parse_date(row[3])
        if date is None:
            raise ValueError("Incident date could not be parsed")
        flag = str(row[1] or "").strip().upper()
        incidents.append(
            {
                "captain_id": mapping.get(normalize_key(row[5])),
                "day": pd.Timestamp(date.date()),
                "flag": flag or "UNSPECIFIED",
                "included": flag not in settings["excluded_incident_flags"],
                "route": normalize_key(row[4]),
            }
        )
    book.close()
    return pd.DataFrame(incidents)


def prepare():
    settings = config()
    destination = ROOT / "data" / "processed"
    destination.mkdir(parents=True, exist_ok=True)
    mapping_path = destination / "driver_map.parquet"
    mapping = (
        dict(pd.read_parquet(mapping_path).itertuples(index=False, name=None))
        if mapping_path.exists()
        else {}
    )
    used = set(mapping.values())
    sources = {key: ROOT / settings[key] for key in ("alarm_file", "accident_file")}
    for path in sources.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = {
        key: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
        for key, path in sources.items()
    }
    seen, counts, missing, by_sheet = set(), Counter(), Counter(), Counter()
    buffer = []
    temporary = destination / "events.tmp.parquet"
    final = destination / "events.parquet"
    LOG.info("Streaming all alarm sheets; duplicate source copy is not ingested")
    with pq.ParquetWriter(temporary, SCHEMA, compression="zstd") as writer:
        for sheet, row_number, row in alarm_rows(sources["alarm_file"]):
            if row_number == 1:
                if row != EXPECTED:
                    raise ValueError(f"Unexpected schema in {sheet}")
                continue
            if not any(v not in (None, "") for v in row):
                continue
            counts["raw_events"] += 1
            by_sheet[sheet] += 1
            for name, value in zip(EXPECTED, row):
                if value in (None, ""):
                    missing[name] += 1
            digest = hashlib.blake2b(
                json.dumps(row, separators=(",", ":")).encode(), digest_size=16
            ).digest()
            if digest in seen:
                counts["exact_duplicate_rows_removed"] += 1
                continue
            seen.add(digest)
            driver = normalize_key(row[0])
            if driver and driver not in mapping:
                candidate = "BC-" + uuid.uuid4().hex[:8].upper()
                while candidate in used:
                    candidate = "BC-" + uuid.uuid4().hex[:8].upper()
                mapping[driver] = candidate
                used.add(candidate)
            start, end = parse_date(row[4]), parse_date(row[5])
            try:
                speed, duration = float(row[3]), float(row[6])
            except (ValueError, TypeError):
                counts["invalid_events_quarantined"] += 1
                continue
            if (
                start is None
                or end is None
                or end < start
                or not np.isfinite(speed)
                or not np.isfinite(duration)
                or speed < 0
                or duration < 0
                or abs((end - start).total_seconds() - duration) > 1
            ):
                counts["invalid_events_quarantined"] += 1
                continue
            if row[2] is None:
                counts["invalid_events_quarantined"] += 1
                continue
            night = (
                start.hour >= settings["night_start_hour"]
                or start.hour < settings["night_end_hour"]
            )
            buffer.append(
                {
                    "captain_id": mapping.get(driver),
                    "route": normalize_key(row[1]),
                    "alarm_type": row[2],
                    "speed": speed,
                    "start": start,
                    "end": end,
                    "duration": duration,
                    "day": start.date(),
                    "night": night,
                }
            )
            counts["clean_events"] += 1
            if len(buffer) == 100000:
                writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
                buffer.clear()
                LOG.info("Prepared %s events", f"{counts['clean_events']:,}")
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA))
    assert counts["raw_events"] == (
        counts["clean_events"]
        + counts["exact_duplicate_rows_removed"]
        + counts["invalid_events_quarantined"]
    )
    temporary.replace(final)
    pd.DataFrame(mapping.items(), columns=["driver_key", "captain_id"]).to_parquet(
        mapping_path, index=False
    )
    events = pd.read_parquet(final)
    events["day"] = pd.to_datetime(events["day"])
    incidents = read_incidents(sources["accident_file"], mapping, settings)
    incidents.to_parquet(destination / "incidents.parquet", index=False)
    features = attach_target(build_features(events, settings), incidents, settings)
    features.to_parquet(destination / "driver_features.parquet", index=False)
    events.groupby(["day", "alarm_type"], observed=True, dropna=False).size().rename(
        "events"
    ).reset_index().to_parquet(destination / "daily.parquet", index=False)
    events.groupby(["captain_id", "day", "alarm_type"], observed=True, dropna=False).size().rename(
        "events"
    ).reset_index().to_parquet(destination / "driver_daily.parquet", index=False)
    known = features.loc[features["label_known"]]
    summary = {
        "created_at": datetime.now().astimezone().isoformat(),
        "sources": manifest,
        "settings": settings,
        "counts": dict(counts),
        "raw_missing": dict(missing),
        "raw_sheet_counts": dict(by_sheet),
        "feature_drivers": len(features),
        "model_cohort_drivers": len(known),
        "positive_drivers": int(known["target"].sum()),
        "unknown_label_drivers": int((~features["label_known"]).sum()),
        "incident_records": len(incidents),
        "excluded_incident_records": int((~incidents["included"]).sum()),
        "unmatched_incident_records": int(incidents["captain_id"].isna().sum()),
        "feature_min": events["start"].min().isoformat(),
        "feature_max": events["start"].max().isoformat(),
        "alarm_types": events["alarm_type"].value_counts().to_dict(),
        "clean_missing_driver": int(events["captain_id"].isna().sum()),
        "clean_missing_route": int(events["route"].isna().sum()),
    }
    save_json(ROOT / "outputs" / "preparation.json", summary)
    LOG.info(
        "Prepared %s clean events; %s model drivers; %s positive drivers; %s ambiguous labels excluded",
        counts["clean_events"],
        len(known),
        known["target"].sum(),
        (~features["label_known"]).sum(),
    )


if __name__ == "__main__":
    prepare()
