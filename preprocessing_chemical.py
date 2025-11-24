from pathlib import Path
import re
import pandas as pd

# -------------------------------------------------
# input paths
# -------------------------------------------------
EVENTS_PATH = "/Users/beatewais/PycharmProjects/edt-ts-chemical/data/process_event_data2.csv"
SENSOR_DIR = Path("/Users/beatewais/PycharmProjects/edt-ts-chemical/data/sensor_data")

# output paths
EVENTS_OUT = "/Users/beatewais/PycharmProjects/edt-ts-chemical/data/process_events3.csv"
SENSORS_OUT = "/Users/beatewais/PycharmProjects/edt-ts-chemical/data/sensor_long3.csv"

# -------------------------------------------------
# load and prepare event log
# -------------------------------------------------
print(f"Loading event log from {EVENTS_PATH}")

events = (
    pd.read_csv(EVENTS_PATH)
    .rename(
        columns={
            "CaseID": "case_id",
            "CompleteTimestamp": "timestamp",
            "lifecycle:transition": "lifecycle",
            "ActivityID": "activity",
            "Vessel": "vessel",
        }
    )
)

# clean up and sort
events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)
events = events.dropna(subset=["activity", "timestamp", "case_id"])
events = events.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

# instantaneous process steps
SINGLE_ACTIVITIES = {"Pump start", "Pump stop", "Pump adjustment"}


def build_intervals(df: pd.DataFrame) -> pd.DataFrame:
    intervals, open_stack = [], {}

    for _, row in df.iterrows():
        cid = row["case_id"]
        act = row["activity"]
        vessel = row["vessel"]
        ts = row["timestamp"]
        life = str(row.get("lifecycle", "")).lower().strip()
        key = (cid, act, vessel)

        if act in SINGLE_ACTIVITIES:
            intervals.append({
                "case_id": cid,
                "activity": act,
                "vessel": vessel,
                "event_timestamp": ts,
                "start_time": ts,
                "end_time": ts,
                "source": "instant",
            })
            continue

        if life == "start":
            open_stack.setdefault(key, []).append(row)
        elif life == "complete" and open_stack.get(key):
            start_row = open_stack[key].pop(0)
            intervals.append({
                "case_id": cid,
                "activity": act,
                "vessel": vessel,
                "event_timestamp": start_row["timestamp"],
                "start_time": start_row["timestamp"],
                "end_time": ts,
                "source": "paired",
            })
        else:
            intervals.append({
                "case_id": cid,
                "activity": act,
                "vessel": vessel,
                "event_timestamp": ts,
                "start_time": ts,
                "end_time": ts,
                "source": "unmatched",
            })

    # Handle unmatched starts
    for (cid, act, vessel), starts in open_stack.items():
        for s in starts:
            ts = s["timestamp"]
            intervals.append({
                "case_id": cid,
                "activity": act,
                "vessel": vessel,
                "event_timestamp": ts,
                "start_time": ts,
                "end_time": ts,
                "source": "unmatched_start",
            })

    out = pd.DataFrame(intervals)
    if not out.empty:
        out["duration_min"] = (
            (out["end_time"] - out["start_time"]).dt.total_seconds() / 60
        )
        out = out.sort_values(["case_id", "start_time", "end_time"]).reset_index(drop=True)
    return out


print("Building intervals...")
df_events = build_intervals(events)
print(f"Built {len(df_events)} intervals from {df_events['case_id'].nunique()} cases.")

df_events.to_csv(EVENTS_OUT, index=False)
print(f"Saved processed event log to {EVENTS_OUT}")

# -------------------------------------------------
# load and combine sensor data
# -------------------------------------------------
print(f"Loading sensor data from {SENSOR_DIR}")

sensor_frames = []
case_pattern = re.compile(r"(?:20|21|22)[A-Z]\d+\w*", re.IGNORECASE)


for path in SENSOR_DIR.glob("*.csv"):
    if not path.is_file():
        continue

    case_match = case_pattern.search(path.stem)
    case_id = case_match.group(0) if case_match else None

    print(f" -> Reading {path.name} (case_id={case_id or 'not found'})")

    try:
        df_s = pd.read_csv(path, sep=";")
    except Exception:
        df_s = pd.read_csv(path)

    if "New Timestamp" not in df_s.columns:
        raise ValueError(f"'New Timestamp' column not found in {path.name}")

    df_s = df_s.rename(columns={"New Timestamp": "timestamp"})
    df_s["timestamp"] = pd.to_datetime(df_s["timestamp"], errors="coerce", utc=True)
    df_s = df_s.dropna(subset=["timestamp"])
    df_s["case_id"] = case_id

    # vessel info
    vessel_vals = events.loc[events["case_id"] == case_id, "vessel"].dropna().unique()
    df_s["vessel"] = vessel_vals[0] if len(vessel_vals) > 0 else None

    # reshape to long format
    sensor_cols = [c for c in df_s.columns if c not in ["timestamp", "case_id", "vessel"]]
    long_df = (
        df_s.melt(id_vars=["timestamp", "case_id", "vessel"],
                  value_vars=sensor_cols,
                  var_name="sensor",
                  value_name="value")
    )
    sensor_frames.append(long_df)

if not sensor_frames:
    raise RuntimeError("No sensor files found!")

df_sensors = pd.concat(sensor_frames, ignore_index=True)
df_sensors = df_sensors.sort_values(["case_id", "vessel", "timestamp", "sensor"]).reset_index(drop=True)

print(f"Loaded {df_sensors['sensor'].nunique()} sensors, {len(df_sensors):,} readings total.")
df_sensors.to_csv(SENSORS_OUT, index=False)
print(f"Saved long-format sensor file to {SENSORS_OUT}")

print("Done.")