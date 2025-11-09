import re
from pathlib import Path
import pandas as pd

# -------------------------------------------------
# 0) Load and prepare event log
# -------------------------------------------------
EVENTS_PATH = Path("data/process_event_data2.csv")

df = (
    pd.read_csv(EVENTS_PATH)
      .rename(columns={
          "CompleteTimestamp": "timestamp",
          "lifecycle:transition": "lifecycle",
          "ActivityID": "activity",
          "Vessel": "resource",
          "CaseID": "case_id",
      })
)

df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

# -------------------------------------------------
# 1) Build activity intervals (including singletons)
# -------------------------------------------------
SINGLE_ACTIVITIES = {"Pump start", "Pump stop", "Pump adjustment"}



def build_intervals(case_df: pd.DataFrame) -> pd.DataFrame:
    """Convert start/complete lifecycle events into start-end intervals."""
    case_df = case_df.sort_values("timestamp")
    intervals, open_stack = [], {}

    for _, row in case_df.iterrows():
        act, res, ts = row["activity"], row["resource"], row["timestamp"]
        key = (act, res)
        lifecycle = (row.get("lifecycle") or "").lower()

        # Handle one-off events
        if act in SINGLE_ACTIVITIES:
            intervals.append({
                "case_id": row["case_id"],
                "resource": res,
                "activity": act,
                "event_timestamp": ts,
                "start_time": ts,
                "end_time": ts,
                "source": "instant",
            })
            continue

        # Handle paired start/complete logic
        if lifecycle == "start":
            open_stack.setdefault(key, []).append(row)
        elif lifecycle == "complete" and open_stack.get(key):
            start_row = open_stack[key].pop(0)
            intervals.append({
                "case_id": row["case_id"],
                "resource": res,
                "activity": act,
                "event_timestamp": start_row["timestamp"],
                "start_time": start_row["timestamp"],
                "end_time": ts,
                "source": "paired",
            })
        else:
            # No matching lifecycle or unmatched complete
            intervals.append({
                "case_id": row["case_id"],
                "resource": res,
                "activity": act,
                "event_timestamp": ts,
                "start_time": ts,
                "end_time": ts,
                "source": "unmatched_or_no_lifecycle",
            })

    # Handle unmatched starts
    for (act, res), starts in open_stack.items():
        for start_row in starts:
            ts = start_row["timestamp"]
            intervals.append({
                "case_id": start_row["case_id"],
                "resource": res,
                "activity": act,
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


intervals = (
    df.groupby("case_id", group_keys=False)
      .apply(build_intervals)
      .reset_index(drop=True)
)

# -------------------------------------------------
# 2) Load sensor data files
# -------------------------------------------------
SENSOR_FOLDER = Path("data/sensor_data")
FNAME_RE = re.compile(r"^Sensor data (.+?)(?:\.csv|\.txt)?$", re.IGNORECASE)

sensor_frames = []
for path in SENSOR_FOLDER.glob("*"):
    if not path.is_file():
        continue
    m = FNAME_RE.match(path.stem)
    if not m:
        continue

    case_id = m.group(1)
    df_sensor = pd.read_csv(path, sep=";", engine="python")

    # Find timestamp column
    ts_col = next(
        (c for c in df_sensor.columns if str(c).strip().lower() in {"new timestamp", "timestamp", "time"}),
        None
    )
    if ts_col is None:
        raise ValueError(f"No timestamp column found in {path}")

    df_sensor = (
        df_sensor
        .rename(columns={ts_col: "sensor_time"})
        .assign(sensor_time=lambda d: pd.to_datetime(d["sensor_time"]), case_id=case_id)
        .sort_values(["case_id", "sensor_time"])
        .reset_index(drop=True)
    )
    sensor_frames.append(df_sensor)

if not sensor_frames:
    raise RuntimeError("No sensor files found in data/sensor_data")

sensors = pd.concat(sensor_frames, ignore_index=True)
sensor_cols = [c for c in sensors.columns if c not in ["case_id", "sensor_time"]]

# -------------------------------------------------
# 3) Collect sensor values per interval
# -------------------------------------------------
def listify_sensor_values(case_intervals: pd.DataFrame, case_sensors: pd.DataFrame) -> pd.DataFrame:
    """Return one list column per sensor with all sensor values within each interval."""
    out = {col: [] for col in sensor_cols}
    t = case_sensors["sensor_time"].values if case_sensors is not None else None

    for _, row in case_intervals.iterrows():
        if t is None:
            for c in sensor_cols:
                out[c].append([])
            continue

        mask = (t >= row["start_time"]) & (t <= row["end_time"])
        sub = case_sensors.loc[mask, sensor_cols]

        for c in sensor_cols:
            out[c].append(sub[c].tolist() if not sub.empty else [])

    return pd.DataFrame(out, index=case_intervals.index)


intervals_by_case = dict(tuple(intervals.groupby("case_id", sort=False)))
sensors_by_case = dict(tuple(sensors.groupby("case_id", sort=False)))

sensor_blocks = []
for cid, case_int in intervals_by_case.items():
    block = listify_sensor_values(case_int, sensors_by_case.get(cid))
    sensor_blocks.append(block)

sensor_lists = pd.concat(sensor_blocks).sort_index()

# -------------------------------------------------
# 4) Combine intervals and sensor data
# -------------------------------------------------
events_with_sensors = pd.concat([intervals, sensor_lists], axis=1)

# Merge lifecycle info back for traceability
df_base = df.rename(columns={"timestamp": "event_timestamp"})
events_with_sensors = events_with_sensors.merge(
    df_base[["case_id", "resource", "activity", "event_timestamp", "lifecycle"]],
    on=["case_id", "resource", "activity", "event_timestamp"],
    how="left",
)

# -------------------------------------------------
# 5) Export
# -------------------------------------------------
OUT_PATH = Path("data/event_with_sensor_lists2.csv")
events_with_sensors.to_csv(OUT_PATH, index=False)

print(f"Finished generating events with sensor lists → {OUT_PATH}")