from pathlib import Path
import re
import pandas as pd

# input paths
EVENTS_PATH = "/Users/beatewais/PycharmProjects/edt-ts-chemical/data/process_event_data2.csv"
SENSOR_DIR = Path("/Users/beatewais/PycharmProjects/edt-ts-chemical/data/sensor_data")

# output paths
EVENTS_OUT = "/Users/beatewais/PycharmProjects/edt-ts-chemical/data/process_events1.csv"
SENSORS_OUT = "/Users/beatewais/PycharmProjects/edt-ts-chemical/data/sensor_long1.csv"

# -------------------------------------------------
# load and prepare event log
# -------------------------------------------------
print(f"loading event log from {EVENTS_PATH}")

events = (
    pd.read_csv(EVENTS_PATH)
    .rename(
        columns={
            "CaseID": "case_id",
            "CompleteTimestamp": "timestamp",
            "lifecycle:transition": "lifecycle",
            "ActivityID": "activity",
            "Vessel": "resource",
        }
    )
)

# clean up and sort
events["timestamp"] = pd.to_datetime(events["timestamp"], errors="coerce", utc=True)
events = events.dropna(subset=["activity", "timestamp", "case_id"])
events = events.sort_values(["case_id", "timestamp"]).reset_index(drop=True)

# activities that are instantaneous
SINGLE_ACTIVITIES = {"Pump start", "Pump stop", "Pump adjustment"}


def build_intervals(df: pd.DataFrame) -> pd.DataFrame:
    # turn start/complete transitions into time intervals
    intervals, open_stack = [], {}

    for _, row in df.iterrows():
        cid = row["case_id"]
        act = row["activity"]
        res = row["resource"]
        ts = row["timestamp"]
        life = str(row.get("lifecycle", "")).lower().strip()
        key = (cid, act, res)

        # single point events
        if act in SINGLE_ACTIVITIES:
            intervals.append({
                "case_id": cid,
                "activity": act,
                "resource": res,
                "event_timestamp": ts,
                "start_time": ts,
                "end_time": ts,
                "source": "instant",
            })
            continue

        # start–complete matching
        if life == "start":
            open_stack.setdefault(key, []).append(row)
        elif life == "complete" and open_stack.get(key):
            start_row = open_stack[key].pop(0)
            intervals.append({
                "case_id": cid,
                "activity": act,
                "resource": res,
                "event_timestamp": start_row["timestamp"],
                "start_time": start_row["timestamp"],
                "end_time": ts,
                "source": "paired",
            })
        else:
            intervals.append({
                "case_id": cid,
                "activity": act,
                "resource": res,
                "event_timestamp": ts,
                "start_time": ts,
                "end_time": ts,
                "source": "unmatched",
            })

    # handle unmatched starts
    for (cid, act, res), starts in open_stack.items():
        for s in starts:
            ts = s["timestamp"]
            intervals.append({
                "case_id": cid,
                "activity": act,
                "resource": res,
                "event_timestamp": ts,
                "start_time": ts,
                "end_time": ts,
                "source": "unmatched_start",
            })

    out = pd.DataFrame(intervals)
    if not out.empty:
        out["duration_min"] = (out["end_time"] - out["start_time"]).dt.total_seconds() / 60
        out = out.sort_values(["case_id", "start_time", "end_time"]).reset_index(drop=True)
    return out


print("building intervals...")
df_events = build_intervals(events)
print(
    f"built {len(df_events)} intervals from {df_events['case_id'].nunique()} cases"
)

df_events.to_csv(EVENTS_OUT, index=False)
print(f"saved to {EVENTS_OUT}")

# -------------------------------------------------
# load and combine sensor data
# -------------------------------------------------
print(f"loading sensor data from {SENSOR_DIR}")

sensor_frames = []
for path in SENSOR_DIR.glob("*"):
    if not path.is_file():
        continue

    try:
        df_s = pd.read_csv(path, sep=";", engine="python")
    except Exception:
        df_s = pd.read_csv(path)

    # find timestamp column
    ts_col = next(
        (c for c in df_s.columns if str(c).strip().lower() in {"timestamp", "time", "new timestamp"}),
        None,
    )
    if not ts_col:
        raise ValueError(f"no timestamp column found in {path}")

    df_s = df_s.rename(columns={ts_col: "timestamp"})
    df_s["timestamp"] = pd.to_datetime(df_s["timestamp"], errors="coerce", utc=True)
    df_s = df_s.dropna(subset=["timestamp"])

    sensor_cols = [c for c in df_s.columns if c != "timestamp"]
    for sc in sensor_cols:
        temp = df_s[["timestamp", sc]].rename(columns={sc: "value"})
        temp["sensor"] = sc
        sensor_frames.append(temp)

if not sensor_frames:
    raise RuntimeError("no sensor files found")

df_sensors = pd.concat(sensor_frames, ignore_index=True)
df_sensors = df_sensors.sort_values(["timestamp", "sensor"]).reset_index(drop=True)

print(f"loaded {df_sensors['sensor'].nunique()} sensors, {len(df_sensors):,} readings total")

df_sensors.to_csv(SENSORS_OUT, index=False)
print(f"saved to {SENSORS_OUT}")

print("done.")