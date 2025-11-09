import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn import tree
import graphviz
from multiprocessing import freeze_support
from tree_code import learn_tree
from joblib import Parallel, delayed


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
SENSOR_COLS = [
    "Filter 1 DeltaP",
    "Filter 1 Inlet Pressure",
    "Filter 2 DeltaP",
    "Pump Circulation Flow",
    "Tank Pressure"
]

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def calc_slope(x: np.ndarray) -> float:
    return 0.0 if len(x) < 2 else (x[-1] - x[0]) / (len(x) - 1)

def calc_autocorr(x: np.ndarray, lag=1) -> float:
    if len(x) <= lag:
        return np.nan
    x = x - x.mean()
    denom = np.dot(x, x)
    return np.nan if denom == 0 else np.dot(x[:-lag], x[lag:]) / denom

def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    sx, sy = x.std(ddof=0), y.std(ddof=0)
    return np.nan if sx == 0 or sy == 0 else np.corrcoef(x, y)[0, 1]


def enrich_event_log_with_context(df, case_col="case_id", activity_col="activity", n_lookback=3):
    df = df.sort_values([case_col, "event_timestamp"]).copy()
    context_col = []

    for _, case_df in df.groupby(case_col, sort=False):
        acts = case_df[activity_col].to_numpy()
        temp_ctx = []
        for i, act in enumerate(acts):
            if act != "Pump adjustment":
                temp_ctx.append(np.nan)
            else:
                prev_window = acts[max(0, i - n_lookback): i]
                if "Pump adjustment" in prev_window:
                    temp_ctx.append("repeat_adjustment")
                else:
                    temp_ctx.append("first_adjustment")
        context_col.extend(temp_ctx)

    df["adjustment_type"] = context_col
    return df

def prune_features(
    X: pd.DataFrame,
    y: pd.Series = None,
    importance_threshold: float = 0.0,
    corr_threshold: float = 0.9,
    variance_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Prune redundant or uninformative features using variance, correlation, and importance.
    Keeps exactly one feature per highly correlated cluster.

    Parameters
    ----------
    X : pd.DataFrame
        Features DataFrame.
    y : pd.Series, optional
        Target labels (needed for importance threshold).
    importance_threshold : float
        Minimum importance to keep (if y provided).
    corr_threshold : float
        Pearson correlation threshold for pruning (0.9 = highly correlated).
    variance_threshold : float
        Drop features with variance <= threshold (default 0 — keeps all).

    Returns
    -------
    pd.DataFrame : reduced feature set
    """
    X_pruned = X.copy()

    # --- 1️⃣ Drop low-variance columns ---------------------------------------
    if variance_threshold > 0:
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(X_pruned.fillna(0))
        X_pruned = X_pruned.loc[:, selector.get_support()]
        print(f"Variance pruning → kept {X_pruned.shape[1]} features")

    # --- 2️⃣ Correlation-based pruning ---------------------------------------
    corr_matrix = X_pruned.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop_corr = [column for column in upper.columns if any(upper[column] > corr_threshold)]
    if to_drop_corr:
        print(f"Correlation pruning → dropping {len(to_drop_corr)} correlated features")
        X_pruned = X_pruned.drop(columns=to_drop_corr)

    # --- 3️⃣ Optional importance pruning -------------------------------------
    if y is not None and importance_threshold > 0:
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced"
        )
        rf.fit(X_pruned.fillna(0), y)
        importances = pd.Series(rf.feature_importances_, index=X_pruned.columns)
        keep_cols = importances[importances > importance_threshold].index
        removed = len(importances) - len(keep_cols)
        X_pruned = X_pruned[keep_cols]
        print(f"Importance pruning → removed {removed} low-importance features")

    print(f"✅ Final feature count after pruning: {X_pruned.shape[1]}")
    return X_pruned
def verify_consecutive_pump_adjustments(df, case_col="case_id", activity_col="activity"):
    """
    Identifies and inspects sequences where 'Pump adjustment' immediately
    follows another 'Pump adjustment' within the same case.

    Returns a DataFrame of those consecutive pairs and a count per case.
    """
    df_sorted = df.sort_values([case_col, "event_timestamp"]).copy()
    df_sorted["next_activity"] = df_sorted.groupby(case_col)[activity_col].shift(-1)

    # Rows where current and next event are both "Pump adjustment"
    consecutive = df_sorted[
        (df_sorted[activity_col] == "Pump adjustment")
        & (df_sorted["next_activity"] == "Pump adjustment")
    ]

    # count per case — are some cases responsible for many of them?
    per_case = consecutive[case_col].value_counts().rename("count")
    print(f"\nTotal consecutive Pump Adjustment pairs: {len(consecutive)}")
    print(f"Distinct cases involved: {per_case.shape[0]}")
    print("\nTop few cases with most consecutive pairs:")
    print(per_case.head())

    return consecutive

def count_predecessor_events(df, case_col="case_id", activity_col="activity"):
    """
    Counts which activities directly precede a 'Pump adjustment' within each process case.

    Parameters
    ----------
    df : pd.DataFrame
        Event log with at least: [case_id, activity_name, timestamp].
        Must be ordered chronologically per case.
    case_col : str, default="case_id"
        Column that identifies each process instance.
    activity_col : str, default="activity"
        Column containing the activity/event name.

    Returns
    -------
    pd.Series
        Counts of preceding activities for all 'Pump adjustment' events,
        sorted descending.
    """
    # Ensure chronological ordering within each case
    df_sorted = df.sort_values([case_col, "event_timestamp"]).copy()

    # Shift the activity names to align each event with its predecessor
    df_sorted["prev_activity"] = (
        df_sorted.groupby(case_col)[activity_col].shift(1)
    )

    # Filter only rows where the current activity is 'Pump adjustment'
    mask = df_sorted[activity_col] == "Pump adjustment"
    preceding = df_sorted.loc[mask, "prev_activity"]

    # Count and sort
    counts = preceding.value_counts().sort_values(ascending=False)
    return counts

# ------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------

def extract_window_features_fast(df_events, df_signals, windows=(45, 60), n_jobs=-1):
    """
    Faster alternative to extract_window_features using grouped access and parallelization.
    Keeps same output format and naming.
    """
    use_event_id_col = "event_id" in df_events.columns
    event_ids = df_events["event_id"].values if use_event_id_col else df_events.index
    results = []

    # --- group signals by case to avoid repeated DataFrame filtering ---
    grouped_signals = {cid: g.sort_values("time_index") for cid, g in df_signals.groupby("case_id")}
    all_sensors = df_signals["sensor"].unique()
    has_flow = "Flow" in all_sensors

    def process_event(ev):
        cid = ev["case_id"]
        if cid not in grouped_signals:
            return None
        case_signals = grouped_signals[cid]

        eid = ev["event_id"] if use_event_id_col else ev.name
        feat_row = {"id": eid}

        for w in windows:
            start = ev["event_timestamp"] - pd.Timedelta(minutes=w)
            end = ev["event_timestamp"]
            win = case_signals.loc[
                (case_signals["time_index"] >= start)
                & (case_signals["time_index"] < end)
            ]
            if win.empty:
                continue

            sensor_groups = {s: g["value"].values for s, g in win.groupby("sensor")}
            # --- per sensor features ---
            for sensor, vals in sensor_groups.items():
                name = sensor.replace(" ", "_")
                if len(vals) == 0:
                    continue

                mean, std = vals.mean(), vals.std(ddof=0)
                minv, maxv = vals.min(), vals.max()
                start_val, end_val = vals[0], vals[-1]

                feat_row.update({
                    f"{name}__mean_win{w}m": mean,
                    f"{name}__std_win{w}m": std,
                    f"{name}__min_win{w}m": minv,
                    f"{name}__max_win{w}m": maxv,
                    f"{name}__range_win{w}m": maxv - minv,
                    f"{name}__slope_win{w}m": calc_slope(vals),
                    f"{name}__last_minus_first_win{w}m": end_val - start_val,
                    f"{name}__ratio_last_first_win{w}m": end_val / start_val if start_val != 0 else np.nan,
                    f"{name}__cv_win{w}m": std / mean if mean != 0 else np.nan,
                    f"{name}__autocorr_lag1_win{w}m": calc_autocorr(vals, lag=1),
                })

                if len(vals) > 1:
                    diffs = np.abs(np.diff(vals))
                    feat_row[f"{name}__max_jump_win{w}m"] = diffs.max()
                    feat_row[f"{name}__num_jumps_gt1p_win{w}m"] = (diffs > 0.01 * abs(mean)).sum()
                else:
                    feat_row[f"{name}__max_jump_win{w}m"] = 0.0
                    feat_row[f"{name}__num_jumps_gt1p_win{w}m"] = 0

                if has_flow and "Flow" in sensor_groups:
                    denom = sensor_groups["Flow"].mean()
                    if denom != 0:
                        feat_row[f"{name}__mean_normByFlow_win{w}m"] = mean / denom

            # --- pairwise correlations (only compute if >1 sensor present) ---
            sensors = list(sensor_groups.keys())
            if len(sensors) > 1:
                for s1, s2 in itertools.combinations(sensors, 2):
                    v1, v2 = sensor_groups[s1], sensor_groups[s2]
                    if len(v1) > 1 and len(v1) == len(v2):
                        corr = safe_corr(v1, v2)
                        feat_row[f"corr_{s1.replace(' ','_')}__{s2.replace(' ','_')}_win{w}m"] = corr

        return feat_row if len(feat_row) > 1 else None

    # --- parallel event loop ---
    feats_list = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(process_event)(row) for _, row in df_events.iterrows()
    )

    feats_list = [f for f in feats_list if f is not None]

    if not feats_list:
        return pd.DataFrame(index=event_ids)

    final_feats = pd.DataFrame(feats_list).set_index("id").reindex(event_ids)

    # --- optional trend & stability comparison for 10m vs 60m ---
    for sensor in all_sensors:
        nm = sensor.replace(" ", "_")
        if f"{nm}__slope_win10m" in final_feats and f"{nm}__slope_win60m" in final_feats:
            final_feats[f"{nm}__slope_accel"] = (
                final_feats[f"{nm}__slope_win10m"] - final_feats[f"{nm}__slope_win60m"]
            )
        if f"{nm}__mean_win10m" in final_feats and f"{nm}__mean_win60m" in final_feats:
            final_feats[f"{nm}__mean_diff10_60"] = (
                final_feats[f"{nm}__mean_win10m"] - final_feats[f"{nm}__mean_win60m"]
            )
        if f"{nm}__std_win10m" in final_feats and f"{nm}__std_win60m" in final_feats:
            final_feats[f"{nm}__std_ratio10_60"] = (
                final_feats[f"{nm}__std_win10m"] / (final_feats[f"{nm}__std_win60m"] + 1e-6)
            )
        if f"{nm}__cv_win10m" in final_feats and f"{nm}__cv_win60m" in final_feats:
            final_feats[f"{nm}__cv_ratio10_60"] = (
                final_feats[f"{nm}__cv_win10m"] / (final_feats[f"{nm}__cv_win60m"] + 1e-6)
            )

    # --- drop columns with no variance ---
    nunique = final_feats.nunique(dropna=True)
    final_feats = final_feats.drop(columns=nunique[nunique <= 1].index)

    return final_feats

def engineer_domain_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add domain-specific binary or derived features reflecting physical process heuristics."""
    X_new = X.copy()

    # --- Existing domain heuristics ---
    if "Tank_Pressure__slope_win10m" in X:
        X_new["feat_tank_pressure_rise"] = (X["Tank_Pressure__slope_win10m"] > 0.05).astype(int)

    if "Filter_1_DeltaP__max_jump_win60m" in X:
        X_new["feat_filter1_dp_spike"] = (X["Filter_1_DeltaP__max_jump_win60m"] > 0.2).astype(int)

    if {"Filter_1_Inlet_Pressure__mean_win60m", "Filter_2_DeltaP__mean_win60m"}.issubset(X.columns):
        ratio = X["Filter_1_Inlet_Pressure__mean_win60m"] / (X["Filter_2_DeltaP__mean_win60m"] + 1e-6)
        X_new["feat_inlet_outlet_imbalance"] = (ratio > 3).astype(int)

    if "Tank_Pressure__cv_win60m" in X:
        X_new["feat_tank_pressure_unstable"] = (X["Tank_Pressure__cv_win60m"] > 0.05).astype(int)

    corr_col = "corr_Pump_Circulation_Flow__Tank_Pressure_win60m"
    if corr_col in X:
        X_new["feat_flow_pressure_uncoupled"] = (X[corr_col] < 0.2).astype(int)

    # --- Added only the impactful new feature ---
    if "Filter_1_Inlet_Pressure__mean_win60m" in X:
        # Rolling skewness over past events as a signal-of-variability proxy
        X_new["feat_filter1_pressure_skew_evt"] = (
            X["Filter_1_Inlet_Pressure__mean_win60m"]
            .rolling(window=10, min_periods=5)
            .apply(lambda x: pd.Series(x).skew(), raw=False)
            .fillna(0)
        )

    return X_new
# ------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------
def parse_array(x):
    if isinstance(x, str) and x.startswith("["):
        try:
            return np.array(eval(x))
        except Exception:
            return np.array([])
    return np.array([])

def build_long_format_sensor_data(df):
    """Convert event rows with sensor arrays to long-format time series."""
    df = df.copy()
    for col in SENSOR_COLS:
        df[col] = df[col].apply(parse_array)

    records = []
    for idx, row in df.iterrows():
        for col in SENSOR_COLS:
            values = row[col]
            if len(values) == 0:
                continue
            base_time = row["event_timestamp"]
            records.extend({
                "event_id": idx,
                "case_id": row["case_id"],
                "sensor": col,
                "time_index": base_time + pd.to_timedelta(int(t), unit="s"),
                "value": v
            } for t, v in enumerate(values))

    return pd.DataFrame.from_records(records)

def build_last_events_features(df, window_feats, N=1, target_activity="Pump adjustment"):
    """Construct features from N previous events and join with sensor features."""
    for n in range(1, N + 1):
        df[f"prev_event_{n}"] = df.groupby("case_id")["activity"].shift(n)

    prev_feats = pd.get_dummies(
        df[[f"prev_event_{n}" for n in range(1, N + 1)]],
        prefix=[f"prev_event_{n}" for n in range(1, N + 1)],
        drop_first=True
    )

    resource_enc = pd.get_dummies(df[["resource"]], drop_first=True)
    df[f"activity_{target_activity}"] = (df["activity"] == target_activity).astype(int)
    target = df.groupby("case_id")[f"activity_{target_activity}"].shift(-1)

    features = window_feats.join(resource_enc, how="left").join(prev_feats, how="left")
    features["target"] = target
    features["adjustment_type"] = df["adjustment_type"]
    features = features.dropna(subset=["target"])
    features["target"] = features["target"].astype(int)
    return features



def visualize_tree(model, feature_names, class_names, filename="decision_tree"):
    dot = tree.export_graphviz(
        model,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True,
        proportion=False,
        precision=2
    )
    graphviz.Source(dot).render(filename, format="png", cleanup=True)
    print(f"Decision tree saved as {filename}.png")

# ------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------
def main():
    freeze_support()
    df = pd.read_csv("data/event_with_sensor_lists.csv", parse_dates=["event_timestamp", "start_time", "end_time"])
    df = df.sort_values(by=["case_id", "event_timestamp"]).reset_index(drop=True)
    df["next_event"] = df.groupby("case_id")["activity"].shift(-1)
    df["pump_adjustment_next"] = (df["next_event"] == "Pump adjustment").astype(int)
    #counts = count_predecessor_events(df)
    #print(verify_consecutive_pump_adjustments(df))
    #print(counts)
    #df = df[df["adjustment_type"] == "first_adjustment"]
    df = enrich_event_log_with_context(df)

    long_df = build_long_format_sensor_data(df)

    print("Built long-format sensor data")

    feats = extract_window_features_fast(df, long_df)
    feats = engineer_domain_features(feats)

    feats = build_last_events_features(df, feats, N=3, target_activity="Pump adjustment")

    feats.to_csv("data/feats_from_analysis.csv", index=False)

    print("Feature set constructed")
    # --- Feature pruning step ---
    #first = feat_df[feat_df["adjustment_type"] == "first_adjustment"]
    #repeat = feat_df[feat_df["adjustment_type"] == "repeat_adjustment"]
    #X, y = feats.drop(columns="target"), feats["target"]
    X = pd.get_dummies(feats.drop(columns=["target"]), columns=["adjustment_type"], drop_first=True)
    y = feats["target"]

    # Prune redundant features
    #X = prune_features(X,y,importance_threshold=0.0009,corr_threshold=0.90,variance_threshold=0.0)
    #X, y = feats.drop(columns="target"), feats["target"]
    #learn_tree(df=X, result_column=y, names=X.columns, result="Pump adjustment", final=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)

    depths = range(3, 15)
    scores = [
        cross_val_score(
            DecisionTreeClassifier(max_depth=d, random_state=42, class_weight="balanced"),
            X_train, y_train, cv=5, scoring="precision"
        ).mean()
        for d in depths
    ]
    best_depth = depths[int(np.argmax(scores))]
    print(f"Best max_depth: {best_depth}")



    clf = DecisionTreeClassifier(max_depth=best_depth, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print(export_text(clf, feature_names=list(X.columns)))

    visualize_tree(clf, X.columns, ["No Adjustment", "Pump Adjustment"], filename="tree_output")

    importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 20 Features:\n", importances.head(20))



if __name__ == "__main__":
    main()