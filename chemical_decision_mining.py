#!/usr/bin/env python3

import itertools
import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import classification_report
from sklearn import tree
from scipy.stats import ks_2samp
from joblib import Parallel, delayed
from multiprocessing import freeze_support
import graphviz


def calc_slope(x):
    if len(x) < 2:
        return np.nan
    return np.polyfit(np.arange(len(x)), x, 1)[0]

def calc_autocorr(x, lag=1):
    if len(x) <= lag:
        return np.nan
    x = x - x.mean()
    denom = np.dot(x, x)
    return np.nan if denom == 0 else np.dot(x[:-lag], x[lag:]) / denom

def safe_corr(x, y):
    if len(x) != len(y) or len(x) < 2:
        return np.nan
    sx, sy = x.std(ddof=0), y.std(ddof=0)
    if sx == 0 or sy == 0:
        return np.nan
    return np.corrcoef(x, y)[0, 1]

def count_predecessor_events(df, case_col="case_id", activity_col="activity"):
    df = df.sort_values([case_col, "event_timestamp"]).copy()
    df["prev_activity"] = df.groupby(case_col)[activity_col].shift(1)
    mask = df[activity_col] == "Pump adjustment"
    preceding = df.loc[mask, "prev_activity"]
    return preceding.value_counts().sort_values(ascending=False)

def prune_features(X, y=None, importance_threshold=0.0, corr_threshold=0.9, variance_threshold=0.0):
    Xp = X.copy()

    if variance_threshold > 0:
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(Xp.fillna(0))
        Xp = Xp.loc[:, selector.get_support()]
        print(f"variance pruning: kept {Xp.shape[1]} features")

    corr = Xp.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > corr_threshold)]
    if to_drop:
        print(f"correlation pruning: dropping {len(to_drop)} correlated features")
        Xp = Xp.drop(columns=to_drop)

    if y is not None and importance_threshold > 0:
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=6, n_jobs=-1, random_state=42, class_weight="balanced"
        )
        rf.fit(Xp.fillna(0), y)
        imp = pd.Series(rf.feature_importances_, index=Xp.columns)
        keep = imp[imp > importance_threshold].index
        print(f"importance pruning: removed {len(imp) - len(keep)} low-importance features")
        Xp = Xp[keep]

    print(f"final feature count: {Xp.shape[1]}")
    return Xp

def add_sensor_interactions(df, sensor_keywords=None, max_pairs=30, max_triples=5,
                            include_types=("product", "difference", "ratio", "sum", "norm_diff"), eps=1e-6):
    df = df.copy()
    if sensor_keywords is None:
        sensor_keywords = ["filter 1 delta", "filter 1 inlet", "filter 2 delta", "pump circulation", "tank pressure"]

    sensor_cols = [c for c in df.columns if any(k in c.lower() for k in sensor_keywords)]
    if len(sensor_cols) < 2:
        print("not enough sensor columns for interactions")
        return df

    top = df[sensor_cols].var().sort_values(ascending=False).head(int(max_pairs * 1.2)).index
    sensor_cols = list(top)

    pairs = list(itertools.combinations(sensor_cols, 2))[:max_pairs]
    print(f"found {len(pairs)} sensor pairs")

    for a, b in pairs:
        if "product" in include_types:
            df[f"{a}_x_{b}"] = df[a] * df[b]
        if "difference" in include_types:
            df[f"{a}_minus_{b}"] = df[a] - df[b]
        if "ratio" in include_types:
            df[f"{a}_div_{b}"] = df[a] / (df[b].replace(0, np.nan) + eps)
        if "sum" in include_types:
            df[f"{a}_plus_{b}"] = df[a] + df[b]
        if "norm_diff" in include_types:
            df[f"{a}_normdiff_{b}"] = (df[a] - df[b]) / (df[a] + df[b] + eps)

    triples = list(itertools.combinations(sensor_cols, 3))[:max_triples]
    for a, b, c in triples:
        df[f"{a}_x_{b}_x_{c}"] = df[a] * df[b] * df[c]

    inter_cols = [c for c in df.columns if any(k in c for k in ["_x_", "_minus_", "_plus_", "_div_", "_normdiff_"])]
    df["interaction_mean"] = df[inter_cols].mean(axis=1)
    df["interaction_std"] = df[inter_cols].std(axis=1)
    df["interaction_abs_mean"] = df[inter_cols].abs().mean(axis=1)

    print(f"added {len(inter_cols)} interaction features")
    return df

def add_runtime_and_recent_adjustment_features(df_events, lookback_events=2):
    df = df_events.copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], errors="coerce", utc=True)
    df = df.dropna(subset=["event_timestamp"])
    df = df.sort_values(["case_id", "event_timestamp"]).reset_index(drop=True)

    df["first_event_time"] = df.groupby("case_id")["event_timestamp"].transform("min")
    df["runtime_min"] = (df["event_timestamp"] - df["first_event_time"]).dt.total_seconds() / 60.0

    def recent_flag(g):
        is_adj = (g["activity"] == "Pump adjustment").astype(int)
        return (is_adj.rolling(lookback_events + 1, min_periods=1).sum().shift(1) > 0).astype(int)

    df["recent_pump_adjustment_flag"] = df.groupby("case_id", group_keys=False).apply(recent_flag)
    return df.drop(columns=["first_event_time"])

def build_last_events_features(df, window_feats, N=2, target_activity="Pump adjustment"):
    df_sorted = df.sort_values(["case_id", "event_timestamp"]).copy()

    for col in ["runtime_min", "recent_pump_adjustment_flag"]:
        if col not in df_sorted:
            raise ValueError(f"Missing feature {col}")

    current_activity = pd.get_dummies(df_sorted[["activity"]], prefix="activity", drop_first=True)

    for n in range(1, N + 1):
        df_sorted[f"prev_event_{n}"] = df_sorted.groupby("case_id")["activity"].shift(n)
    prev_feats = pd.get_dummies(
        df_sorted[[f"prev_event_{n}" for n in range(1, N + 1)]],
        prefix=[f"prev_event_{n}" for n in range(1, N + 1)],
        drop_first=True,
    )

    resource_enc = pd.get_dummies(df_sorted[["resource"]], prefix="resource", drop_first=True)

    df_sorted[f"activity_{target_activity}"] = (df_sorted["activity"] == target_activity).astype(int)
    target = df_sorted.groupby("case_id")[f"activity_{target_activity}"].shift(-1)

    numeric_cols = ["runtime_min", "recent_pump_adjustment_flag"]
    for col in numeric_cols:
        if col in window_feats.columns:
            window_feats = window_feats.drop(columns=[col])

    feats = (
        window_feats
        .join(df_sorted[numeric_cols], how="left")
        .join(resource_enc, how="left")
        .join(current_activity, how="left")
        .join(prev_feats, how="left")
    )

    feats["target"] = target
    feats = feats.dropna(subset=["target"])
    feats["target"] = feats["target"].astype(int)
    return feats

def engineer_domain_features(X: pd.DataFrame) -> pd.DataFrame:
    Xn = X.copy()
    if "Tank_Pressure__slope_win10m" in X:
        Xn["feat_tank_pressure_rise"] = (X["Tank_Pressure__slope_win10m"] > 0.05).astype(int)
    if "Filter_1_DeltaP__max_jump_win60m" in X:
        Xn["feat_filter1_dp_spike"] = (X["Filter_1_DeltaP__max_jump_win60m"] > 0.2).astype(int)
    if {"Filter_1_Inlet_Pressure__mean_win60m", "Filter_2_DeltaP__mean_win60m"}.issubset(X.columns):
        ratio = X["Filter_1_Inlet_Pressure__mean_win60m"] / (X["Filter_2_DeltaP__mean_win60m"] + 1e-6)
        Xn["feat_inlet_outlet_imbalance"] = (ratio > 3).astype(int)
    if "Tank_Pressure__cv_win60m" in X:
        Xn["feat_tank_pressure_unstable"] = (X["Tank_Pressure__cv_win60m"] > 0.05).astype(int)
    corr_col = "corr_Pump_Circulation_Flow__Tank_Pressure_win60m"
    if corr_col in X:
        Xn["feat_flow_pressure_uncoupled"] = (X[corr_col] < 0.2).astype(int)
    return Xn

def compute_window_features(df_events, df_signals, windows=(0.5,2,5, 15,30,45, 60), n_jobs=-1):
    all_sensors = df_signals["sensor"].unique()

    def process_event(ev):
        etime = ev["event_timestamp"]
        feat = {"event_id": ev.name}
        for w in windows:
            start = etime - pd.Timedelta(minutes=w)
            end = etime
            win = df_signals[(df_signals["time_index"] >= start) &
                             (df_signals["time_index"] < end)]
            if win.empty:
                continue

            sensor_groups = {s: g["value"].to_numpy() for s, g in win.groupby("sensor")}
            has_flow = any("Flow" in s for s in sensor_groups.keys())

            for sensor, vals in sensor_groups.items():
                if vals.size == 0:
                    continue
                name = sensor.replace(" ", "_")
                mean, std = vals.mean(), vals.std(ddof=0)
                minv, maxv = vals.min(), vals.max()
                start_val, end_val = vals[0], vals[-1]
                feat.update({
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
                    feat[f"{name}__max_jump_win{w}m"] = diffs.max()
                    feat[f"{name}__num_jumps_gt1p_win{w}m"] = (diffs > 0.01 * abs(mean)).sum()
                else:
                    feat[f"{name}__max_jump_win{w}m"] = 0.0
                    feat[f"{name}__num_jumps_gt1p_win{w}m"] = 0

                if has_flow and "Flow" in sensor_groups:
                    flow_vals = next((v for s, v in sensor_groups.items() if "Flow" in s), None)
                    if flow_vals is not None and flow_vals.mean() != 0:
                        feat[f"{name}__mean_normByFlow_win{w}m"] = mean / flow_vals.mean()

            # pairwise correlations
            sensors = list(sensor_groups.keys())
            for s1, s2 in itertools.combinations(sensors, 2):
                v1, v2 = sensor_groups[s1], sensor_groups[s2]
                if len(v1) > 1 and len(v1) == len(v2):
                    feat[f"corr_{s1.replace(' ','_')}__{s2.replace(' ','_')}_win{w}m"] = safe_corr(v1, v2)
        return feat

    print(f"⏱️ Computing window features for {len(df_events)} events ...")
    feat_list = Parallel(n_jobs=n_jobs, backend="loky")(delayed(process_event)(ev) for _, ev in df_events.iterrows())
    feat_list = [f for f in feat_list if f]
    feat_df = pd.DataFrame(feat_list).set_index("event_id")

    for sensor in all_sensors:
        nm = sensor.replace(" ", "_")
        if f"{nm}__slope_win10m" in feat_df and f"{nm}__slope_win60m" in feat_df:
            feat_df[f"{nm}__slope_accel"] = feat_df[f"{nm}__slope_win10m"] - feat_df[f"{nm}__slope_win60m"]
        if f"{nm}__mean_win10m" in feat_df and f"{nm}__mean_win60m" in feat_df:
            feat_df[f"{nm}__mean_diff10_60"] = feat_df[f"{nm}__mean_win10m"] - feat_df[f"{nm}__mean_win60m"]
        if f"{nm}__std_win10m" in feat_df and f"{nm}__std_win60m" in feat_df:
            feat_df[f"{nm}__std_ratio10_60"] = feat_df[f"{nm}__std_win10m"] / (feat_df[f"{nm}__std_win60m"] + 1e-6)
        if f"{nm}__cv_win10m" in feat_df and f"{nm}__cv_win60m" in feat_df:
            feat_df[f"{nm}__cv_ratio10_60"] = feat_df[f"{nm}__cv_win10m"] / (feat_df[f"{nm}__cv_win60m"] + 1e-6)

    nunique = feat_df.nunique(dropna=True)
    feat_df = feat_df.drop(columns=nunique[nunique <= 1].index, errors="ignore")
    print(f"✅ Generated {feat_df.shape[1]} feature columns")
    return pd.concat([df_events.reset_index(drop=True), feat_df.reset_index(drop=True)], axis=1)

def generate_features(df_events, df_signals):
    #df_wide = (df_signals
    #           .pivot_table(index="time_index", columns="sensor", values="value", aggfunc="mean")
    #           .reset_index())
    #df_wide = add_sensor_interactions(df_wide, max_pairs=50)
    #df_signals = (df_wide
    #              .melt(id_vars=["time_index"], var_name="sensor", value_name="value")
    #              .sort_values(["time_index", "sensor"]).reset_index(drop=True))
    #print("added sensor correlations")

    df_events = add_runtime_and_recent_adjustment_features(df_events, lookback_events=15)
    print("added runtime and pump adjustment features")

    window_feats = compute_window_features(df_events, df_signals)
    window_feats = engineer_domain_features(window_feats)

    full_feats = build_last_events_features(df_events, window_feats, N=5)
    drop_cols = ["duration_min", "case_id", "activity", "resource", "event_timestamp",
                 "start_time", "end_time", "source"]
    full_feats = full_feats.drop(columns=drop_cols, errors="ignore")
    full_feats.to_csv("data/feats_from_analysis_without_drifts_and_interactions.csv", index=False)
    return full_feats

def visualize_tree(model, feature_names, filename="tree_output"):
    dot = tree.export_graphviz(
        model,
        feature_names=feature_names,
        class_names=["no adjustment", "pump adjustment next"],
        filled=True,
        rounded=True,
        precision=2
    )
    graphviz.Source(dot).render(filename, format="png", cleanup=True)
    print(f"saved decision tree: {filename}.png")

def main():
    freeze_support()
    df_events = pd.read_csv("data/process_events.csv", parse_dates=["start_time", "end_time", "event_timestamp"])
    df_signals = pd.read_csv("data/sensor_long.csv", parse_dates=["timestamp"]).rename(columns={"timestamp": "time_index"})

    print(f"loaded {len(df_events)} events and {len(df_signals)} sensor rows")
    full_feats = generate_features(df_events, df_signals)
    X = full_feats.drop(columns="target", errors="ignore")
    y = full_feats["target"]
    X = prune_features(X, y)

    print(f"feature matrix: {X.shape[0]} x {X.shape[1]}  positives={y.sum()}")
    print(count_predecessor_events(df_events))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
    depths = range(1, 6)
    scores = [cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=42, class_weight="balanced"),
                              X_train, y_train, cv=3, scoring="f1").mean() for d in depths]
    best_depth = depths[int(np.argmax(scores))]
    print(f"best max_depth: {best_depth}")

    model = DecisionTreeClassifier(max_depth=best_depth, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nclassification report:\n", classification_report(y_test, y_pred))
    print(export_text(model, feature_names=list(X.columns)))
    visualize_tree(model, X.columns)

    imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\ntop 20 features:\n", imp.head(20))
    full_feats.to_csv("data/enriched_with_history.csv", index=False)
    print("saved data/enriched_with_history.csv")

def main_split_by_activity():
    full_feats = pd.read_csv("data/feats_from_analysis_without_drifts_and_interactions.csv")

    X = full_feats.drop(columns="target", errors="ignore")
    y = full_feats["target"]
    print(f"Feature matrix: {X.shape[0]} × {X.shape[1]}  (positives={y.sum()})")

    # X = prune_features(X, y, importance_threshold=0.0009, corr_threshold=0.9)

    # === GLOBAL MODEL ==========================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )

    depths = range(1, 6)
    scores = [
        cross_val_score(
            DecisionTreeClassifier(max_depth=d, random_state=42, class_weight="balanced"),
            X_train, y_train, cv=5, scoring="accuracy"
        ).mean()
        for d in depths
    ]
    best_depth = depths[int(np.argmax(scores))]
    print(f"Best global max_depth: {best_depth}")

    model = DecisionTreeClassifier(max_depth=best_depth,
                                   random_state=42,
                                   class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\nGlobal Classification Report:\n", classification_report(y_test, y_pred))
    visualize_tree(model, X.columns)

    importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 20 Global Features:\n", importances.head(20))

    # === ACTIVITY-SPECIFIC ANALYSIS ===========================================
    print("\n🔍 Evaluating per-activity model performance ...")

    results = []
    activity_cols = [c for c in X.columns if c.startswith("activity_")]

    for activity_col in activity_cols:
        # Subset data where this current-activity onehot == 1
        subset = X[X[activity_col] == 1]
        y_sub = y.loc[subset.index]

        if y_sub.nunique() < 2 or len(subset) < 20:
            continue
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                subset, y_sub, test_size=0.25, stratify=y_sub, random_state=42
            )

            depths = range(1, 6)
            scores = [
            cross_val_score(
                DecisionTreeClassifier(max_depth=d, random_state=42, class_weight="balanced"),
                X_train, y_train, cv=5, scoring="accuracy"
            ).mean()
            for d in depths
            ]
            local_depth = depths[int(np.argmax(scores))]

            model = DecisionTreeClassifier(max_depth=local_depth,
                                       random_state=42,
                                       class_weight="balanced")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            rep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            acc = rep["accuracy"]
            rec = rep["1"]["recall"]

            f1 = rep["1"]["f1-score"]
            results.append({
            "activity": activity_col.replace("activity_",""),
            "n_samples": len(subset),
            "best_depth": local_depth,
            "accuracy": acc,
            "f1_positive": f1,
            "recall_positive": rec
            })
            print(f"🔸 {activity_col}: acc={acc:.3f}, f1={f1:.3f}, rec={rec}, best_depth={local_depth}, n={len(subset)}")
        except Exception as e: continue

    res_df = pd.DataFrame(results).sort_values("recall_positive", ascending=False)
    best = res_df.iloc[0]
    print("\n🏁 Best-performing activity:")
    print(best)
    res_df.to_csv("data/activity_tree_performance.csv", index=False)
    print("💾 Saved → data/activity_tree_performance.csv")

    full_feats.to_csv("data/enriched_with_history.csv", index=False)
    print("💾 Saved → data/enriched_with_history.csv")


if __name__ == "__main__":
    main()
    #main_split_by_activity()