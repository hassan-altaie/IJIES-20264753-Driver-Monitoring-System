# distraction_event_evaluation.py
# Event-Based Multi-Label Distraction Evaluation
# Classes: PhoneUse, HeadLeft, HeadRight, PassengerTalk, CenterConsole
# Folders:
#   distraction_csv/
#   distraction_gt/
# Run:
#   python distraction_event_evaluation.py

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV_DIR = Path("distraction_csv")
GT_DIR = Path("distraction_gt")
OUTPUT_DIR = Path("distraction_results")
OUTPUT_DIR.mkdir(exist_ok=True)

TIME_CANDIDATES = ["ADJUSTED TIME", "Adjusted time", "AdjustedTime", "ElapsedSeconds", "Elapsed Seconds"]
SPEED_CANDIDATES = ["SpeedDisplayKmh", "VehicleSpeed", "Speed", "Vehicle Speed"]
ALERTS_CANDIDATES = ["AlertsEnabled", "Alerts Enabled"]

CLASS_COLUMNS = {
    "PhoneUse": "ConfirmedPhoneUse",
    "HeadLeft": "ConfirmedHeadLeft",
    "HeadRight": "ConfirmedHeadRight",
    "PassengerTalk": "ConfirmedPassengerTalk",
    "CenterConsole": "ConfirmedCenterConsole",
}

SAFETY_COLUMNS = {
    "MedicalEmergencyConfirmed": "MedicalEmergencyConfirmed",
    "HazardRequest": "HazardRequest",
    "BrakeRequest": "BrakeRequest",
}

GT_START_CANDIDATES = ["Start_Time", "Start_Time_sec", "Start", "Start Time", "Start_sec"]
GT_END_CANDIDATES = ["End_Time", "End_Time_sec", "End", "End Time", "End_sec"]
GT_EVENT_CANDIDATES = ["Event_ID", "EventID", "Event", "ID"]
GT_LABEL_CANDIDATES = ["GroundTruth", "Ground_Truth", "Target", "Target_Decision", "Label"]

MATCH_TOLERANCE_SECONDS = 0.75
MIN_VALID_EVENT_DURATION_SEC = 1.0
USE_ALERTS_ENABLED_FILTER = False
USE_SPEED_FILTER = False
SPEED_THRESHOLD = 10.0


def normalize_name(x):
    return str(x).strip().lower().replace("_", "").replace(" ", "")


def normalize_label(x):
    return str(x).strip().replace(" ", "").replace("_", "").upper()


LABEL_MAP = {normalize_label(x): x for x in CLASS_COLUMNS}
LABEL_MAP["IGNORE"] = "IGNORE"


def find_col(df, candidates, required=True):
    normalized = {normalize_name(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_name(cand)
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(f"Column not found. Needed one of {candidates}. Existing columns: {list(df.columns)}")
    return None


def participant_id_from_name(filename):
    m = re.search(r"p\s*0?(\d+)", filename.lower())
    if not m:
        return Path(filename).stem
    return f"P{int(m.group(1)):02d}"


def to_binary(series):
    s = series.copy()
    if s.dtype == object:
        s = s.astype(str).str.strip().str.lower()
        return s.isin(["1", "true", "yes", "y", "on", "active"]).astype(int)
    return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).gt(0).astype(int)


def load_csv(path):
    df = pd.read_csv(path)
    time_col = find_col(df, TIME_CANDIDATES)
    speed_col = find_col(df, SPEED_CANDIDATES, required=False)
    alerts_col = find_col(df, ALERTS_CANDIDATES, required=False)

    out = pd.DataFrame()
    out["time"] = pd.to_numeric(df[time_col], errors="coerce")
    out["speed"] = pd.to_numeric(df[speed_col], errors="coerce") if speed_col else pd.NA
    out["alerts_enabled"] = to_binary(df[alerts_col]) if alerts_col else 1

    for label, col in CLASS_COLUMNS.items():
        out[label] = to_binary(df[col]) if col in df.columns else 0

    for label, col in SAFETY_COLUMNS.items():
        out[label] = to_binary(df[col]) if col in df.columns else 0

    out = out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)

    if USE_ALERTS_ENABLED_FILTER:
        out = out[out["alerts_enabled"] == 1].copy()
    if USE_SPEED_FILTER and out["speed"].notna().any():
        out = out[out["speed"] >= SPEED_THRESHOLD].copy()

    return out.reset_index(drop=True)


def load_gt(path):
    gt = pd.read_excel(path)
    start_col = find_col(gt, GT_START_CANDIDATES)
    end_col = find_col(gt, GT_END_CANDIDATES)
    event_col = find_col(gt, GT_EVENT_CANDIDATES, required=False)
    label_col = find_col(gt, GT_LABEL_CANDIDATES)

    out = pd.DataFrame()
    out["start"] = pd.to_numeric(gt[start_col], errors="coerce")
    out["end"] = pd.to_numeric(gt[end_col], errors="coerce")
    out["event_id"] = gt[event_col].astype(str) if event_col else [f"D{i+1}" for i in range(len(gt))]
    out["raw_label"] = gt[label_col].astype(str)
    out["label"] = out["raw_label"].apply(lambda x: LABEL_MAP.get(normalize_label(x), "UNKNOWN"))

    out = out.dropna(subset=["start", "end"])
    out = out[out["end"] >= out["start"]]
    out = out[out["label"].isin(list(CLASS_COLUMNS.keys()) + ["IGNORE"])]
    return out.sort_values("start").reset_index(drop=True)


def binary_intervals(df, col, event_type):
    s = df[col].astype(int).values
    times = df["time"].values
    rows, active, start_t, idx, prev = [], False, None, 1, 0

    for i, val in enumerate(s):
        if prev == 0 and val == 1:
            active, start_t = True, float(times[i])
        if prev == 1 and val == 0 and active:
            end_t = float(times[i - 1])
            rows.append({"Pred_ID": f"{event_type}_P{idx}", "Class": event_type,
                         "Pred_Start": start_t, "Pred_End": end_t,
                         "Duration_s": max(0.0, end_t - start_t)})
            idx += 1
            active = False
        prev = val

    if active:
        end_t = float(times[-1])
        rows.append({"Pred_ID": f"{event_type}_P{idx}", "Class": event_type,
                     "Pred_Start": start_t, "Pred_End": end_t,
                     "Duration_s": max(0.0, end_t - start_t)})
    return pd.DataFrame(rows)


def build_predicted_events(df):
    pred = pd.concat([binary_intervals(df, label, label) for label in CLASS_COLUMNS], ignore_index=True)
    if pred.empty:
        return pred, pd.DataFrame()
    valid = pred[pred["Duration_s"] >= MIN_VALID_EVENT_DURATION_SEC].copy()
    rejected = pred[pred["Duration_s"] < MIN_VALID_EVENT_DURATION_SEC].copy()
    if not rejected.empty:
        rejected["Reject_Reason"] = "short_event_below_minimum_duration"
    return valid.reset_index(drop=True), rejected.reset_index(drop=True)


def intervals_overlap(a_start, a_end, b_start, b_end):
    return (a_start <= b_end) and (a_end >= b_start)


def overlap_seconds(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def match_events(gt, pred_events, pid):
    gt_events = gt[gt["label"].isin(CLASS_COLUMNS.keys())].copy()
    pred = pred_events.copy()
    if pred.empty:
        pred["Matched"] = []
    else:
        pred["Matched"] = False

    rows = []
    for _, ev in gt_events.iterrows():
        label = ev["label"]
        gs = float(ev["start"]) - MATCH_TOLERANCE_SECONDS
        ge = float(ev["end"]) + MATCH_TOLERANCE_SECONDS

        candidates = pd.DataFrame() if pred.empty else pred[
            (pred["Class"] == label) & (~pred["Matched"]) &
            (pred["Pred_Start"] <= ge) & (pred["Pred_End"] >= gs)
        ].copy()

        if candidates.empty:
            rows.append({"Participant": pid, "Event_ID": ev["event_id"], "GroundTruth": label,
                         "GT_Start": float(ev["start"]), "GT_End": float(ev["end"]),
                         "Detected": 0, "Pred_ID": "", "Pred_Start": np.nan, "Pred_End": np.nan,
                         "Detection_Delay_s": np.nan, "Result": "FN"})
        else:
            # Earliest valid match is used to calculate the first confirmed detection delay.
            # This is more appropriate for real-time event-based evaluation than selecting
            # the segment with maximum temporal overlap.
            candidates["Overlap_s"] = candidates.apply(
                lambda r: overlap_seconds(gs, ge, float(r["Pred_Start"]), float(r["Pred_End"])), axis=1)
            candidates = candidates[candidates["Overlap_s"] > 0].copy()
            if candidates.empty:
                rows.append({"Participant": pid, "Event_ID": ev["event_id"], "GroundTruth": label,
                             "GT_Start": float(ev["start"]), "GT_End": float(ev["end"]),
                             "Detected": 0, "Pred_ID": "", "Pred_Start": np.nan, "Pred_End": np.nan,
                             "Detection_Delay_s": np.nan, "Result": "FN"})
                continue
            candidates["Distance_to_GT_Start"] = (candidates["Pred_Start"] - float(ev["start"])).abs()
            best_idx = candidates.sort_values(["Pred_Start", "Distance_to_GT_Start"], ascending=[True, True]).index[0]
            pred.loc[best_idx, "Matched"] = True
            p = pred.loc[best_idx]
            delay = max(0.0, float(p["Pred_Start"]) - float(ev["start"]))
            rows.append({"Participant": pid, "Event_ID": ev["event_id"], "GroundTruth": label,
                         "GT_Start": float(ev["start"]), "GT_End": float(ev["end"]),
                         "Detected": 1, "Pred_ID": p["Pred_ID"], "Pred_Start": float(p["Pred_Start"]),
                         "Pred_End": float(p["Pred_End"]), "Detection_Delay_s": delay, "Result": "TP"})

    event_results = pd.DataFrame(rows)

    fp_rows = []
    if not pred.empty:
        ignore = gt[gt["label"] == "IGNORE"]
        for _, p in pred[~pred["Matched"]].iterrows():
            inside_ignore = any(intervals_overlap(float(p["Pred_Start"]), float(p["Pred_End"]),
                                                  float(ig["start"]), float(ig["end"]))
                                for _, ig in ignore.iterrows())
            if inside_ignore:
                continue
            fp_rows.append({"Participant": pid, "Class": p["Class"], "Pred_ID": p["Pred_ID"],
                            "FP_Start": float(p["Pred_Start"]), "FP_End": float(p["Pred_End"]),
                            "Duration_s": float(p["Duration_s"]), "Result": "FP"})
    return event_results, pd.DataFrame(fp_rows), pred


def evaluate_safety_outputs(df, gt, pid):
    ignore = gt[gt["label"] == "IGNORE"]
    rows = []
    for col in SAFETY_COLUMNS:
        intervals = binary_intervals(df, col, col)
        for _, ev in intervals.iterrows():
            inside_ignore = any(intervals_overlap(float(ev["Pred_Start"]), float(ev["Pred_End"]),
                                                  float(ig["start"]), float(ig["end"]))
                                for _, ig in ignore.iterrows())
            if not inside_ignore:
                rows.append({"Participant": pid, "Type": col, "Start": float(ev["Pred_Start"]),
                             "End": float(ev["Pred_End"]), "Duration_s": float(ev["Duration_s"])})
    return pd.DataFrame(rows)


def safe_div(a, b):
    return float(a / b) if b else 0.0


def metrics_by_class(event_results, fp_df):
    rows = []
    for label in CLASS_COLUMNS:
        subset = event_results[event_results["GroundTruth"] == label]
        tp = int((subset["Detected"] == 1).sum())
        fn = int((subset["Detected"] == 0).sum())
        fp = int((fp_df["Class"] == label).sum()) if not fp_df.empty else 0
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        rows.append({"Class": label, "GT_Events": len(subset), "TP": tp, "FN": fn, "FP": fp,
                     "Precision_%": precision * 100, "Recall_Detection_Rate_%": recall * 100,
                     "F1_score_%": f1 * 100,
                     "Average_Detection_Delay_s": subset.loc[subset["Detected"] == 1, "Detection_Delay_s"].mean()})
    return pd.DataFrame(rows)


def participant_summary(event_results, fp_df):
    rows = []
    for pid, group in event_results.groupby("Participant"):
        fp_p = fp_df[fp_df["Participant"] == pid] if not fp_df.empty else pd.DataFrame()
        tp = int((group["Detected"] == 1).sum())
        fn = int((group["Detected"] == 0).sum())
        fp = int(len(fp_p))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        acc = safe_div(tp, tp + fp + fn)
        rows.append({"Participant": pid, "GT_Events": len(group), "TP": tp, "FN": fn, "FP": fp,
                     "Event_Accuracy_no_TN_%": acc * 100, "Precision_%": precision * 100,
                     "Recall_Detection_Rate_%": recall * 100, "F1_score_%": f1 * 100})
    return pd.DataFrame(rows)


def make_event_matrix(metrics_df, out_path):
    matrix = metrics_df[["TP", "FN", "FP"]].to_numpy()
    labels = metrics_df["Class"].tolist()
    plt.figure(figsize=(8, 5))
    plt.imshow(matrix)
    plt.title("Distraction Event-Based Detection Matrix")
    plt.xticks([0, 1, 2], ["TP", "FN", "FP"])
    plt.yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_bar(x, y, title, ylabel, out_path, ylim=None):
    plt.figure(figsize=(9, 5))
    plt.bar(x, y)
    if ylim:
        plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    csv_files = sorted(CSV_DIR.glob("*.csv"))
    gt_files = sorted(list(GT_DIR.glob("*.xlsx")) + list(GT_DIR.glob("*.xls")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {CSV_DIR}")
    if not gt_files:
        raise FileNotFoundError(f"No GT files found in {GT_DIR}")

    gt_by_pid = {participant_id_from_name(p.name): p for p in gt_files}
    all_events, all_fp, all_pred, all_rejected, all_safety, all_gt = [], [], [], [], [], []

    for csv_path in csv_files:
        pid = participant_id_from_name(csv_path.name)
        gt_path = gt_by_pid.get(pid)
        if gt_path is None:
            print(f"Warning: no GT for {pid}; skipped {csv_path.name}")
            continue

        print(f"Processing {pid}: {csv_path.name} + {gt_path.name}")
        df = load_csv(csv_path)
        gt = load_gt(gt_path)
        pred_valid, pred_rejected = build_predicted_events(df)
        events, fp, pred_matched = match_events(gt, pred_valid, pid)
        safety = evaluate_safety_outputs(df, gt, pid)

        if not events.empty: all_events.append(events)
        if not fp.empty: all_fp.append(fp)
        if not pred_matched.empty:
            pred_matched["Participant"] = pid
            all_pred.append(pred_matched)
        if not pred_rejected.empty:
            pred_rejected["Participant"] = pid
            all_rejected.append(pred_rejected)
        if not safety.empty: all_safety.append(safety)
        gt["Participant"] = pid
        all_gt.append(gt)

    if not all_events:
        raise RuntimeError("No matching files processed. Make sure names contain P01/P02...")

    event_results = pd.concat(all_events, ignore_index=True)
    fp_df = pd.concat(all_fp, ignore_index=True) if all_fp else pd.DataFrame(columns=["Participant", "Class"])
    pred_df = pd.concat(all_pred, ignore_index=True) if all_pred else pd.DataFrame()
    rejected_df = pd.concat(all_rejected, ignore_index=True) if all_rejected else pd.DataFrame()
    safety_df = pd.concat(all_safety, ignore_index=True) if all_safety else pd.DataFrame()
    gt_df = pd.concat(all_gt, ignore_index=True)

    class_metrics = metrics_by_class(event_results, fp_df)
    participant_metrics = participant_summary(event_results, fp_df)

    TP, FN, FP = int(class_metrics["TP"].sum()), int(class_metrics["FN"].sum()), int(class_metrics["FP"].sum())
    precision, recall = safe_div(TP, TP + FP), safe_div(TP, TP + FN)
    f1 = safe_div(2 * precision * recall, precision + recall)
    acc = safe_div(TP, TP + FN + FP)

    overall = pd.DataFrame([{"Participants": participant_metrics["Participant"].nunique(),
                             "Distraction_Events": int(class_metrics["GT_Events"].sum()),
                             "TP": TP, "FN": FN, "FP": FP,
                             "Event_Accuracy_no_TN_%": acc * 100,
                             "Precision_%": precision * 100,
                             "Recall_Detection_Rate_%": recall * 100,
                             "F1_score_%": f1 * 100,
                             "Average_Detection_Delay_s": event_results.loc[event_results["Detected"] == 1, "Detection_Delay_s"].mean(),
                             "Rejected_Short_Events": len(rejected_df),
                             "Safety_False_Activations": len(safety_df)}])

    delay_stats = event_results.dropna(subset=["Detection_Delay_s"]).groupby("GroundTruth")["Detection_Delay_s"].agg(
        Count="count", Mean_s="mean", Median_s="median", Std_s="std", Min_s="min", Max_s="max"
    ).reset_index()

    out_xlsx = OUTPUT_DIR / "distraction_event_evaluation_results.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        overall.to_excel(writer, sheet_name="Overall", index=False)
        class_metrics.to_excel(writer, sheet_name="Summary_By_Class", index=False)
        participant_metrics.to_excel(writer, sheet_name="Participant_Summary", index=False)
        event_results.to_excel(writer, sheet_name="Event_Results", index=False)
        fp_df.to_excel(writer, sheet_name="False_Positives", index=False)
        pred_df.to_excel(writer, sheet_name="Predicted_Events", index=False)
        rejected_df.to_excel(writer, sheet_name="Rejected_Short_Events", index=False)
        safety_df.to_excel(writer, sheet_name="Safety_False_Activations", index=False)
        delay_stats.to_excel(writer, sheet_name="Delay_Statistics", index=False)
        gt_df.to_excel(writer, sheet_name="Ground_Truth_Loaded", index=False)

    make_event_matrix(class_metrics, OUTPUT_DIR / "distraction_event_detection_matrix.png")
    make_bar(class_metrics["Class"], class_metrics["Recall_Detection_Rate_%"], "Distraction Detection Rate by Class", "Detection Rate (%)", OUTPUT_DIR / "detection_rate_by_class.png", (0, 105))
    make_bar(class_metrics["Class"], class_metrics["Precision_%"], "Distraction Precision by Class", "Precision (%)", OUTPUT_DIR / "precision_by_class.png", (0, 105))
    make_bar(class_metrics["Class"], class_metrics["F1_score_%"], "Distraction F1-score by Class", "F1-score (%)", OUTPUT_DIR / "f1_by_class.png", (0, 105))
    make_bar(["Precision", "Recall", "F1-score"], [overall["Precision_%"].iloc[0], overall["Recall_Detection_Rate_%"].iloc[0], overall["F1_score_%"].iloc[0]], "Overall Distraction Event-Based Performance", "Percentage (%)", OUTPUT_DIR / "overall_metrics.png", (0, 105))
    make_bar(participant_metrics["Participant"], participant_metrics["F1_score_%"], "Distraction F1-score by Participant", "F1-score (%)", OUTPUT_DIR / "f1_by_participant.png", (0, 105))
    make_bar(class_metrics["Class"], class_metrics["GT_Events"], "Ground Truth Distraction Events by Class", "Number of Events", OUTPUT_DIR / "gt_events_by_class.png")

    delays = event_results.dropna(subset=["Detection_Delay_s"])
    if not delays.empty:
        plt.figure(figsize=(9, 5))
        for label in CLASS_COLUMNS.keys():
            d = delays[delays["GroundTruth"] == label]["Detection_Delay_s"].values
            if len(d) > 0:
                plt.plot(range(1, len(d) + 1), d, marker="o", label=label)
        plt.xlabel("Detected Event Index")
        plt.ylabel("Detection Delay (s)")
        plt.title("Distraction Detection Delay by Class")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "detection_delay_by_class.png", dpi=300)
        plt.close()

        plt.figure(figsize=(7, 5))
        plt.hist(delays["Detection_Delay_s"], bins=10)
        plt.xlabel("Detection Delay (s)")
        plt.ylabel("Number of Events")
        plt.title("Distraction Detection Delay Histogram")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "detection_delay_histogram.png", dpi=300)
        plt.close()

        data, labels = [], []
        for label in CLASS_COLUMNS.keys():
            vals = delays[delays["GroundTruth"] == label]["Detection_Delay_s"].dropna().values
            if len(vals) > 0:
                data.append(vals); labels.append(label)
        if data:
            plt.figure(figsize=(9, 5))
            plt.boxplot(data, labels=labels)
            plt.ylabel("Detection Delay (s)")
            plt.title("Distraction Detection Delay Boxplot")
            plt.xticks(rotation=25, ha="right")
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / "detection_delay_boxplot.png", dpi=300)
            plt.close()

    print("\n========== DISTRACTION OVERALL ==========")
    print(overall.to_string(index=False))
    print("\n========== SUMMARY BY CLASS ==========")
    print(class_metrics.to_string(index=False))
    print(f"\nSaved Excel: {out_xlsx}")
    print(f"Saved figures in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
