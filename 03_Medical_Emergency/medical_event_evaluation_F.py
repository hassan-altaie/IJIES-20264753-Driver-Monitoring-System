# medical_event_evaluation.py
# Event-Based Medical Emergency Evaluation
# يحسب: Precision, Recall/Detection Rate, F1, Event Accuracy, Detection Delay,
# Hazard Success, Brake Success

import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_DIR = Path("medical_csv")
GT_DIR = Path("medical_gt")
OUTPUT_DIR = Path("medical_results")
OUTPUT_DIR.mkdir(exist_ok=True)

TIME_CANDIDATES = ["Addjusted time", "Adjusted time", "Adjusted Time", "ADJUSTED TIME", "AdjustedTime", "ElapsedSeconds", "Elapsed Seconds"]

# Allow a small tolerance after the ground-truth event end so that
# the last frames of the same detected emergency are not counted as false positives.
GT_TOLERANCE_BEFORE_S = 0.5
GT_TOLERANCE_AFTER_S = 2.0
MEDICAL_COL = "MedicalEmergencyConfirmed"
HAZARD_COL = "HazardRequest"
BRAKE_COL = "BrakeRequest"

GT_START_CANDIDATES = ["Start_Time", "Start_Time_sec", "Start", "Start Time", "Start_sec"]
GT_END_CANDIDATES = ["End_Time", "End_Time_sec", "End", "End Time", "End_sec"]
GT_EVENT_CANDIDATES = ["Event_ID", "EventID", "Event", "ID"]
GT_LABEL_CANDIDATES = ["GroundTruth", "Ground_Truth", "Target", "Target_Decision", "Label"]


def find_col(df, candidates, required=True):
    normalized = {str(c).strip().lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.strip().lower().replace("_", "").replace(" ", "")
        if key in normalized:
            return normalized[key]
    if required:
        raise ValueError(f"Column not found. Needed one of {candidates}. Existing: {list(df.columns)}")
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
    out = pd.DataFrame()
    out["time"] = pd.to_numeric(df[time_col], errors="coerce")

    for col in [MEDICAL_COL, HAZARD_COL, BRAKE_COL]:
        out[col] = to_binary(df[col]) if col in df.columns else 0

    return out.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)


def load_gt(path):
    gt = pd.read_excel(path)
    start_col = find_col(gt, GT_START_CANDIDATES)
    end_col = find_col(gt, GT_END_CANDIDATES)
    event_col = find_col(gt, GT_EVENT_CANDIDATES, required=False)
    label_col = find_col(gt, GT_LABEL_CANDIDATES, required=False)

    out = pd.DataFrame()
    out["start"] = pd.to_numeric(gt[start_col], errors="coerce")
    out["end"] = pd.to_numeric(gt[end_col], errors="coerce")
    out["event_id"] = gt[event_col].astype(str) if event_col else [f"M{i+1}" for i in range(len(gt))]
    out["label"] = gt[label_col].astype(str) if label_col else "Medical"

    out = out.dropna(subset=["start", "end"])
    out = out[out["end"] >= out["start"]]
    out = out[out["label"].str.lower().str.contains("medical", na=False)]
    return out.sort_values("start").reset_index(drop=True)


def evaluate_one_participant(csv_path, gt_path):
    pid = participant_id_from_name(csv_path.name)
    df = load_csv(csv_path)
    gt = load_gt(gt_path)

    rows = []
    for _, ev in gt.iterrows():
        start, end = float(ev["start"]), float(ev["end"])
        segment = df[(df["time"] >= start) & (df["time"] <= end)]

        detected = int(segment[MEDICAL_COL].max()) if not segment.empty else 0
        hazard = int(segment[HAZARD_COL].max()) if not segment.empty else 0
        brake = int(segment[BRAKE_COL].max()) if not segment.empty else 0

        first_det = np.nan
        delay = np.nan
        if detected:
            det_times = segment.loc[segment[MEDICAL_COL] == 1, "time"]
            if len(det_times) > 0:
                first_det = float(det_times.iloc[0])
                delay = first_det - start

        rows.append({
            "Participant": pid,
            "Event_ID": ev["event_id"],
            "Start": start,
            "End": end,
            "Detected": detected,
            "Hazard": hazard,
            "Brake": brake,
            "First_Detection_Time": first_det,
            "Detection_Delay_s": delay,
            "Result": "TP" if detected else "FN"
        })

    event_results = pd.DataFrame(rows)

    # False positive = MedicalEmergencyConfirmed outside all medical GT events
    # False positive events = MedicalEmergencyConfirmed segments that do not overlap
    # with any GT medical event, using a small tolerance around GT boundaries.
    medical_df = df[df[MEDICAL_COL] == 1][["time"]].copy()
    fp_times = []
    if not medical_df.empty:
        outside_times = []
        for t in medical_df["time"].values:
            inside_any_event = (((gt["start"] - GT_TOLERANCE_BEFORE_S) <= t) & ((gt["end"] + GT_TOLERANCE_AFTER_S) >= t)).any()
            if not inside_any_event:
                outside_times.append(float(t))

        # Count continuous outside detections as one FP event, not one FP per frame/sample.
        if outside_times:
            outside_times = sorted(outside_times)
            fp_times.append(outside_times[0])
            for prev_t, cur_t in zip(outside_times, outside_times[1:]):
                if cur_t - prev_t > 2.0:  # new independent false-positive episode
                    fp_times.append(cur_t)

    return event_results, fp_times


def safe_div(a, b):
    return float(a / b) if b else 0.0


def main():
    csv_files = sorted(CSV_DIR.glob("*.csv"))
    gt_files = sorted(list(GT_DIR.glob("*.xlsx")) + list(GT_DIR.glob("*.xls")))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {CSV_DIR}")
    if not gt_files:
        raise FileNotFoundError(f"No GT Excel files found in {GT_DIR}")

    gt_by_pid = {participant_id_from_name(p.name): p for p in gt_files}

    all_events = []
    fp_rows = []

    for csv_path in csv_files:
        pid = participant_id_from_name(csv_path.name)
        gt_path = gt_by_pid.get(pid)
        if gt_path is None:
            print(f"Warning: no GT file found for {pid}; skipped {csv_path.name}")
            continue

        print(f"Processing {pid}: {csv_path.name} + {gt_path.name}")
        event_results, fp_times = evaluate_one_participant(csv_path, gt_path)
        all_events.append(event_results)
        for t in fp_times:
            fp_rows.append({"Participant": pid, "False_Positive_Time": t})

    if not all_events:
        raise RuntimeError("No files were processed. Check file names contain P01, P02, etc.")

    events_df = pd.concat(all_events, ignore_index=True)
    fp_df = pd.DataFrame(fp_rows)

    TP = int((events_df["Detected"] == 1).sum())
    FN = int((events_df["Detected"] == 0).sum())
    FP = int(len(fp_df))

    event_accuracy = safe_div(TP, TP + FN + FP)
    precision = safe_div(TP, TP + FP)
    recall = safe_div(TP, TP + FN)
    f1 = safe_div(2 * precision * recall, precision + recall)

    hazard_success = safe_div(events_df.loc[events_df["Detected"] == 1, "Hazard"].sum(), TP)
    brake_success = safe_div(events_df.loc[events_df["Detected"] == 1, "Brake"].sum(), TP)
    avg_delay = events_df.loc[events_df["Detected"] == 1, "Detection_Delay_s"].mean()

    summary = pd.DataFrame([{
        "Participants": events_df["Participant"].nunique(),
        "Medical_Events": len(events_df),
        "TP_Detected_Events": TP,
        "FN_Missed_Events": FN,
        "FP_Outside_GT": FP,
        "Event_Accuracy_no_TN_%": event_accuracy * 100,
        "Precision_%": precision * 100,
        "Recall_Detection_Rate_%": recall * 100,
        "F1_score_%": f1 * 100,
        "Hazard_Activation_Success_%": hazard_success * 100,
        "Brake_Activation_Success_%": brake_success * 100,
        "Average_Detection_Delay_s": avg_delay
    }])

    out_xlsx = OUTPUT_DIR / "medical_event_evaluation_results.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        events_df.to_excel(writer, sheet_name="Event_Results", index=False)
        fp_df.to_excel(writer, sheet_name="False_Positives", index=False)

    metrics = {
        "Detection Rate": recall * 100,
        "Precision": precision * 100,
        "F1-score": f1 * 100,
        "Hazard Success": hazard_success * 100,
        "Brake Success": brake_success * 100
    }

    plt.figure(figsize=(9, 5))
    plt.bar(list(metrics.keys()), list(metrics.values()))
    plt.ylim(0, 105)
    plt.ylabel("Percentage (%)")
    plt.title("Medical Emergency Event-Based Performance")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "medical_metrics_bar_chart.png", dpi=300)
    plt.close()

    cm = np.array([[TP, FN], [FP, 0]])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Medical Event Confusion Matrix")
    plt.xticks([0, 1], ["Predicted Medical", "Missed"])
    plt.yticks([0, 1], ["Actual Medical", "False Positive"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "medical_confusion_matrix.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    delays = events_df.loc[events_df["Detected"] == 1, "Detection_Delay_s"].dropna()
    plt.plot(range(1, len(delays) + 1), delays, marker="o")
    plt.xlabel("Detected Medical Event")
    plt.ylabel("Detection Delay (s)")
    plt.title("Medical Detection Delay per Event")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "medical_detection_delay.png", dpi=300)
    plt.close()

    print("\n========== SUMMARY ==========")
    print(summary.to_string(index=False))
    print(f"\nSaved Excel results: {out_xlsx}")
    print(f"Saved figures in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
