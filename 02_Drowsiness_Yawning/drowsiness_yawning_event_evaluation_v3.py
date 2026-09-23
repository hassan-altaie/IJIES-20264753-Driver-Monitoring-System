# drowsiness_yawning_event_evaluation_v3.py
# النسخة V3: Event-Based Evaluation مع فلترة الأحداث القصيرة
#
# الفرق عن V2:
# - لا يحسب أي 0->1 كحدث مباشرة.
# - يحذف/يرفض الأحداث القصيرة التي لم تحقق شرط الزمن.
# - يحفظ الأحداث المرفوضة في Sheet باسم Rejected_Short_Events.
#
# ضع الكود داخل مجلد التجربة:
#   drowsiness_csv/
#   drowsiness_gt/
# ثم شغل:
#   python drowsiness_yawning_event_evaluation_v3.py

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


CSV_DIR = Path("drowsiness_csv")
GT_DIR = Path("drowsiness_gt")
OUTPUT_DIR = Path("drowsiness_results_v3")
OUTPUT_DIR.mkdir(exist_ok=True)

TIME_CANDIDATES = ["ADJUSTED TIME", "Adjusted time", "AdjustedTime", "ElapsedSeconds", "Elapsed Seconds"]
SPEED_CANDIDATES = ["SpeedDisplayKmh", "VehicleSpeed", "Speed", "Vehicle Speed"]
ALERTS_CANDIDATES = ["AlertsEnabled", "Alerts Enabled"]

DROWSY_COL = "ConfirmedDrowsinessFlag"
YAWN_ACTIVE_COL = "YawnCandidateActive"
FREQ_YAWN_COL = "FrequentYawningAlert"
YAWN_COUNT_COL = "YawningEventCount"
DROWSY_COUNT_COL = "DrowsyEventCount"
FINAL_DECISION_COL = "FinalSystemDecision"

GT_START_CANDIDATES = ["Start_Time", "Start_Time_sec", "Start", "Start Time", "Start_sec"]
GT_END_CANDIDATES = ["End_Time", "End_Time_sec", "End", "End Time", "End_sec"]
GT_EVENT_CANDIDATES = ["Event_ID", "EventID", "Event", "ID"]
GT_LABEL_CANDIDATES = ["GroundTruth", "Ground_Truth", "Target", "Target_Decision", "Label"]

# سماحية بسيطة حول بداية/نهاية Ground Truth
MATCH_TOLERANCE_SECONDS = 0.50

# فلترة الأحداث القصيرة
# إذا كان الحدث أقصر من هذه المدة ولا توجد زيادة بالعداد، لا يحسب FP ولا TP، بل Rejected.
MIN_DROWSINESS_DURATION_SEC = 3.0
MIN_YAWN_DURATION_SEC = 2.0
MIN_FREQUENT_YAWN_ALERT_DURATION_SEC = 0.0  # التنبيه قد يكون قصيراً، لذلك لا نفلتره زمنياً

USE_ALERTS_ENABLED_FILTER = False
USE_SPEED_FILTER = False
SPEED_THRESHOLD = 10.0


def normalize_name(x):
    return str(x).strip().lower().replace("_", "").replace(" ", "")


def find_col(df, candidates, required=True):
    normalized = {normalize_name(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_name(cand)
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
    speed_col = find_col(df, SPEED_CANDIDATES, required=False)
    alerts_col = find_col(df, ALERTS_CANDIDATES, required=False)

    out = pd.DataFrame()
    out["time"] = pd.to_numeric(df[time_col], errors="coerce")
    out["speed"] = pd.to_numeric(df[speed_col], errors="coerce") if speed_col else np.nan
    out["alerts_enabled"] = to_binary(df[alerts_col]) if alerts_col else 1

    for col in [DROWSY_COL, YAWN_ACTIVE_COL, FREQ_YAWN_COL]:
        out[col] = to_binary(df[col]) if col in df.columns else 0

    for col in [YAWN_COUNT_COL, DROWSY_COUNT_COL]:
        out[col] = pd.to_numeric(df[col], errors="coerce").fillna(0) if col in df.columns else 0

    out[FINAL_DECISION_COL] = df[FINAL_DECISION_COL].astype(str) if FINAL_DECISION_COL in df.columns else ""

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
    out["event_id"] = gt[event_col].astype(str) if event_col else [f"E{i+1}" for i in range(len(gt))]
    out["label"] = gt[label_col].astype(str).str.strip().str.upper()
    out["label"] = out["label"].replace({
        "FREQUENT_YAWNING_ALERT": "FREQUENT YAWNING ALERT",
        "FREQUENT YAWNING": "FREQUENT YAWNING ALERT",
        "FY": "FREQUENT YAWNING ALERT"
    })

    out = out.dropna(subset=["start", "end"])
    out = out[out["end"] >= out["start"]]
    out = out[out["label"].isin(["DROWSINESS", "YAWN", "FREQUENT YAWNING ALERT"])]
    return out.sort_values("start").reset_index(drop=True)


def intervals_from_binary(df, col, event_type):
    """
    يحول الإشارة الثنائية إلى أحداث:
    كل transition من 0 إلى 1 = بداية حدث واحد.
    الحدث ينتهي عند العودة إلى 0.
    """
    if df.empty:
        return pd.DataFrame(columns=["pred_id", "type", "start", "end", "duration", "start_index", "end_index"])

    s = df[col].astype(int).values
    times = df["time"].values

    rows = []
    active = False
    start_t = None
    start_i = None
    pred_idx = 1
    prev = 0

    for i, val in enumerate(s):
        if prev == 0 and val == 1:
            active = True
            start_t = float(times[i])
            start_i = i
        if prev == 1 and val == 0 and active:
            end_i = i - 1
            end_t = float(times[end_i])
            rows.append({
                "pred_id": f"{event_type[:2]}P{pred_idx}",
                "type": event_type,
                "start": start_t,
                "end": end_t,
                "duration": max(0.0, end_t - start_t),
                "start_index": start_i,
                "end_index": end_i
            })
            pred_idx += 1
            active = False
        prev = val

    if active:
        end_i = len(times) - 1
        end_t = float(times[-1])
        rows.append({
            "pred_id": f"{event_type[:2]}P{pred_idx}",
            "type": event_type,
            "start": start_t,
            "end": end_t,
            "duration": max(0.0, end_t - start_t),
            "start_index": start_i,
            "end_index": end_i
        })

    return pd.DataFrame(rows)


def counter_increased_near_event(df, event, counter_col, lookahead_sec=1.5):
    """
    يتحقق هل العداد زاد قرب الحدث.
    هذا مهم لأن بعض الحالات القصيرة لا تُسجل في العداد، وبالتالي لا نعتبرها حدثاً مؤكداً.
    """
    if counter_col not in df.columns:
        return False

    start = float(event["start"])
    end = float(event["end"]) + lookahead_sec
    before = df[df["time"] < start][counter_col]
    during_after = df[(df["time"] >= start) & (df["time"] <= end)][counter_col]

    if during_after.empty:
        return False

    before_max = float(before.max()) if not before.empty else 0.0
    after_max = float(during_after.max())
    return after_max > before_max


def split_valid_and_rejected(df, events):
    """
    يقسم الأحداث المتنبأ بها إلى:
    - valid_events: تدخل في الحساب
    - rejected_events: قصيرة ولم تحقق العداد، لا تدخل FP
    """
    if events.empty:
        return events.copy(), events.copy()

    valid_rows = []
    rejected_rows = []

    for _, ev in events.iterrows():
        typ = ev["type"]
        duration = float(ev["duration"])

        if typ == "DROWSINESS":
            min_duration = MIN_DROWSINESS_DURATION_SEC
            count_ok = counter_increased_near_event(df, ev, DROWSY_COUNT_COL)
        elif typ == "YAWN":
            min_duration = MIN_YAWN_DURATION_SEC
            count_ok = counter_increased_near_event(df, ev, YAWN_COUNT_COL)
        else:
            min_duration = MIN_FREQUENT_YAWN_ALERT_DURATION_SEC
            count_ok = True

        duration_ok = duration >= min_duration

        if duration_ok or count_ok:
            valid_rows.append(dict(ev))
        else:
            row = dict(ev)
            row["Reject_Reason"] = f"short_duration_{duration:.2f}s_and_no_counter_increase"
            rejected_rows.append(row)

    return pd.DataFrame(valid_rows), pd.DataFrame(rejected_rows)


def overlap_seconds(a_start, a_end, b_start, b_end):
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def match_events(gt_events, pred_events, pid):
    """
    مطابقة One-to-One:
    كل GT event يطابق أفضل predicted event من نفس النوع إذا يوجد تداخل زمني.
    أي predicted event صالح وغير مطابق يحسب FP واحد.
    """
    event_rows = []
    pred_events = pred_events.copy()
    pred_events["matched"] = False

    for _, gt in gt_events.iterrows():
        label = gt["label"]
        gs = float(gt["start"]) - MATCH_TOLERANCE_SECONDS
        ge = float(gt["end"]) + MATCH_TOLERANCE_SECONDS

        candidates = pred_events[
            (pred_events["type"] == label) &
            (~pred_events["matched"]) &
            (pred_events["start"] <= ge) &
            (pred_events["end"] >= gs)
        ].copy()

        if candidates.empty:
            event_rows.append({
                "Participant": pid,
                "Event_ID": gt["event_id"],
                "GroundTruth": label,
                "GT_Start": float(gt["start"]),
                "GT_End": float(gt["end"]),
                "Pred_ID": "",
                "Pred_Start": np.nan,
                "Pred_End": np.nan,
                "Detected": 0,
                "Detection_Delay_s": np.nan,
                "Result": "FN"
            })
        else:
            candidates["overlap"] = candidates.apply(
                lambda r: overlap_seconds(gs, ge, r["start"], r["end"]), axis=1
            )
            best_idx = candidates.sort_values(["overlap", "start"], ascending=[False, True]).index[0]
            pred_events.loc[best_idx, "matched"] = True
            pred = pred_events.loc[best_idx]
            delay = max(0.0, float(pred["start"]) - float(gt["start"]))

            event_rows.append({
                "Participant": pid,
                "Event_ID": gt["event_id"],
                "GroundTruth": label,
                "GT_Start": float(gt["start"]),
                "GT_End": float(gt["end"]),
                "Pred_ID": pred["pred_id"],
                "Pred_Start": float(pred["start"]),
                "Pred_End": float(pred["end"]),
                "Detected": 1,
                "Detection_Delay_s": delay,
                "Result": "TP"
            })

    fp_rows = []
    unmatched = pred_events[~pred_events["matched"]]
    for _, pred in unmatched.iterrows():
        fp_rows.append({
            "Participant": pid,
            "Type": pred["type"],
            "Pred_ID": pred["pred_id"],
            "False_Positive_Start": float(pred["start"]),
            "False_Positive_End": float(pred["end"]),
            "Duration_s": float(pred["duration"])
        })

    return pd.DataFrame(event_rows), pd.DataFrame(fp_rows), pred_events


def evaluate_one(csv_path, gt_path):
    pid = participant_id_from_name(csv_path.name)
    df = load_csv(csv_path)
    gt = load_gt(gt_path)

    raw_drowsy = intervals_from_binary(df, DROWSY_COL, "DROWSINESS")
    raw_yawn = intervals_from_binary(df, YAWN_ACTIVE_COL, "YAWN")
    raw_fy = intervals_from_binary(df, FREQ_YAWN_COL, "FREQUENT YAWNING ALERT")
    raw_events = pd.concat([raw_drowsy, raw_yawn, raw_fy], ignore_index=True)

    valid_events, rejected_events = split_valid_and_rejected(df, raw_events)

    event_results, fp_results, matched_pred = match_events(gt, valid_events, pid)

    if not rejected_events.empty:
        rejected_events["Participant"] = pid

    counters = pd.DataFrame([{
        "Participant": pid,
        "Max_DrowsyEventCount": float(df[DROWSY_COUNT_COL].max()) if DROWSY_COUNT_COL in df.columns else 0,
        "Max_YawningEventCount": float(df[YAWN_COUNT_COL].max()) if YAWN_COUNT_COL in df.columns else 0,
        "Raw_Drowsiness_Transitions": len(raw_drowsy),
        "Raw_Yawn_Transitions": len(raw_yawn),
        "Raw_FrequentYawningAlert_Transitions": len(raw_fy),
        "Valid_Predicted_Events": len(valid_events),
        "Rejected_Short_Events": len(rejected_events),
        "GT_Drowsiness_Events": int((gt["label"] == "DROWSINESS").sum()),
        "GT_Yawn_Events": int((gt["label"] == "YAWN").sum()),
        "GT_FrequentYawningAlert_Events": int((gt["label"] == "FREQUENT YAWNING ALERT").sum()),
    }])

    return event_results, fp_results, counters, valid_events, rejected_events


def safe_div(a, b):
    return float(a / b) if b else 0.0


def metrics_for_type(events_df, fp_df, label):
    subset = events_df[events_df["GroundTruth"] == label]
    tp = int((subset["Detected"] == 1).sum())
    fn = int((subset["Detected"] == 0).sum())
    fp = int((fp_df["Type"] == label).sum()) if not fp_df.empty else 0

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    event_accuracy_no_tn = safe_div(tp, tp + fn + fp)
    avg_delay = subset.loc[subset["Detected"] == 1, "Detection_Delay_s"].mean()

    return {
        "Type": label,
        "GT_Events": len(subset),
        "TP": tp,
        "FN": fn,
        "FP": fp,
        "Event_Accuracy_no_TN_%": event_accuracy_no_tn * 100,
        "Precision_%": precision * 100,
        "Recall_Detection_Rate_%": recall * 100,
        "F1_score_%": f1 * 100,
        "Average_Detection_Delay_s": avg_delay
    }


def make_detection_matrix(summary_df, out_path):
    labels = list(summary_df["Type"])
    matrix = summary_df[["TP", "FN", "FP"]].to_numpy()

    plt.figure(figsize=(8, 4.8))
    plt.imshow(matrix)
    plt.title("Event-Based Detection Matrix")
    plt.xticks([0, 1, 2], ["TP", "FN", "FP"])
    plt.yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    csv_files = sorted(CSV_DIR.glob("*.csv"))
    gt_files = sorted(list(GT_DIR.glob("*.xlsx")) + list(GT_DIR.glob("*.xls")))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {CSV_DIR}")
    if not gt_files:
        raise FileNotFoundError(f"No GT Excel files found in {GT_DIR}")

    gt_by_pid = {participant_id_from_name(p.name): p for p in gt_files}

    all_events, all_fp, all_counters, all_pred, all_rejected = [], [], [], [], []

    for csv_path in csv_files:
        pid = participant_id_from_name(csv_path.name)
        gt_path = gt_by_pid.get(pid)
        if gt_path is None:
            print(f"Warning: no GT for {pid}; skipped {csv_path.name}")
            continue

        print(f"Processing {pid}: {csv_path.name} + {gt_path.name}")
        events, fp, counters, valid_pred, rejected = evaluate_one(csv_path, gt_path)
        all_events.append(events)
        all_fp.append(fp)
        all_counters.append(counters)

        valid_pred["Participant"] = pid
        all_pred.append(valid_pred)

        if not rejected.empty:
            all_rejected.append(rejected)

    if not all_events:
        raise RuntimeError("No matching files processed. Make sure file names contain P01/P02...")

    events_df = pd.concat(all_events, ignore_index=True)
    fp_df = pd.concat(all_fp, ignore_index=True) if all_fp else pd.DataFrame(
        columns=["Participant", "Type", "Pred_ID", "False_Positive_Start", "False_Positive_End", "Duration_s"]
    )
    counters_df = pd.concat(all_counters, ignore_index=True)
    pred_df = pd.concat(all_pred, ignore_index=True)
    rejected_df = pd.concat(all_rejected, ignore_index=True) if all_rejected else pd.DataFrame()

    labels = ["DROWSINESS", "YAWN", "FREQUENT YAWNING ALERT"]
    summary_df = pd.DataFrame([metrics_for_type(events_df, fp_df, lab) for lab in labels])

    TP = int(summary_df["TP"].sum())
    FN = int(summary_df["FN"].sum())
    FP = int(summary_df["FP"].sum())
    precision = safe_div(TP, TP + FP)
    recall = safe_div(TP, TP + FN)
    f1 = safe_div(2 * precision * recall, precision + recall)
    acc = safe_div(TP, TP + FN + FP)

    overall_df = pd.DataFrame([{
        "Module": "Overall Drowsiness + Yawning Event-Based V3",
        "Participants": events_df["Participant"].nunique(),
        "GT_Events": int(summary_df["GT_Events"].sum()),
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "Event_Accuracy_no_TN_%": acc * 100,
        "Precision_%": precision * 100,
        "Recall_Detection_Rate_%": recall * 100,
        "F1_score_%": f1 * 100,
        "Average_Detection_Delay_s": events_df.loc[events_df["Detected"] == 1, "Detection_Delay_s"].mean(),
        "Rejected_Short_Events": int(len(rejected_df))
    }])

    out_xlsx = OUTPUT_DIR / "drowsiness_yawning_evaluation_v3_results.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        overall_df.to_excel(writer, sheet_name="Overall", index=False)
        summary_df.to_excel(writer, sheet_name="Summary_By_Type", index=False)
        events_df.to_excel(writer, sheet_name="Event_Results", index=False)
        fp_df.to_excel(writer, sheet_name="False_Positives", index=False)
        pred_df.to_excel(writer, sheet_name="Valid_Predicted_Events", index=False)
        counters_df.to_excel(writer, sheet_name="Counters", index=False)
        rejected_df.to_excel(writer, sheet_name="Rejected_Short_Events", index=False)

    # Figures
    plt.figure(figsize=(9, 5))
    plt.bar(summary_df["Type"], summary_df["Recall_Detection_Rate_%"])
    plt.ylim(0, 105)
    plt.ylabel("Detection Rate (%)")
    plt.title("Drowsiness and Yawning Detection Rate by Type")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "v3_detection_rate_by_type.png", dpi=300)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.bar(["Precision", "Recall", "F1-score"], [
        overall_df["Precision_%"].iloc[0],
        overall_df["Recall_Detection_Rate_%"].iloc[0],
        overall_df["F1_score_%"].iloc[0]
    ])
    plt.ylim(0, 105)
    plt.ylabel("Percentage (%)")
    plt.title("Overall Drowsiness + Yawning Event-Based Performance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "v3_overall_metrics.png", dpi=300)
    plt.close()

    make_detection_matrix(summary_df, OUTPUT_DIR / "v3_event_detection_matrix.png")

    delay_data = events_df.dropna(subset=["Detection_Delay_s"])
    if not delay_data.empty:
        plt.figure(figsize=(9, 5))
        for label in labels:
            d = delay_data[delay_data["GroundTruth"] == label]["Detection_Delay_s"].values
            if len(d) > 0:
                plt.plot(range(1, len(d)+1), d, marker="o", label=label)
        plt.xlabel("Detected Event Index")
        plt.ylabel("Detection Delay (s)")
        plt.title("Detection Delay by Event Type")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "v3_detection_delay_by_type.png", dpi=300)
        plt.close()

    participant_summary = []
    for pid, g in events_df.groupby("Participant"):
        fp_p = fp_df[fp_df["Participant"] == pid] if not fp_df.empty else fp_df
        tp = int((g["Detected"] == 1).sum())
        fn = int((g["Detected"] == 0).sum())
        fp = int(len(fp_p))
        pr = safe_div(tp, tp + fp)
        rc = safe_div(tp, tp + fn)
        f1p = safe_div(2 * pr * rc, pr + rc)
        participant_summary.append({
            "Participant": pid,
            "GT_Events": len(g),
            "TP": tp, "FN": fn, "FP": fp,
            "Precision_%": pr*100,
            "Recall_%": rc*100,
            "F1_%": f1p*100
        })
    participant_df = pd.DataFrame(participant_summary)
    participant_df.to_csv(OUTPUT_DIR / "participant_summary.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.bar(participant_df["Participant"], participant_df["F1_%"])
    plt.ylim(0, 105)
    plt.ylabel("F1-score (%)")
    plt.title("F1-score by Participant")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "v3_f1_by_participant.png", dpi=300)
    plt.close()

    print("\n========== OVERALL ==========")
    print(overall_df.to_string(index=False))
    print("\n========== SUMMARY BY TYPE ==========")
    print(summary_df.to_string(index=False))
    print("\nSaved Excel:", out_xlsx)
    print("Saved figures in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
