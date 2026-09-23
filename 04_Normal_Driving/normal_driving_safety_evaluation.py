# normal_driving_safety_evaluation.py
# Normal Driving Evaluation - Safety-Decision Based
#
# الهدف:
# تقييم القيادة الطبيعية على أساس القرارات النهائية وإجراءات السلامة، وليس الفلاكات اللحظية.
#
# في NORMAL:
# - لا نحسب ConfirmedDistractionFlag أو ConfirmedDrowsinessFlag أو FinalSystemDecision القصيرة كـ False Alarm.
# - نحسب فقط الإنذارات/الإجراءات النهائية:
#     FrequentYawningAlert
#     MedicalEmergencyConfirmed
#     HazardRequest
#     BrakeRequest
#
# المخرجات:
# - Normal Stability
# - Specificity
# - False Alarm Episodes
# - False Alarm Rate
# - Medical False Alarm
# - Hazard False Activation
# - Brake False Activation
# - Frequent Yawning False Alert
# - Transient candidate analysis
# - Excel + figures
#
# المجلدات:
# normal_test/
#   normal_driving_safety_evaluation.py
#   normal_csv/
#   normal_gt/
#
# تشغيل:
# python normal_driving_safety_evaluation.py

import re
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


CSV_DIR = Path("normal_csv")
GT_DIR = Path("normal_gt")
OUTPUT_DIR = Path("normal_results_safety")
OUTPUT_DIR.mkdir(exist_ok=True)

TIME_CANDIDATES = ["ADJUSTED TIME", "Adjusted time", "AdjustedTime", "ElapsedSeconds", "Elapsed Seconds"]
SPEED_CANDIDATES = ["SpeedDisplayKmh", "VehicleSpeed", "Speed", "Vehicle Speed"]
ALERTS_CANDIDATES = ["AlertsEnabled", "Alerts Enabled"]

FINAL_DECISION_COL = "FinalSystemDecision"
DISTRACTION_COL = "ConfirmedDistractionFlag"
DROWSINESS_COL = "ConfirmedDrowsinessFlag"
YAWN_ALERT_COL = "FrequentYawningAlert"
MEDICAL_COL = "MedicalEmergencyConfirmed"
HAZARD_COL = "HazardRequest"
BRAKE_COL = "BrakeRequest"

GT_START_CANDIDATES = ["Start_Time", "Start_Time_sec", "Start", "Start Time", "Start_sec"]
GT_END_CANDIDATES = ["End_Time", "End_Time_sec", "End", "End Time", "End_sec"]
GT_EVENT_CANDIDATES = ["Event_ID", "EventID", "Event", "ID"]
GT_LABEL_CANDIDATES = ["GroundTruth", "Ground_Truth", "Target", "Target_Decision", "Label"]

# هذه تستخدم فقط للتحليل الوصفي، ولا تدخل في حساب False Alarm
TRANSIENT_DISTRACTION_THRESHOLD_SEC = 3.0
TRANSIENT_DROWSINESS_THRESHOLD_SEC = 5.0

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
    out["speed"] = pd.to_numeric(df[speed_col], errors="coerce") if speed_col else pd.NA
    out["alerts_enabled"] = to_binary(df[alerts_col]) if alerts_col else 1

    for col in [DISTRACTION_COL, DROWSINESS_COL, YAWN_ALERT_COL, MEDICAL_COL, HAZARD_COL, BRAKE_COL]:
        out[col] = to_binary(df[col]) if col in df.columns else 0

    out[FINAL_DECISION_COL] = df[FINAL_DECISION_COL].astype(str) if FINAL_DECISION_COL in df.columns else "UNKNOWN"
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

    out = out.dropna(subset=["start", "end"])
    out = out[out["end"] >= out["start"]]
    out = out[out["label"].isin(["NORMAL", "IGNORE"])]
    return out.sort_values("start").reset_index(drop=True)


def intervals_from_binary(df, col, event_type):
    if df.empty:
        return pd.DataFrame(columns=["Type", "Start", "End", "Duration_s"])

    s = df[col].astype(int).values
    times = df["time"].values

    rows = []
    active = False
    start_t = None
    prev = 0

    for i, val in enumerate(s):
        if prev == 0 and val == 1:
            active = True
            start_t = float(times[i])
        if prev == 1 and val == 0 and active:
            end_t = float(times[i - 1])
            rows.append({
                "Type": event_type,
                "Start": start_t,
                "End": end_t,
                "Duration_s": max(0.0, end_t - start_t)
            })
            active = False
        prev = val

    if active:
        end_t = float(times[-1])
        rows.append({
            "Type": event_type,
            "Start": start_t,
            "End": end_t,
            "Duration_s": max(0.0, end_t - start_t)
        })

    return pd.DataFrame(rows)


def intervals_from_final_decision(df):
    if df.empty or FINAL_DECISION_COL not in df.columns:
        return pd.DataFrame(columns=["Type", "Start", "End", "Duration_s", "Decision"])

    normal_words = ["normal", "attentive", "safe", "monitor", "none"]
    rows = []
    current_decision = None
    start_t = None
    prev_is_abnormal = False

    for _, r in df.iterrows():
        t = float(r["time"])
        dec = str(r[FINAL_DECISION_COL]).strip()
        low = dec.lower()
        is_abnormal = not any(w in low for w in normal_words)

        if is_abnormal and not prev_is_abnormal:
            current_decision = dec
            start_t = t
        elif is_abnormal and prev_is_abnormal and dec != current_decision:
            rows.append({
                "Type": "FinalDecisionTransient",
                "Start": start_t,
                "End": t,
                "Duration_s": max(0.0, t - start_t),
                "Decision": current_decision
            })
            current_decision = dec
            start_t = t
        elif (not is_abnormal) and prev_is_abnormal:
            rows.append({
                "Type": "FinalDecisionTransient",
                "Start": start_t,
                "End": t,
                "Duration_s": max(0.0, t - start_t),
                "Decision": current_decision
            })
            current_decision = None
            start_t = None

        prev_is_abnormal = is_abnormal

    if prev_is_abnormal and start_t is not None:
        rows.append({
            "Type": "FinalDecisionTransient",
            "Start": start_t,
            "End": float(df["time"].iloc[-1]),
            "Duration_s": max(0.0, float(df["time"].iloc[-1]) - start_t),
            "Decision": current_decision
        })

    return pd.DataFrame(rows)


def overlaps_normal(start, end, normal_gt):
    for _, seg in normal_gt.iterrows():
        if start <= float(seg["end"]) and end >= float(seg["start"]):
            return True
    return False


def filter_to_normal(events, normal_gt):
    if events.empty:
        return events
    rows = []
    for _, ev in events.iterrows():
        if overlaps_normal(float(ev["Start"]), float(ev["End"]), normal_gt):
            rows.append(dict(ev))
    return pd.DataFrame(rows)


def evaluate_one(csv_path, gt_path):
    pid = participant_id_from_name(csv_path.name)
    df = load_csv(csv_path)
    gt = load_gt(gt_path)
    normal_gt = gt[gt["label"] == "NORMAL"].copy()

    # Safety final decisions: these are real false alarms if they appear in Normal
    safety_events = pd.concat([
        intervals_from_binary(df, YAWN_ALERT_COL, "FrequentYawningAlert"),
        intervals_from_binary(df, MEDICAL_COL, "MedicalEmergencyConfirmed"),
        intervals_from_binary(df, HAZARD_COL, "HazardRequest"),
        intervals_from_binary(df, BRAKE_COL, "BrakeRequest"),
    ], ignore_index=True)

    safety_events = filter_to_normal(safety_events, normal_gt)
    if not safety_events.empty:
        safety_events["Participant"] = pid

    # Transient/candidate analysis only, not counted as false alarms
    transient_events = pd.concat([
        intervals_from_binary(df, DISTRACTION_COL, "DistractionCandidate"),
        intervals_from_binary(df, DROWSINESS_COL, "DrowsinessCandidate"),
        intervals_from_final_decision(df),
    ], ignore_index=True)

    transient_events = filter_to_normal(transient_events, normal_gt)
    if not transient_events.empty:
        transient_events["Participant"] = pid
        transient_events["Counted_As_FP"] = False

    normal_segments = len(normal_gt)
    false_alarm_count = len(safety_events)
    stable = 1 if normal_segments > 0 and false_alarm_count == 0 else 0

    summary = pd.DataFrame([{
        "Participant": pid,
        "Normal_Segments": normal_segments,
        "Stable_Normal": stable,
        "Safety_False_Alarm_Episodes": false_alarm_count,
        "FrequentYawning_False_Alerts": int((safety_events["Type"] == "FrequentYawningAlert").sum()) if not safety_events.empty else 0,
        "Medical_False_Alarms": int((safety_events["Type"] == "MedicalEmergencyConfirmed").sum()) if not safety_events.empty else 0,
        "Hazard_False_Activations": int((safety_events["Type"] == "HazardRequest").sum()) if not safety_events.empty else 0,
        "Brake_False_Activations": int((safety_events["Type"] == "BrakeRequest").sum()) if not safety_events.empty else 0,
        "Transient_Events_Not_Counted": len(transient_events),
        "Max_Transient_Duration_s": float(transient_events["Duration_s"].max()) if not transient_events.empty else 0.0,
        "Distraction_Candidate_Events": int((transient_events["Type"] == "DistractionCandidate").sum()) if not transient_events.empty else 0,
        "Drowsiness_Candidate_Events": int((transient_events["Type"] == "DrowsinessCandidate").sum()) if not transient_events.empty else 0,
        "FinalDecision_Transient_Events": int((transient_events["Type"] == "FinalDecisionTransient").sum()) if not transient_events.empty else 0,
    }])

    gt["Participant"] = pid
    return summary, safety_events, transient_events, gt


def safe_div(a, b):
    return float(a / b) if b else 0.0


def make_bar(values, title, ylabel, out_path, ylim=None):
    plt.figure(figsize=(8, 5))
    plt.bar(list(values.keys()), list(values.values()))
    if ylim:
        plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
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

    all_summary = []
    all_safety = []
    all_transient = []
    all_gt = []

    for csv_path in csv_files:
        pid = participant_id_from_name(csv_path.name)
        gt_path = gt_by_pid.get(pid)
        if gt_path is None:
            print(f"Warning: no GT for {pid}; skipped {csv_path.name}")
            continue

        print(f"Processing {pid}: {csv_path.name} + {gt_path.name}")
        summary, safety_events, transient_events, gt = evaluate_one(csv_path, gt_path)

        all_summary.append(summary)
        if not safety_events.empty:
            all_safety.append(safety_events)
        if not transient_events.empty:
            all_transient.append(transient_events)
        all_gt.append(gt)

    if not all_summary:
        raise RuntimeError("No matching files processed. Make sure names contain P01/P02...")

    participant_df = pd.concat(all_summary, ignore_index=True)
    safety_df = pd.concat(all_safety, ignore_index=True) if all_safety else pd.DataFrame()
    transient_df = pd.concat(all_transient, ignore_index=True) if all_transient else pd.DataFrame()
    gt_df = pd.concat(all_gt, ignore_index=True)

    total_segments = int(participant_df["Normal_Segments"].sum())
    stable_segments = int(participant_df["Stable_Normal"].sum())
    false_alarms = int(participant_df["Safety_False_Alarm_Episodes"].sum())

    stability = safe_div(stable_segments, total_segments) * 100
    specificity = stability
    false_alarm_rate = safe_div(false_alarms, total_segments)
    false_alarm_free_rate = stability

    overall_df = pd.DataFrame([{
        "Participants": participant_df["Participant"].nunique(),
        "Normal_Segments": total_segments,
        "Stable_Normal_Segments": stable_segments,
        "Safety_False_Alarm_Episodes": false_alarms,
        "Normal_Stability_%": stability,
        "Specificity_%": specificity,
        "False_Alarm_Free_Rate_%": false_alarm_free_rate,
        "False_Alarm_Rate_per_Normal_Segment": false_alarm_rate,
        "FrequentYawning_False_Alerts": int(participant_df["FrequentYawning_False_Alerts"].sum()),
        "Medical_False_Alarms": int(participant_df["Medical_False_Alarms"].sum()),
        "Hazard_False_Activations": int(participant_df["Hazard_False_Activations"].sum()),
        "Brake_False_Activations": int(participant_df["Brake_False_Activations"].sum()),
        "Transient_Events_Not_Counted": int(participant_df["Transient_Events_Not_Counted"].sum()),
        "Max_Transient_Duration_s": float(participant_df["Max_Transient_Duration_s"].max())
    }])

    safety_by_type = safety_df.groupby("Type").size().reset_index(name="False_Alarm_Count") if not safety_df.empty else pd.DataFrame()
    transient_by_type = transient_df.groupby("Type").size().reset_index(name="Transient_Count") if not transient_df.empty else pd.DataFrame()

    out_xlsx = OUTPUT_DIR / "normal_driving_safety_evaluation_results.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        overall_df.to_excel(writer, sheet_name="Overall", index=False)
        participant_df.to_excel(writer, sheet_name="Participant_Summary", index=False)
        safety_df.to_excel(writer, sheet_name="Safety_False_Alarms", index=False)
        transient_df.to_excel(writer, sheet_name="Transient_Events_Not_FP", index=False)
        safety_by_type.to_excel(writer, sheet_name="False_Alarms_By_Type", index=False)
        transient_by_type.to_excel(writer, sheet_name="Transient_By_Type", index=False)
        gt_df.to_excel(writer, sheet_name="Ground_Truth_Loaded", index=False)

    make_bar(
        {"Normal Stability": stability, "Specificity": specificity, "False-Alarm-Free": false_alarm_free_rate},
        "Normal Driving Safety Stability",
        "Percentage (%)",
        OUTPUT_DIR / "normal_safety_stability.png",
        ylim=(0, 105)
    )

    plt.figure(figsize=(8, 5))
    plt.bar(participant_df["Participant"], participant_df["Stable_Normal"] * 100)
    plt.ylim(0, 105)
    plt.ylabel("Stable Normal (%)")
    plt.title("Normal Stability by Participant")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "normal_stability_by_participant.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(participant_df["Participant"], participant_df["Safety_False_Alarm_Episodes"])
    plt.ylabel("Safety False Alarm Episodes")
    plt.title("Safety False Alarms by Participant")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "safety_false_alarms_by_participant.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.bar(participant_df["Participant"], participant_df["Transient_Events_Not_Counted"])
    plt.ylabel("Transient Events")
    plt.title("Transient Events Not Counted as False Alarms")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "transient_events_by_participant.png", dpi=300)
    plt.close()

    if not transient_by_type.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(transient_by_type["Type"], transient_by_type["Transient_Count"])
        plt.ylabel("Count")
        plt.title("Transient Candidate Events by Type")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "transient_events_by_type.png", dpi=300)
        plt.close()

    if not safety_by_type.empty:
        plt.figure(figsize=(8, 5))
        plt.bar(safety_by_type["Type"], safety_by_type["False_Alarm_Count"])
        plt.ylabel("Count")
        plt.title("Safety False Alarms by Type")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "safety_false_alarms_by_type.png", dpi=300)
        plt.close()

    print("\n========== NORMAL SAFETY OVERALL ==========")
    print(overall_df.to_string(index=False))
    print(f"\nSaved Excel: {out_xlsx}")
    print(f"Saved figures in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
