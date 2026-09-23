# final_overall_driver_monitoring_evaluation.py
# النسخة النهائية المعتمدة لتجميع نتائج نظام مراقبة السائق
#
# هذه النسخة مخصصة لملفاتك الأربعة:
# 1) normal_driving_safety_evaluation_results.xlsx
# 2) distraction_event_evaluation_results.xlsx
# 3) drowsiness_yawning_evaluation_v3_results.xlsx
# 4) medical_event_evaluation_results.xlsx
#
# طريقة التشغيل:
# 1. أنشئ مجلد باسم:
#    final_results_inputs
# 2. ضع ملفات Excel الأربعة داخله.
# 3. ضع هذا الكود بجانب المجلد.
# 4. شغل:
#    python final_overall_driver_monitoring_evaluation.py
#
# المخرجات:
# final_overall_system_results/
#   final_overall_driver_monitoring_evaluation.xlsx
#   figures/*.png

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


INPUT_DIR = Path("final_results_inputs")
OUTPUT_DIR = Path("final_overall_system_results")
FIG_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)


# =========================
# أدوات عامة
# =========================

def safe_div(a, b):
    return float(a / b) if b else 0.0


def find_file(keywords):
    files = list(INPUT_DIR.glob("*.xlsx"))
    for f in files:
        name = f.name.lower()
        if all(k.lower() in name for k in keywords):
            return f
    return None


def read_sheet(path, sheet_names):
    xls = pd.ExcelFile(path)
    existing = {s.lower(): s for s in xls.sheet_names}
    for s in sheet_names:
        if s.lower() in existing:
            return pd.read_excel(path, sheet_name=existing[s.lower()])
    raise ValueError(f"Cannot find sheets {sheet_names} in {path.name}. Existing sheets: {xls.sheet_names}")


def get_first(df, names, default=np.nan):
    if df is None or df.empty:
        return default
    norm = {str(c).strip().lower().replace("_", "").replace(" ", ""): c for c in df.columns}
    for n in names:
        key = str(n).strip().lower().replace("_", "").replace(" ", "")
        if key in norm:
            v = df[norm[key]].iloc[0]
            try:
                return float(v)
            except Exception:
                return v
    return default


def make_bar(labels, values, title, ylabel, out_path, ylim=None, rotation=20):
    plt.figure(figsize=(9, 5))
    plt.bar(labels, values)
    if ylim:
        plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_grouped_bar(df, label_col, metric_cols, title, ylabel, out_path, ylim=(0, 105)):
    labels = df[label_col].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.8 / len(metric_cols)

    plt.figure(figsize=(10, 5.5))
    for i, col in enumerate(metric_cols):
        vals = pd.to_numeric(df[col], errors="coerce").fillna(0)
        plt.bar(x + i * width - 0.4 + width / 2, vals, width, label=col.replace("_%", "").replace("_", " "))
    plt.ylim(*ylim)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_detection_matrix(modules_df, out_path):
    matrix = modules_df[["TP", "FN", "FP"]].fillna(0).to_numpy()
    labels = modules_df["Module"].tolist()

    plt.figure(figsize=(7.5, 4.8))
    plt.imshow(matrix)
    plt.title("Overall Event-Based Detection Matrix")
    plt.xticks([0, 1, 2], ["TP", "FN", "FP"])
    plt.yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(int(matrix[i, j])), ha="center", va="center", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_radar(modules_df, out_path):
    metrics = ["Precision_%", "Recall_Detection_Rate_%", "F1_score_%"]
    labels = ["Precision", "Recall", "F1-score"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, polar=True)

    for _, row in modules_df.iterrows():
        values = [float(row[m]) if not pd.isna(row[m]) else 0 for m in metrics]
        values += values[:1]
        ax.plot(angles, values, marker="o", label=row["Module"])
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_title("Performance Radar Chart by Module")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def make_delay_figures(events_df):
    if events_df.empty or "Detection_Delay_s" not in events_df.columns:
        return

    df = events_df.copy()
    df["Detection_Delay_s"] = pd.to_numeric(df["Detection_Delay_s"], errors="coerce")
    df = df.dropna(subset=["Detection_Delay_s"])

    if df.empty:
        return

    # Histogram
    plt.figure(figsize=(8, 5))
    plt.hist(df["Detection_Delay_s"], bins=15)
    plt.xlabel("Detection Delay (s)")
    plt.ylabel("Number of Events")
    plt.title("Overall Detection Delay Histogram")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "overall_detection_delay_histogram.png", dpi=300)
    plt.close()

    # Boxplot by module
    data = []
    labels = []
    for module, g in df.groupby("Module"):
        vals = g["Detection_Delay_s"].dropna().values
        if len(vals) > 0:
            data.append(vals)
            labels.append(module)

    if data:
        plt.figure(figsize=(8, 5))
        plt.boxplot(data, labels=labels)
        plt.ylabel("Detection Delay (s)")
        plt.title("Detection Delay Distribution by Module")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(FIG_DIR / "detection_delay_boxplot_by_module.png", dpi=300)
        plt.close()

    # Mean delay by module
    delay_by_module = df.groupby("Module")["Detection_Delay_s"].mean().reset_index()
    make_bar(
        delay_by_module["Module"],
        delay_by_module["Detection_Delay_s"],
        "Average Detection Delay by Module",
        "Average Detection Delay (s)",
        FIG_DIR / "average_detection_delay_by_module.png",
        ylim=None
    )


# =========================
# تحميل الملفات
# =========================

def load_all_inputs():
    if not INPUT_DIR.exists():
        raise FileNotFoundError("Create folder final_results_inputs and put the four Excel result files inside it.")

    normal_file = find_file(["normal"])
    distraction_file = find_file(["distraction"])
    drowsiness_file = find_file(["drowsiness"])
    medical_file = find_file(["medical"])

    missing = []
    if normal_file is None:
        missing.append("Normal")
    if distraction_file is None:
        missing.append("Distraction")
    if drowsiness_file is None:
        missing.append("Drowsiness/Yawning")
    if medical_file is None:
        missing.append("Medical")
    if missing:
        raise FileNotFoundError("Missing result files: " + ", ".join(missing))

    print("Loaded files:")
    print("Normal:", normal_file.name)
    print("Distraction:", distraction_file.name)
    print("Drowsiness/Yawning:", drowsiness_file.name)
    print("Medical:", medical_file.name)

    return normal_file, distraction_file, drowsiness_file, medical_file


# =========================
# استخراج نتائج كل حالة
# =========================

def extract_normal(normal_file):
    overall = read_sheet(normal_file, ["Overall"])
    participant = read_sheet(normal_file, ["Participant_Summary"])

    normal_metrics = pd.DataFrame([{
        "Module": "Normal Driving",
        "Participants": get_first(overall, ["Participants"], 0),
        "Normal_Segments": get_first(overall, ["Normal_Segments"], 0),
        "Stable_Normal_Segments": get_first(overall, ["Stable_Normal_Segments"], 0),
        "Normal_Stability_%": get_first(overall, ["Normal_Stability_%"], np.nan),
        "Specificity_%": get_first(overall, ["Specificity_%"], np.nan),
        "False_Alarm_Free_Rate_%": get_first(overall, ["False_Alarm_Free_Rate_%"], np.nan),
        "False_Alarm_Rate_per_Normal_Segment": get_first(overall, ["False_Alarm_Rate_per_Normal_Segment"], np.nan),
        "Safety_False_Alarm_Episodes": get_first(overall, ["Safety_False_Alarm_Episodes", "False_Alarm_Episodes"], 0),
        "Medical_False_Alarms": get_first(overall, ["Medical_False_Alarms"], 0),
        "Hazard_False_Activations": get_first(overall, ["Hazard_False_Activations"], 0),
        "Brake_False_Activations": get_first(overall, ["Brake_False_Activations"], 0),
        "Transient_Events_Not_Counted": get_first(overall, ["Transient_Events_Not_Counted"], 0),
    }])

    participant["Module"] = "Normal Driving"
    return normal_metrics, participant


def extract_positive_module(path, module_name):
    overall = read_sheet(path, ["Overall", "Summary"])

    metrics = pd.DataFrame([{
        "Module": module_name,
        "Participants": get_first(overall, ["Participants"], 0),
        "Events": get_first(overall, ["Distraction_Events", "GT_Events", "Medical_Events"], 0),
        "TP": get_first(overall, ["TP", "TP_Detected_Events"], 0),
        "FN": get_first(overall, ["FN", "FN_Missed_Events"], 0),
        "FP": get_first(overall, ["FP", "FP_Outside_GT"], 0),
        "Event_Accuracy_no_TN_%": get_first(overall, ["Event_Accuracy_no_TN_%"], np.nan),
        "Precision_%": get_first(overall, ["Precision_%"], np.nan),
        "Recall_Detection_Rate_%": get_first(overall, ["Recall_Detection_Rate_%"], np.nan),
        "F1_score_%": get_first(overall, ["F1_score_%"], np.nan),
        "Average_Detection_Delay_s": get_first(overall, ["Average_Detection_Delay_s"], np.nan),
        "Hazard_Activation_Success_%": get_first(overall, ["Hazard_Activation_Success_%", "Hazard_Success_%"], np.nan),
        "Brake_Activation_Success_%": get_first(overall, ["Brake_Activation_Success_%", "Brake_Success_%"], np.nan),
        "Safety_False_Activations": get_first(overall, ["Safety_False_Activations"], 0),
    }])

    return metrics


def extract_class_summary(path, module_name):
    try:
        df = read_sheet(path, ["Summary_By_Class", "Summary_By_Type"])
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    if "Class" not in df.columns and "Type" in df.columns:
        df = df.rename(columns={"Type": "Class"})

    df["Module"] = module_name
    return df


def extract_participant_summary(path, module_name):
    try:
        df = read_sheet(path, ["Participant_Summary"])
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    df["Module"] = module_name
    return df


def extract_event_results(path, module_name):
    try:
        df = read_sheet(path, ["Event_Results"])
    except Exception:
        return pd.DataFrame()

    if df.empty:
        return df

    df["Module"] = module_name
    return df


# =========================
# Main
# =========================

def main():
    normal_file, distraction_file, drowsiness_file, medical_file = load_all_inputs()

    normal_metrics, normal_participant = extract_normal(normal_file)

    distraction_metrics = extract_positive_module(distraction_file, "Distraction")
    drowsiness_metrics = extract_positive_module(drowsiness_file, "Drowsiness + Yawning")
    medical_metrics = extract_positive_module(medical_file, "Medical Emergency")

    module_metrics = pd.concat(
        [distraction_metrics, drowsiness_metrics, medical_metrics],
        ignore_index=True
    )

    # الحساب الشامل للفئات الموجبة فقط
    TP = int(module_metrics["TP"].sum())
    FN = int(module_metrics["FN"].sum())
    FP = int(module_metrics["FP"].sum())
    EVENTS = int(module_metrics["Events"].sum())

    overall_precision = safe_div(TP, TP + FP) * 100
    overall_recall = safe_div(TP, TP + FN) * 100
    overall_f1 = safe_div(2 * overall_precision * overall_recall, overall_precision + overall_recall)
    overall_CSI = safe_div(TP, TP + FN + FP) * 100

    all_events = pd.concat([
        extract_event_results(distraction_file, "Distraction"),
        extract_event_results(drowsiness_file, "Drowsiness + Yawning"),
        extract_event_results(medical_file, "Medical Emergency")
    ], ignore_index=True)

    if not all_events.empty and "Detection_Delay_s" in all_events.columns:
        delays = pd.to_numeric(all_events["Detection_Delay_s"], errors="coerce").dropna()
    else:
        delays = pd.Series(dtype=float)

    normal_stability = float(normal_metrics["Normal_Stability_%"].iloc[0])
    specificity = float(normal_metrics["Specificity_%"].iloc[0])
    false_alarm_rate = float(normal_metrics["False_Alarm_Rate_per_Normal_Segment"].iloc[0])

    overall_system = pd.DataFrame([{
        "Positive_Detection_Events": EVENTS,
        "TP": TP,
        "FN": FN,
        "FP": FP,
        "Overall_CSI_%": overall_CSI,
        "Overall_Precision_%": overall_precision,
        "Overall_Recall_Detection_Rate_%": overall_recall,
        "Overall_F1_score_%": overall_f1,
        "Overall_Average_Detection_Delay_s": delays.mean(),
        "Normal_Stability_%": normal_stability,
        "Specificity_%": specificity,
        "False_Alarm_Rate_per_Normal_Segment": false_alarm_rate,
        "Medical_False_Alarms_During_Normal": normal_metrics["Medical_False_Alarms"].iloc[0],
        "Hazard_False_Activations_During_Normal": normal_metrics["Hazard_False_Activations"].iloc[0],
        "Brake_False_Activations_During_Normal": normal_metrics["Brake_False_Activations"].iloc[0],
        "Medical_Hazard_Activation_Success_%": medical_metrics["Hazard_Activation_Success_%"].iloc[0],
        "Medical_Brake_Activation_Success_%": medical_metrics["Brake_Activation_Success_%"].iloc[0],
    }])

    # Summary by class/type
    class_level = pd.concat([
        extract_class_summary(distraction_file, "Distraction"),
        extract_class_summary(drowsiness_file, "Drowsiness + Yawning")
    ], ignore_index=True)

    # Participant level
    participant_level = pd.concat([
        extract_participant_summary(distraction_file, "Distraction"),
        extract_participant_summary(drowsiness_file, "Drowsiness + Yawning"),
        extract_participant_summary(medical_file, "Medical Emergency")
    ], ignore_index=True)

    # Delay statistics
    delay_statistics = pd.DataFrame([{
        "Count": int(delays.count()),
        "Mean_s": delays.mean(),
        "Median_s": delays.median(),
        "Std_s": delays.std(),
        "Min_s": delays.min(),
        "Max_s": delays.max()
    }])

    safety_summary = pd.DataFrame([{
        "Normal_Stability_%": normal_stability,
        "Specificity_%": specificity,
        "False_Alarm_Rate_per_Normal_Segment": false_alarm_rate,
        "Normal_Safety_False_Alarm_Episodes": normal_metrics["Safety_False_Alarm_Episodes"].iloc[0],
        "Medical_False_Alarms_During_Normal": normal_metrics["Medical_False_Alarms"].iloc[0],
        "Hazard_False_Activations_During_Normal": normal_metrics["Hazard_False_Activations"].iloc[0],
        "Brake_False_Activations_During_Normal": normal_metrics["Brake_False_Activations"].iloc[0],
        "Medical_Hazard_Activation_Success_%": medical_metrics["Hazard_Activation_Success_%"].iloc[0],
        "Medical_Brake_Activation_Success_%": medical_metrics["Brake_Activation_Success_%"].iloc[0],
    }])

    # حفظ Excel النهائي
    out_xlsx = OUTPUT_DIR / "final_overall_driver_monitoring_evaluation.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        overall_system.to_excel(writer, sheet_name="Overall_System", index=False)
        module_metrics.to_excel(writer, sheet_name="Performance_By_Module", index=False)
        normal_metrics.to_excel(writer, sheet_name="Normal_Specificity", index=False)
        class_level.to_excel(writer, sheet_name="Class_Level_Performance", index=False)
        participant_level.to_excel(writer, sheet_name="Participant_Level", index=False)
        delay_statistics.to_excel(writer, sheet_name="Delay_Statistics", index=False)
        safety_summary.to_excel(writer, sheet_name="Safety_Summary", index=False)
        all_events.to_excel(writer, sheet_name="All_Event_Detections", index=False)

    # =========================
    # الرسوم
    # =========================

    make_bar(
        ["CSI", "Precision", "Recall", "F1-score"],
        [overall_CSI, overall_precision, overall_recall, overall_f1],
        "Overall Event-Based System Performance",
        "Percentage (%)",
        FIG_DIR / "01_overall_event_based_performance.png",
        ylim=(0, 105)
    )

    make_grouped_bar(
        module_metrics,
        "Module",
        ["Precision_%", "Recall_Detection_Rate_%", "F1_score_%"],
        "Performance by Detection Module",
        "Percentage (%)",
        FIG_DIR / "02_performance_by_module.png"
    )

    make_bar(
        ["Normal Stability", "Specificity"],
        [normal_stability, specificity],
        "Normal Driving Stability and Specificity",
        "Percentage (%)",
        FIG_DIR / "03_normal_stability_specificity.png",
        ylim=(0, 105)
    )

    make_detection_matrix(
        module_metrics[["Module", "TP", "FN", "FP"]],
        FIG_DIR / "04_overall_event_detection_matrix.png"
    )

    make_radar(
        module_metrics,
        FIG_DIR / "05_module_performance_radar.png"
    )

    make_delay_figures(all_events)

    if not class_level.empty and "Class" in class_level.columns:
        class_plot = class_level.copy()
        class_plot["Class_Label"] = class_plot["Module"].astype(str) + " - " + class_plot["Class"].astype(str)

        if "F1_score_%" in class_plot.columns:
            make_bar(
                class_plot["Class_Label"],
                pd.to_numeric(class_plot["F1_score_%"], errors="coerce").fillna(0),
                "F1-score by Detection Class",
                "F1-score (%)",
                FIG_DIR / "08_f1_by_detection_class.png",
                ylim=(0, 105),
                rotation=30
            )

        if "Recall_Detection_Rate_%" in class_plot.columns:
            make_bar(
                class_plot["Class_Label"],
                pd.to_numeric(class_plot["Recall_Detection_Rate_%"], errors="coerce").fillna(0),
                "Detection Rate by Detection Class",
                "Detection Rate (%)",
                FIG_DIR / "09_detection_rate_by_detection_class.png",
                ylim=(0, 105),
                rotation=30
            )

    if not participant_level.empty and "Participant" in participant_level.columns and "F1_score_%" in participant_level.columns:
        p = participant_level.copy()
        p["F1_score_%"] = pd.to_numeric(p["F1_score_%"], errors="coerce")
        p_avg = p.groupby("Participant")["F1_score_%"].mean().reset_index()

        make_bar(
            p_avg["Participant"],
            p_avg["F1_score_%"],
            "Average F1-score by Participant",
            "F1-score (%)",
            FIG_DIR / "10_average_f1_by_participant.png",
            ylim=(0, 105),
            rotation=0
        )

    print("\n========== FINAL OVERALL SYSTEM ==========")
    print(overall_system.to_string(index=False))
    print(f"\nSaved Excel: {out_xlsx}")
    print(f"Saved figures in: {FIG_DIR}")


if __name__ == "__main__":
    main()
