# IJIES-20264753 Driver Monitoring System – Reproducibility Repository

This public repository provides reproducibility materials associated with IJIES Manuscript 20264753:

**“A Real-Time Multimodal Driver Monitoring System with Context-Dependent Rule-Based Fusion for the Detection of Distraction, Drowsiness, and Sudden Medical Conditions”**

The repository contains event-level evaluation data, evaluation scripts, documented fusion and safety-response logic, centralized decision thresholds, YOLOv8n detector-training information, the trained detector checkpoint, detector validation outputs, and software dependencies.

The materials are organized to distinguish between detector-level training/validation and event-level evaluation of the complete multimodal driver-monitoring system.

## 1. Repository Structure

### `01_Distraction`

Contains event-level ground-truth and prediction/evaluation materials for the distraction-monitoring component, together with the corresponding evaluation script.

The final real-time system evaluates the distraction-related behavioral categories adopted by the implemented monitoring pipeline.

### `02_Drowsiness_Yawning`

Contains event-level ground-truth and prediction/evaluation materials for drowsiness and yawning, together with the corresponding evaluation script.

These events are evaluated using the temporal definitions and matching rules documented in the manuscript and repository materials.

### `03_Medical_Emergency`

Contains event-level evaluation materials for the simulated possible-medical-emergency scenarios and the corresponding evaluation script.

The medical-emergency experiments represent controlled simulated events and should not be interpreted as clinical validation.

### `04_Normal_Driving`

Contains normal-driving evaluation materials used to examine false safety-critical activations during normal driving.

The normal-driving evaluation is separated from the positive-event evaluation because event-level true-negative counts are not directly defined in the same way as positive event detections.

### `05_Overall_Evaluation`

Contains the materials and script used to reproduce the overall event-level performance evaluation of the complete driver-monitoring system.

The system-level evaluation includes event-level true positives, false positives, false negatives, precision, recall, F1-score, and detection-delay analysis.

### `06_Fusion_Response_Logic`

Documents the context-dependent rule-based decision-level fusion and progressive safety-response logic used by the final system.

Important files include:

- `fusion_response_logic.py` – documented fusion, state-assignment, and response logic.
- `thresholds_config.py` – centralized temporal and vehicle-context thresholds.

The state-assignment logic explicitly distinguishes the final driver states:

- Normal
- Distraction
- Drowsiness
- Possible Medical Emergency

The documented logic also prevents eye-closure evidence from being assigned through the generic distraction pathway and defines the intended state precedence and response progression.

### `07_Detector_Training`

Contains the reproducibility materials associated with the YOLOv8n visual distraction detector.

Important contents include:

- `README.md` – detector-training and system-level class-selection documentation.
- `args.yaml` – saved YOLO training configuration.
- `results.csv` – recorded training/validation history.
- `weights/best.pt` – trained YOLOv8n model checkpoint.
- `training_results/` – detector-level validation curves and confusion matrices.

The original YOLOv8n detector was trained and validated using nine visual behavior classes. The final real-time driver-monitoring implementation used a reduced set of five selected/consolidated visual behavioral categories relevant to the adopted distraction-monitoring logic.

The original phone-related YOLO classes `phone_chatting_texting` and `phone_talking` were consolidated into a single system-level `phone use` behavior.

The detector-level nine-class results and the final system-level behavioral/event evaluation therefore represent different processing stages and should not be interpreted as identical classification tasks.

## 2. YOLOv8n Training Reproducibility

The detector was initialized from the pretrained `yolov8n.pt` model.

The saved training configuration includes:

- Input image size: 640 × 640
- Batch size: 16
- Maximum configured epochs: 50
- Recorded training run completed at epoch 43
- Early-stopping patience: 15
- Optimizer: AdamW
- Initial learning rate: 0.0007
- Weight decay: 0.0005
- Warm-up epochs: 3
- Random seed: 42
- Deterministic mode: enabled
- Pretrained initialization: enabled

The complete saved configuration is provided in:

`07_Detector_Training/args.yaml`

The recorded training history is provided in:

`07_Detector_Training/results.csv`

The trained checkpoint is provided in:

`07_Detector_Training/weights/best.pt`

Detector-level validation plots and confusion matrices are provided in:

`07_Detector_Training/training_results/`

## 3. Detector-Level and System-Level Evaluation

Two evaluation levels should be distinguished.

### Detector level

The detector-level results describe the original nine-class YOLOv8n visual detector.

The supplied detector outputs include:

- F1-Confidence curve
- Precision-Confidence curve
- Precision-Recall curve
- Recall-Confidence curve
- Confusion matrix
- Normalized confusion matrix

The supplied Precision-Recall output reports an overall mAP@0.5 of approximately 0.993 for the original nine-class detector.

### Complete-system level

The complete driver-monitoring system is evaluated at the event level after system-level behavioral interpretation, temporal confirmation, multimodal processing, vehicle-context reasoning, and context-dependent rule-based decision-level fusion.

Accordingly, detector-level mAP values and complete-system event-level precision, recall, F1-score, and detection-delay values represent different evaluation stages and should not be treated as interchangeable metrics.

## 4. Fusion and Decision Logic

The multimodal system integrates available evidence from:

- YOLOv8n-based visual behavioral detection
- facial/ocular analysis
- head pose and gaze information
- physiological sensing
- vehicle-speed/context information
- temporal confirmation and event history

The general processing sequence is:

YOLO/visual behavioral evidence  
→ facial/ocular evidence  
→ class-specific temporal confirmation  
→ physiological validation when applicable  
→ vehicle-context reasoning  
→ context-dependent rule-based decision-level fusion  
→ final driver-state classification  
→ progressive safety response

The implementation uses predefined rule-based logic rather than online learning.

The corresponding documented logic and centralized thresholds are provided in `06_Fusion_Response_Logic`.

## 5. Reproducing the Event-Level Evaluation

The event-level evaluation scripts are provided with the corresponding evaluation data in the repository directories.

To reproduce a reported evaluation:

1. Install the required Python dependencies.
2. Download or clone this repository.
3. Navigate to the relevant evaluation directory.
4. Inspect the accompanying data and evaluation script.
5. Run the corresponding Python evaluation script using the supplied files.

The evaluation materials are organized by distraction, drowsiness/yawning, possible medical emergency, normal driving, and overall system evaluation so that the reported results can be inspected separately.

## 6. Software Dependencies

The principal Python dependencies are listed in the repository-level:

`requirements.txt`

The repository includes dependencies used by the evaluation and documented system components, including Ultralytics YOLO, OpenCV, MediaPipe, NumPy, Pandas, and PySerial.

Because detector training and real-time deployment were performed in different computational environments, exact hardware-specific installation details may depend on the target platform.

## 7. Random Seed and Deterministic Training

The YOLOv8n training configuration used:

`seed: 42`

with deterministic execution enabled in the saved training configuration.

These settings are preserved in:

`07_Detector_Training/args.yaml`

The event-level evaluation scripts operate on the supplied event data and do not require random train/test resampling to reproduce the reported event-level calculations.

## 8. Model Checkpoint

The trained YOLOv8n checkpoint is publicly provided at:

`07_Detector_Training/weights/best.pt`

The checkpoint retains the original nine-class detector output structure used during detector training and validation.

Selection and consolidation of detector outputs for the final real-time driver-monitoring implementation are documented separately in:

`07_Detector_Training/README.md`

## 9. Data Availability and Ethical Restrictions

The repository provides non-identifying reproducibility materials that can be publicly shared, including:

- event-level annotations/evaluation data,
- evaluation scripts,
- detector-training configuration,
- detector training history,
- detector validation outputs,
- trained model checkpoint,
- fusion and response logic,
- temporal and vehicle-context thresholds, and
- software dependency information.

Raw participant video recordings from the real-vehicle experiments are not publicly released because they contain identifiable participant imagery.

This restriction applies to identifiable raw recordings and does not prevent public access to the non-identifying reproducibility materials included in this repository.

The possible-medical-emergency scenarios were simulated for system evaluation and do not constitute clinical validation.

## 10. Scope of the Repository

This repository is intended to support reproducibility and inspection of the methods and evaluation reported in IJIES Manuscript 20264753.

The repository distinguishes between:

1. the original YOLOv8n detector-training task,
2. the selected/consolidated visual behavioral categories used by the final real-time implementation,
3. the multimodal driver-state fusion logic, and
4. the event-level evaluation of the complete system.

The repository should therefore be interpreted together with the methodological definitions and experimental protocol reported in the manuscript.

## 11. Manuscript

**Manuscript ID:** IJIES 20264753

**Title:**  
*A Real-Time Multimodal Driver Monitoring System with Context-Dependent Rule-Based Fusion for the Detection of Distraction, Drowsiness, and Sudden Medical Conditions*

## 12. Repository Purpose

The purpose of this public repository is to provide the reproducibility materials associated with the reported work and to allow readers and reviewers to inspect the detector configuration, trained checkpoint, event-level evaluation procedures, decision thresholds, and documented fusion/response logic used in the study.
