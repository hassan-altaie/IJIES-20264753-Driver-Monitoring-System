# YOLOv8n Distraction Detector and Training Configuration

This directory documents the YOLOv8n-based visual distraction detector used in the real-time multimodal driver-monitoring system reported in IJIES Manuscript 20264753. It provides information about the original detector-training label space, the subset of behavioral categories adopted in the final real-time system, the training configuration, the trained model checkpoint, and the detector-level validation outputs.

The YOLOv8n detector constitutes one component of the multimodal monitoring framework. Its outputs are not used directly as the final driver-state decision. Instead, relevant visual detections are processed together with facial/ocular information, temporal confirmation, physiological information when applicable, and vehicle-context information through the context-dependent rule-based decision-level fusion logic documented in `06_Fusion_Response_Logic`.

## 1. YOLOv8n Detector and Original Training Classes

The visual distraction detector was developed by fine-tuning a pretrained YOLOv8n model for driver-behavior recognition.

The detector-training dataset contained 3,971 annotated images and nine original YOLO classes:

1. `adjusting_radio_dashboard`
2. `drinking_while_driving`
3. `eating_while_driving`
4. `hands_off_steering`
5. `looking_left`
6. `looking_right`
7. `normal_driving`
8. `phone_chatting_texting`
9. `phone_talking`

The dataset was divided into training, validation, and test subsets using a 70%/15%/15% split with a fixed random seed of 42:

- Training: 2,779 images
- Validation: 596 images
- Test: 596 images

These nine classes represent the original YOLO detector-training and validation label space. They should not be interpreted as the final set of behavioral categories used by the complete real-time driver-monitoring system.

## 2. Selection of YOLO Classes for the Final Real-Time System

Although the YOLOv8n detector was originally trained and validated using nine classes, not all of these classes were retained as separate behavioral categories in the final real-time system.

The final implementation used five visual behavioral categories relevant to the adopted distraction-monitoring logic:

1. `phone use`
2. `head right`
3. `head left`
4. `center-console interaction`
5. `normal`

The original YOLO classes `phone_chatting_texting` and `phone_talking` were consolidated into the single system-level category `phone use`. This consolidation was performed at the system level; the trained YOLO checkpoint itself retains the two original phone-related training classes.

The original left- and right-looking detections were used as visual evidence for the corresponding directional distraction behaviors in the final system. Dashboard/console-related visual detections contributed to the center-console interaction behavior used by the final monitoring logic.

The original YOLO classes:

- `drinking_while_driving`
- `eating_while_driving`
- `hands_off_steering`

were not retained as separate behavioral categories in the final real-time implementation. These classes remain part of the original nine-class detector-training and validation results, but they were excluded from the final five-category behavioral configuration used for the real-vehicle system-level evaluation.

Accordingly, two different classification levels must be distinguished:

1. **Original YOLO detector level:** nine classes used during detector training and validation.
2. **Final real-time system level:** five selected/consolidated visual behavioral categories used by the implemented driver-monitoring pipeline.

This distinction explains why the detector-training outputs contain nine classes while the final real-time system reports a reduced set of visual distraction behaviors.

YOLOv8n was therefore not used only for phone-use detection. It provided visual evidence for multiple distraction-related behaviors selected for the final implementation.

## 3. Complementary Facial/Ocular Analysis

The YOLOv8n detector represents only one part of the visual monitoring subsystem.

Complementary facial and ocular information was obtained using the facial-analysis pathway, including information derived from facial landmarks, eye state, head pose, and gaze.

These complementary visual features were used together with relevant YOLO detections to support the behavioral interpretation required by the final driver-monitoring system.

Eye closure, yawning, and drowsiness were not treated as YOLO distraction classes. They were handled separately through the facial/ocular analysis pathway and the corresponding temporal logic.

This separation is important because the YOLO detector-level classes should not be confused with the drowsiness-related or possible-medical-emergency evidence used by the multimodal fusion layer.

## 4. Training Configuration

The YOLOv8n detector was trained using the following principal configuration:

- Base model: `yolov8n.pt`
- Pretrained initialization: enabled
- Input image size: 640 × 640 pixels
- Batch size: 16
- Maximum epochs: 50
- Early-stopping patience: 15 epochs
- Optimizer: AdamW
- Initial learning rate (`lr0`): 0.0007
- Final learning-rate factor (`lrf`): 0.01
- Weight decay: 0.0005
- Warm-up epochs: 3
- Momentum: 0.937
- Random seed: 42
- Deterministic mode: enabled
- Automatic mixed precision (AMP): enabled
- Data-loading workers: 2
- Validation during training: enabled

The saved training configuration is provided in `args.yaml` so that the detector-training parameters associated with the reported model can be inspected directly.

## 5. Data Augmentation

The training procedure used the augmentation settings recorded in the saved YOLO training configuration.

The principal augmentation settings include:

- HSV hue (`hsv_h`): 0.015
- HSV saturation (`hsv_s`): 0.7
- HSV value (`hsv_v`): 0.4
- Horizontal flip probability (`fliplr`): 0.5
- Translation (`translate`): 0.1
- Scale (`scale`): 0.5
- Mosaic (`mosaic`): 1.0
- Close mosaic during the final 10 epochs (`close_mosaic`): 10
- Vertical flip (`flipud`): 0.0
- MixUp (`mixup`): 0.0
- CutMix (`cutmix`): 0.0

The complete saved configuration should be consulted in `args.yaml` for the full set of training parameters.

## 6. Detector-Level Validation Outputs

The detector-training artifacts include standard Ultralytics YOLO validation outputs for the original nine-class detector.

The available outputs include:

- F1-Confidence curve
- Precision-Confidence curve
- Precision-Recall curve
- Recall-Confidence curve
- Confusion matrix
- Normalized confusion matrix
- Training/validation results, when provided in `results.csv`

The supplied Precision-Recall curve reports an overall mAP@0.5 of approximately 0.993 for the original nine-class detector.

The class-specific AP@0.5 values shown in the supplied Precision-Recall output are approximately:

- `adjusting_radio_dashboard`: 0.975
- `drinking_while_driving`: 0.995
- `eating_while_driving`: 0.995
- `hands_off_steering`: 0.995
- `looking_left`: 0.995
- `looking_right`: 0.995
- `normal_driving`: 0.995
- `phone_chatting_texting`: 0.995
- `phone_talking`: 0.995

These results characterize the original nine-class YOLO detector and should not be interpreted as the event-level performance of the complete multimodal driver-monitoring system.

The complete system was evaluated separately using event-level ground truth and temporal event matching.

## 7. Trained Model Checkpoint

The trained YOLOv8n model checkpoint used for reproducibility is provided as:

`weights/best.pt`

The checkpoint retains the original nine-class detector output structure described in Section 1.

The reduction from the original nine training classes to the five visual behavioral categories used by the final real-time system was performed at the system-logic level and does not alter the original YOLO training label space stored in the trained model.

## 8. Relationship Between YOLO Detection and Final Driver-State Classification

The YOLO detector is a source of visual behavioral evidence and is not the final driver-state classifier.

The overall processing sequence can be summarized as:

YOLO visual detection  
→ selection/mapping of relevant behavioral outputs  
→ complementary facial/ocular analysis  
→ class-specific temporal confirmation  
→ physiological validation when applicable  
→ vehicle-speed and contextual reasoning  
→ context-dependent rule-based decision-level fusion  
→ final driver-state classification  
→ progressive safety response

The final driver-state classes used by the fusion layer are:

- Normal
- Distraction
- Drowsiness
- Possible Medical Emergency

Therefore, the following three concepts should not be treated as equivalent:

- the original nine YOLO training classes,
- the five visual behavioral categories selected for the final real-time implementation,
- the four final driver-state classes produced by the multimodal fusion layer.

The state-assignment and progressive response logic are documented in:

`06_Fusion_Response_Logic/fusion_response_logic.py`

The centralized temporal and vehicle-context thresholds are documented in:

`06_Fusion_Response_Logic/thresholds_config.py`

## 9. Reproducibility Files

This directory contains, or is intended to contain, the detector-related reproducibility materials associated with the reported system:

- `README.md` — detector-training and system-mapping documentation
- `args.yaml` — saved YOLO training configuration
- `results.csv` — training/validation history, when available
- `weights/best.pt` — trained YOLOv8n checkpoint
- F1-Confidence curve
- Precision-Confidence curve
- Precision-Recall curve
- Recall-Confidence curve
- Confusion matrix
- Normalized confusion matrix

These detector materials are complemented by the event-level annotations, predictions, evaluation scripts, and fusion/response implementation provided elsewhere in this repository.

## 10. Detector-Level Versus System-Level Evaluation

The detector-level and complete-system evaluations address different stages of the proposed framework.

The YOLO detector-level results evaluate the original nine-class visual detector.

The final system-level evaluation evaluates the complete multimodal pipeline after behavioral selection/mapping, temporal confirmation, facial/ocular processing, physiological processing, vehicle-context reasoning, and context-dependent decision-level fusion.

Consequently, detector-level mAP values should not be directly compared with the event-level precision, recall, F1-score, or detection-delay values reported for the complete driver-monitoring system.

The system-level evaluation materials are provided separately in the corresponding evaluation directories of this repository.

## 11. Training Environment

The documented YOLOv8n training run was performed using Google Colaboratory with GPU acceleration.

The recorded environment included:

- GPU: NVIDIA Tesla T4
- CUDA available: Yes
- Training device: GPU (CUDA)
- Python: 3.12.13
- PyTorch: 2.10.0+cu128
- CUDA: 12.8

The exact software environment can depend on the Ultralytics and Google Colaboratory runtime versions. The saved training configuration and repository dependency information should therefore be used together when reproducing the detector-training environment.

## 12. Data and Privacy

Raw participant video recordings from the real-vehicle experiments are not publicly released because they contain identifiable participant imagery.

The public repository instead provides non-identifying reproducibility materials, including detector configuration, trained model weights, detector-level validation outputs, event-level annotations and evaluation materials, and the documented fusion/response logic.

The restriction on identifiable raw participant recordings does not apply to the non-identifying reproducibility materials provided in this repository.

## 13. Manuscript Reference

These materials correspond to:

**IJIES Manuscript 20264753**

**A Real-Time Multimodal Driver Monitoring System with Context-Dependent Rule-Based Fusion for the Detection of Distraction, Drowsiness, and Sudden Medical Conditions**
