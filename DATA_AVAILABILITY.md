# Data and Code Availability

This public repository provides the reproducibility materials associated with IJIES Manuscript 20264753, entitled:

**“A Real-Time Multimodal Driver Monitoring System with Context-Dependent Rule-Based Fusion for the Detection of Distraction, Drowsiness, and Sudden Medical Conditions.”**

The publicly available materials include:

- anonymized event-level ground-truth annotations;
- system prediction and evaluation data;
- event-level evaluation scripts;
- configuration parameters and decision thresholds;
- documented rule-based fusion and progressive response logic;
- YOLOv8n detector-training configuration;
- detector training history and validation outputs;
- the trained YOLOv8n model checkpoint (`best.pt`);
- software dependencies; and
- supporting reproducibility documentation required to inspect and reproduce the manuscript-specific evaluation procedures.

## Participant Privacy

Raw participant video recordings and personally identifiable information are not publicly released because the recordings contain identifiable participant imagery.

Participant identifiers in the publicly released event-level evaluation data are anonymized as P01–P10.

This privacy restriction applies to identifiable raw recordings and does not prevent public access to the non-identifying event-level annotations, evaluation materials, model checkpoint, configuration files, and documented system logic provided in this repository.

## Scope of the Released Code

This repository is specifically intended to support reproducibility and verification of the methods and experimental results reported in IJIES Manuscript 20264753.

The repository provides the manuscript-specific evaluation code, event-level data, decision thresholds, detector-training materials, trained model checkpoint, and documented fusion and safety-response logic required to inspect the reported methodology and reproduce the supplied evaluation procedures.

The repository does not provide the complete source code of the broader doctoral research platform or the complete vehicle hardware-control application beyond the manuscript-specific logic and reproducibility materials released here.

The fusion and response logic is documented in:

`06_Fusion_Response_Logic/`

The YOLOv8n detector-training configuration, training history, validation outputs, and trained checkpoint are provided in:

`07_Detector_Training/`

Software dependencies are listed in:

`requirements.txt`

## Reproducibility Scope

The repository distinguishes between:

1. the original nine-class YOLOv8n detector-training and validation task;
2. the selected/consolidated visual behavioral categories used by the final real-time implementation;
3. the context-dependent rule-based driver-state fusion and progressive response logic; and
4. the event-level evaluation of the complete multimodal driver-monitoring system.

Raw identifiable participant videos are the principal materials withheld from public release for privacy reasons. The non-identifying materials used to verify the manuscript-specific event-level results are provided publicly in this repository.
