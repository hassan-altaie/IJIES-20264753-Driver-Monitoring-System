# Fusion and Response Logic

This directory documents the context-dependent rule-based decision-level fusion and progressive response logic used in the real-time driver-monitoring system reported in IJIES Manuscript 20264753.

The fusion layer integrates confirmed visual/behavioral events, physiological information, vehicle-speed context, and temporal history. The decision process is predefined and rule-based; no online learning is performed in the fusion stage.

## Driver-State Logic

The system distinguishes the following driver states:

- Normal
- Distraction
- Drowsiness
- Possible Medical Emergency

Distraction behaviors are evaluated using class-specific temporal confirmation thresholds. Eye-closure and drowsiness evidence are processed separately from distraction behaviors to avoid ambiguous state assignment.

When multiple conditions are simultaneously satisfied, the higher-severity state takes precedence according to the following hierarchy:

Possible Medical Emergency > Drowsiness > Distraction > Normal

## Context-Dependent Decision Process

The decision process includes:

1. Temporal confirmation of detected behaviors.
2. Physiological validation and health-warning assessment.
3. Vehicle-speed-aware decision logic.
4. Driver-state classification.
5. Progressive warning and emergency-response logic.
6. Recovery and hysteresis to prevent unstable state switching.

Vehicle speed is used as contextual information for escalation. The safety-response escalation is enabled when the vehicle speed is at or above 10 km/h.

## Response Logic

Depending on the confirmed driver state and persistence of the condition, the system progresses through monitoring and warning stages. If persistent unresponsiveness or a possible medical emergency is confirmed, the emergency-response logic can activate the external hazard warning and brake-assistance request according to the predefined safety rules.

Emergency communication is handled separately by the SIM868-based communication subsystem according to the system response conditions.

## Reproducibility

The files in this directory provide the explicit thresholds, state-assignment rules, precedence logic, and response logic needed to reproduce the decision-level fusion described in the manuscript.

The YOLOv8n distraction detector and its training configuration are documented separately in the detector/training directory.
