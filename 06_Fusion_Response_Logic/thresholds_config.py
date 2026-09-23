"""
Threshold configuration for the context-dependent rule-based
driver-monitoring system reported in IJIES Manuscript 20264753.

This file centralizes the principal temporal and vehicle-context
thresholds used by the fusion and response logic.
"""


# ============================================================
# Distraction temporal-confirmation thresholds (seconds)
# ============================================================

DISTRACTION_CONFIRMATION_S = {
    "phone": 1.0,
    "center_console": 1.4,
    "off_road_gaze": 1.2,
    "passenger_talk": 1.5,
    "head_side": 0.8,
    "head_down": 1.0,
}


# ============================================================
# Drowsiness and yawning thresholds
# ============================================================

# Initial temporal confirmation of drowsiness-related evidence
INITIAL_DROWSINESS_CONFIRMATION_S = 0.7

# Eye closure sustained for this duration is treated as a
# prolonged drowsiness episode.
PROLONGED_EYE_CLOSURE_S = 5.0

# Minimum duration for a valid yawning event
YAWN_MIN_DURATION_S = 3.0

# Frequent-yawning condition
FREQUENT_YAWN_COUNT = 5
FREQUENT_YAWN_WINDOW_S = 180.0


# ============================================================
# Vehicle-context thresholds
# ============================================================

# Below this speed the system remains in monitoring mode.
# At or above this speed, safety-response escalation is enabled.
VEHICLE_SPEED_GATE_KMH = 10.0


# ============================================================
# Medical-emergency / unresponsiveness timing
# ============================================================

# Confirmation window used for the initial medical-emergency
# assessment, followed by the shorter subsequent confirmation.
INITIAL_MEDICAL_CONFIRMATION_S = 12.0
SUBSEQUENT_MEDICAL_CONFIRMATION_S = 5.0


# ============================================================
# Emergency communication
# ============================================================

# Maximum waiting period after the emergency condition is latched
# before emergency communication becomes eligible.
EMERGENCY_COMMUNICATION_MAX_WAIT_S = 180.0

# Emergency communication may also become eligible when the
# vehicle is confirmed at or below the near-stop threshold.
VEHICLE_STOP_SPEED_KMH = 3.0
VEHICLE_STOP_CONFIRMATION_S = 5.0


# ============================================================
# Notes
# ============================================================

# Eye closure is intentionally excluded from the generic
# distraction-confirmation dictionary. It is handled through the
# drowsiness / possible-medical-emergency pathway.
#
# Skin temperature is not defined here as an independent
# driver-state trigger. Physiological measurements are handled
# by the physiological-validation logic described in the
# manuscript.
#
# State precedence used by the fusion logic:
#
# Possible Medical Emergency > Drowsiness > Distraction > Normal
