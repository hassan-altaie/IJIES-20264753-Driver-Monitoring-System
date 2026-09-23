"""
Context-Dependent Rule-Based Fusion and Response Logic
IJIES Manuscript 20264753

Reference implementation of the fixed decision-level fusion,
driver-state assignment, state precedence, progressive response,
emergency latch, and recovery logic described in the manuscript.

The fusion stage is deterministic and rule-based.
No online learning or adaptive weighting is performed here.
"""

from enum import Enum


# ============================================================
# 1. Driver states
# ============================================================

class DriverState(Enum):
    NORMAL = "Normal"
    DISTRACTION = "Distraction"
    DROWSINESS = "Drowsiness"
    POSSIBLE_MEDICAL_EMERGENCY = "Possible Medical Emergency"


# ============================================================
# 2. Response levels
# ============================================================

class ResponseLevel(Enum):
    MONITORING = "Monitoring"
    EARLY_WARNING = "Early Warning"
    STRONG_WARNING = "Strong Warning"
    CRITICAL_WARNING = "Critical Warning"
    EMERGENCY_RESPONSE = "Emergency Response"
    RECOVERY = "Recovery"


# ============================================================
# 3. Fixed parameters reported in Table 1
# ============================================================

# Vehicle-context threshold
VEHICLE_SPEED_GATE_KMH = 10.0

# Class-specific temporal confirmation thresholds (seconds)
DISTRACTION_CONFIRMATION_S = {
    "phone": 1.0,
    "center_console": 1.4,
    "off_road_gaze": 1.2,
    "passenger_talk": 1.5,
    "head_side": 0.8,
    "head_down": 1.0,
}

# Drowsiness parameters
INITIAL_DROWSINESS_CONFIRMATION_S = 0.7
PROLONGED_DROWSINESS_EPISODE_S = 5.0
RECURRENT_DROWSINESS_EPISODES = 3
RECURRENT_DROWSINESS_WINDOW_S = 300.0

# Yawning parameters
MIN_YAWN_DURATION_S = 2.5
FREQUENT_YAWN_COUNT = 5
FREQUENT_YAWN_WINDOW_S = 180.0

# Critical unresponsiveness and recovery
CRITICAL_UNRESPONSIVENESS_S = 5.0
CRITICAL_STATE_RECOVERY_S = 3.0

# Vehicle-stop confirmation
VEHICLE_STOP_SPEED_KMH = 3.0
VEHICLE_STOP_CONFIRMATION_S = 5.0

# Physiological warning thresholds
HR_LOW_BPM = 45
HR_HIGH_BPM = 130
SPO2_LOW_PERCENT = 90
SKIN_TEMP_LOW_C = 28.0
SKIN_TEMP_HIGH_C = 35.0
EARLY_HEALTH_PERSISTENCE_S = 60.0

# Sensor status
SENSOR_NOT_WORN_CONFIRMATION_S = 8.0

# Visual thresholds
EAR_CLOSURE_THRESHOLD = 0.205
EAR_CLOSURE_FRAMES = 12
MAR_THRESHOLD = 0.060
DETECTOR_CONFIDENCE_THRESHOLD = 0.30

# Yawn-profile parameters
YAWN_ONSET_MAR = 0.30
YAWN_PEAK_MAR = 0.65
YAWN_RELEASE_MAR = 0.10
YAWN_RELEASE_HOLD_S = 0.40
MIN_INTERVAL_BETWEEN_YAWNS_S = 2.5

# Progressive response timing
PRE_ALERT_DURATION_S = 5.0
STAGE_1_DURATION_S = 2.0
STAGE_2_DURATION_S = 2.0

# Emergency communication
EMERGENCY_COMMUNICATION_MAX_WAIT_S = 180.0


# ============================================================
# 4. Physiological-support logic
# ============================================================

def physiological_status(
    sensor_valid,
    heart_rate=None,
    spo2=None,
    skin_temperature=None,
):
    """
    Evaluate physiological information.

    HR and SpO2 provide the primary physiological warning evidence.

    Skin temperature is treated as corroborating/supporting
    information and does not independently generate an Early
    Health Warning.

    Missing or invalid sensor measurements are kept separate
    from genuine physiological abnormality.
    """

    if not sensor_valid:
        return {
            "status": "Sensor Not Worn / Invalid",
            "primary_abnormal": False,
            "temperature_support": False,
        }

    hr_abnormal = (
        heart_rate is not None
        and (heart_rate < HR_LOW_BPM or heart_rate > HR_HIGH_BPM)
    )

    spo2_abnormal = (
        spo2 is not None
        and spo2 < SPO2_LOW_PERCENT
    )

    temperature_support = (
        skin_temperature is not None
        and (
            skin_temperature < SKIN_TEMP_LOW_C
            or skin_temperature > SKIN_TEMP_HIGH_C
        )
    )

    primary_abnormal = hr_abnormal or spo2_abnormal

    return {
        "status": "Abnormal Physiology" if primary_abnormal else "Normal",
        "primary_abnormal": primary_abnormal,
        "temperature_support": temperature_support,
    }


def early_health_warning(primary_abnormal, abnormal_duration_s):
    """
    Independent Early Health Warning pathway.

    A valid primary physiological abnormality must persist
    for at least 60 s.

    This warning does not itself classify the driver as a
    Possible Medical Emergency.
    """

    return (
        primary_abnormal
        and abnormal_duration_s >= EARLY_HEALTH_PERSISTENCE_S
    )


# ============================================================
# 5. Driver-state assignment
# ============================================================

def classify_driver_state(
    confirmed_distraction=False,
    drowsiness_alert=False,
    frequent_yawn_alert=False,
    critical_unresponsiveness=False,
):
    """
    Assign the final driver state using explicit precedence.

    Precedence:
        Possible Medical Emergency
        > Drowsiness
        > Distraction
        > Normal

    IMPORTANT:
    Eye closure is NOT included in the generic distraction set.

    Eye closure is processed through the drowsiness pathway.
    Persistent unresponsiveness after the high-alert stage is
    processed through the Possible Medical Emergency pathway.

    Physiological information supports health assessment and
    emergency-category reporting, but does not gate the
    behavioral emergency decision.
    """

    if critical_unresponsiveness:
        return DriverState.POSSIBLE_MEDICAL_EMERGENCY

    if drowsiness_alert or frequent_yawn_alert:
        return DriverState.DROWSINESS

    if confirmed_distraction:
        return DriverState.DISTRACTION

    return DriverState.NORMAL


# ============================================================
# 6. Drowsiness and yawning criteria
# ============================================================

def recurrent_drowsiness_alert(episode_count, window_s):
    """
    Recurrent drowsiness criterion:
    3 episodes within 300 s.
    """

    return (
        episode_count >= RECURRENT_DROWSINESS_EPISODES
        and window_s <= RECURRENT_DROWSINESS_WINDOW_S
    )


def frequent_yawning_alert(yawn_count, window_s):
    """
    Frequent-yawning criterion:
    5 yawns within 180 s.
    """

    return (
        yawn_count >= FREQUENT_YAWN_COUNT
        and window_s <= FREQUENT_YAWN_WINDOW_S
    )


def prolonged_drowsiness(eye_closure_duration_s):
    """
    A prolonged drowsiness episode is confirmed when
    eye closure persists for at least 5.0 s.
    """

    return eye_closure_duration_s >= PROLONGED_DROWSINESS_EPISODE_S


# ============================================================
# 7. Progressive response logic
# ============================================================

def determine_response(
    driver_state,
    vehicle_speed_kmh,
    persistence_s=0.0,
    high_alert_unresponsive_s=0.0,
    emergency_latched=False,
    evidence_clear_s=0.0,
    vehicle_stopped_s=0.0,
):
    """
    Determine the progressive safety-response level.

    Visual and physiological monitoring remain active at all
    vehicle speeds.

    Below 10 km/h, ordinary distraction/drowsiness events remain
    under monitoring without active safety-response escalation.

    At or above 10 km/h, confirmed driver-risk events can progress
    through the warning hierarchy.

    A latched emergency remains active while the vehicle slows.
    Recovery requires:
        - abnormal evidence clear for >= 3.0 s, and
        - vehicle speed < 3 km/h for >= 5.0 s.
    """

    # --------------------------------------------------------
    # Emergency latch and recovery
    # --------------------------------------------------------

    if emergency_latched:

        recovery_ready = (
            evidence_clear_s >= CRITICAL_STATE_RECOVERY_S
            and vehicle_speed_kmh < VEHICLE_STOP_SPEED_KMH
            and vehicle_stopped_s >= VEHICLE_STOP_CONFIRMATION_S
        )

        if recovery_ready:
            return {
                "state": DriverState.NORMAL.value,
                "response": ResponseLevel.RECOVERY.value,
                "escalation_enabled": False,
                "emergency_latched": False,
                "hazard": False,
                "brake_request": False,
            }

        return {
            "state": DriverState.POSSIBLE_MEDICAL_EMERGENCY.value,
            "response": ResponseLevel.EMERGENCY_RESPONSE.value,
            "escalation_enabled": True,
            "emergency_latched": True,
            "hazard": True,
            "brake_request": True,
        }

    # --------------------------------------------------------
    # Normal state
    # --------------------------------------------------------

    if driver_state == DriverState.NORMAL:
        return {
            "state": driver_state.value,
            "response": ResponseLevel.MONITORING.value,
            "escalation_enabled": False,
            "emergency_latched": False,
            "hazard": False,
            "brake_request": False,
        }

    # --------------------------------------------------------
    # Speed-aware escalation
    # --------------------------------------------------------

    escalation_enabled = vehicle_speed_kmh >= VEHICLE_SPEED_GATE_KMH

    if not escalation_enabled:
        return {
            "state": driver_state.value,
            "response": ResponseLevel.MONITORING.value,
            "escalation_enabled": False,
            "emergency_latched": False,
            "hazard": False,
            "brake_request": False,
        }

    # --------------------------------------------------------
    # Possible Medical Emergency
    # --------------------------------------------------------

    if (
        driver_state == DriverState.POSSIBLE_MEDICAL_EMERGENCY
        or high_alert_unresponsive_s >= CRITICAL_UNRESPONSIVENESS_S
    ):
        return {
            "state": DriverState.POSSIBLE_MEDICAL_EMERGENCY.value,
            "response": ResponseLevel.EMERGENCY_RESPONSE.value,
            "escalation_enabled": True,
            "emergency_latched": True,
            "hazard": True,
            "brake_request": True,
        }

    # --------------------------------------------------------
    # Progressive warning stages
    # --------------------------------------------------------

    if persistence_s < PRE_ALERT_DURATION_S:
        response = ResponseLevel.EARLY_WARNING

    elif persistence_s < (
        PRE_ALERT_DURATION_S + STAGE_1_DURATION_S
    ):
        response = ResponseLevel.STRONG_WARNING

    else:
        response = ResponseLevel.CRITICAL_WARNING

    return {
        "state": driver_state.value,
        "response": response.value,
        "escalation_enabled": True,
        "emergency_latched": False,
        "hazard": False,
        "brake_request": False,
    }


# ============================================================
# 8. Emergency communication condition
# ============================================================

def emergency_communication_ready(
    emergency_latched,
    emergency_duration_s,
    vehicle_speed_kmh,
    vehicle_stopped_s,
):
    """
    Emergency communication becomes eligible when an emergency
    is latched and either:

    1. the maximum waiting period of 180 s is reached, or
    2. the vehicle is confirmed stopped below 3 km/h for 5 s.

    The external SIM868 subsystem performs the SMS/GPS and
    voice-call actions.
    """

    if not emergency_latched:
        return False

    timeout_reached = (
        emergency_duration_s >= EMERGENCY_COMMUNICATION_MAX_WAIT_S
    )

    stop_confirmed = (
        vehicle_speed_kmh < VEHICLE_STOP_SPEED_KMH
        and vehicle_stopped_s >= VEHICLE_STOP_CONFIRMATION_S
    )

    return timeout_reached or stop_confirmed
