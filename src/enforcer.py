from __future__ import annotations

import argparse
import copy
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import joblib
import numpy as np

# Logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(name)-18s │ %(levelname)-7s │ %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("enforcer")


# Configuration
@dataclass
class EnforcerConfig:
    """All tuneable parameters live here.

    Attributes
    ----------
    lambda_decay : float
        Decay constant λ for the exponential  s(t) = s₀·e^{-λt}.
        Larger → faster attenuation.  Reasonable range: 1e-4 … 5e-3.
    floor_fraction : float
        Minimum allowed signal level expressed as a fraction of the
        initial value s₀.  The enforced signal will never drop below
        s₀ * floor_fraction.  Set to 0.0 for no floor (not recommended
        on real hardware).
    spike_z_threshold : float
        How many standard deviations above the rolling mean counts as
        a *spike* worth logging in the monitoring report.
    spike_window : int
        Number of samples in the rolling window used to detect spikes.
    counterfactual : bool
        If True, the controller records both the original (un-enforced)
        and the enforced signal for every event so that downstream
        analysis can compare them.
    output_dir : str
        Directory where enforcement results are written.
    """

    lambda_decay: float = 0.001
    floor_fraction: float = 0.05
    spike_z_threshold: float = 3.0
    spike_window: int = 30
    counterfactual: bool = True
    output_dir: str = "/kaggle/working/enforcer_outputs"


# Attenuation schedule (pure math, no side-effects)
@dataclass
class AttenuationSchedule:
    """Pre-computed per-sample multiplier array for one alarm event.

    Attributes
    ----------
    multipliers : np.ndarray
        1-D float64 array of length `n_samples`.
        multipliers[t] ∈ [floor_fraction, 1.0].
    alarm_segment : int
        Sub-segment where the alarm fired (0-indexed).
    remaining_horizon : int
        Number of sub-segments remaining after the alarm.
    remaining_seconds : float
        Wall-clock seconds corresponding to `remaining_horizon`.
    lambda_decay : float
        Decay constant used to build this schedule.
    floor_fraction : float
        Floor fraction used to clamp the decay.
    """

    multipliers: np.ndarray
    alarm_segment: int
    remaining_horizon: int
    remaining_seconds: float
    lambda_decay: float
    floor_fraction: float


class EnforcerPolicy:
    """Given an alarm event, compute the attenuation schedule.

    The schedule is an array of per-sample multipliers in [floor, 1.0]
    that the SignalController applies element-wise to the raw signal.

    The decay model is **exponential with a floor**:

        m(t) = max( e^{-λt} , floor_fraction )

    where t is the sample index *within the enforcement window*
    (starts at 0 when the alarm fires) and λ = `config.lambda_decay`.
    """

    def __init__(self, config: EnforcerConfig | None = None):
        self.cfg = config or EnforcerConfig()

    # public

    def compute_schedule(
        self,
        event: dict[str, Any],
        alpha: int,
        sampling_rate: int,
        segments: int,
    ) -> AttenuationSchedule:
        # Builds an AttenuationSchedule for a single alarm event.
        alarm_seg = event["alarm_segment"]
        remaining_horizon = event["remaining_horizon"]
        remaining_seconds = event["remaining_seconds"]

        # Number of raw samples the schedule must cover.
        # We attenuate from the alarm point to the end of the α-window.
        n_enforcement_samples = remaining_horizon * (alpha // segments)
        # Fall back to at least 1 sample so shapes stay valid.
        n_enforcement_samples = max(n_enforcement_samples, 1)

        t = np.arange(n_enforcement_samples, dtype=np.float64)
        raw_decay = np.exp(-self.cfg.lambda_decay * t)
        multipliers = np.clip(raw_decay, self.cfg.floor_fraction, 1.0)

        return AttenuationSchedule(
            multipliers=multipliers,
            alarm_segment=alarm_seg,
            remaining_horizon=remaining_horizon,
            remaining_seconds=remaining_seconds,
            lambda_decay=self.cfg.lambda_decay,
            floor_fraction=self.cfg.floor_fraction,
        )

    def compute_schedules(
        self,
        events: list[dict[str, Any]],
        alpha: int,
        sampling_rate: int,
        segments: int,
    ) -> list[AttenuationSchedule]:
        """Vectorised convenience: compute schedules for all events."""
        return [
            self.compute_schedule(e, alpha, sampling_rate, segments)
            for e in events
        ]


# Signal controller (applies schedule, monitors, logs)
@dataclass
class EnforcementResult:
    # Container for one enforced alarm event.
    machine_id: str
    window_idx: int
    alarm_segment: int
    true_label: int

    # per-signal results  {signal_name -> dict}
    signals: dict[str, dict] = field(default_factory=dict)

    # aggregate stats
    n_spikes_detected: int = 0
    peak_attenuation: float = 1.0          # lowest multiplier applied
    enforcement_duration_seconds: float = 0.0

    # counterfactual delta  (∑ |original - enforced| across all signals)
    counterfactual_total_delta: float = 0.0


class SignalController:
    # Applies an AttenuationSchedule to the raw sensor signals.

    def __init__(self, config: EnforcerConfig | None = None):
        self.cfg = config or EnforcerConfig()

    # spike detection
    def _detect_spikes(self, signal: np.ndarray) -> list[dict]:
        # Simple z-score spike detector over a rolling window.
        if len(signal) < self.cfg.spike_window:
            return []

        spikes: list[dict] = []
        window = self.cfg.spike_window

        for i in range(window, len(signal)):
            seg = signal[i - window: i]
            mu = seg.mean()
            sigma = seg.std()
            if sigma < 1e-12:
                continue
            z = (signal[i] - mu) / sigma
            if abs(z) >= self.cfg.spike_z_threshold:
                spikes.append({
                    "index": int(i),
                    "value": float(signal[i]),
                    "z_score": float(z),
                })
        return spikes

    # core enforcement

    def apply(
        self,
        event: dict[str, Any],
        schedule: AttenuationSchedule,
        primary_signals: list[str],
        alpha: int,
        segments: int,
    ) -> EnforcementResult:
        # Apply attenuation to every primary signal in `event`.

        raw_signals: dict[str, np.ndarray] = event.get("raw_signals", {})
        alarm_seg = schedule.alarm_segment
        seg_len = alpha // segments           # samples per sub-segment
        enforce_start = alarm_seg * seg_len   # sample index where we begin

        result = EnforcementResult(
            machine_id=event.get("machine_id", "unknown"),
            window_idx=event.get("window_idx", -1),
            alarm_segment=alarm_seg,
            true_label=event.get("true_label", -1),
            peak_attenuation=float(schedule.multipliers.min()),
            enforcement_duration_seconds=float(
                len(schedule.multipliers) * 4  # sampling_rate assumed 4 s
            ),
        )

        total_delta = 0.0
        total_spikes = 0

        for sig_name in primary_signals:
            original = raw_signals.get(sig_name)
            if original is None or len(original) == 0:
                continue

            original = np.asarray(original, dtype=np.float64)

            # The portion of the signal we are attenuating.
            enforce_end = min(
                enforce_start + len(schedule.multipliers),
                len(original),
            )
            n_apply = enforce_end - enforce_start

            if n_apply <= 0:
                continue

            mults = schedule.multipliers[:n_apply]

            # build enforced signal
            enforced = original.copy()

            # s₀ for this signal = value at the alarm point
            s0 = original[enforce_start] if enforce_start < len(original) else original[-1]
            floor_value = abs(s0) * self.cfg.floor_fraction

            enforced[enforce_start:enforce_end] = original[enforce_start:enforce_end] * mults

            # Clamp: never below floor (handles both positive & negative s₀)
            if s0 >= 0:
                enforced[enforce_start:enforce_end] = np.maximum(
                    enforced[enforce_start:enforce_end], floor_value
                )
            else:
                enforced[enforce_start:enforce_end] = np.minimum(
                    enforced[enforce_start:enforce_end], -floor_value
                )

            # spike detection on original
            spikes = self._detect_spikes(original)
            total_spikes += len(spikes)

            # counterfactual delta
            delta = np.sum(np.abs(original - enforced))
            total_delta += delta

            sig_result: dict[str, Any] = {
                "enforced": enforced,
                "s0": float(s0),
                "floor_value": float(floor_value),
                "n_samples_attenuated": int(n_apply),
                "spikes_in_original": spikes,
                "counterfactual_delta": float(delta),
            }
            if self.cfg.counterfactual:
                sig_result["original"] = original.copy()

            result.signals[sig_name] = sig_result

        result.n_spikes_detected = total_spikes
        result.counterfactual_total_delta = total_delta
        return result


# Top-level runner
def run_enforcer(
    payload_path: str,
    config: EnforcerConfig | None = None,
) -> list[EnforcementResult]:
    # End-to-end enforcement pipeline.
    cfg = config or EnforcerConfig()

    # Load
    log.info("Loading enforcer payload from %s", payload_path)
    payload = joblib.load(payload_path)

    alarm_events:    list[dict] = payload["alarm_events"]
    threshold:       float      = payload["threshold"]
    alpha:           int        = payload["alpha"]
    sampling_rate:   int        = payload["sampling_rate"]
    segments:        int        = payload["segments"]
    primary_signals: list[str]  = payload["primary_signals"]

    log.info(
        "Loaded %d alarm events  (α=%d samples, %d segments, threshold=%.6f)",
        len(alarm_events), alpha, segments, threshold,
    )

    if not alarm_events:
        log.warning("No alarm events to enforce – exiting.")
        return []

    # Schedule
    policy = EnforcerPolicy(cfg)
    schedules = policy.compute_schedules(
        alarm_events, alpha, sampling_rate, segments,
    )
    log.info("Computed %d attenuation schedules", len(schedules))

    # Apply
    controller = SignalController(cfg)
    results: list[EnforcementResult] = []

    for event, sched in zip(alarm_events, schedules):
        res = controller.apply(event, sched, primary_signals, alpha, segments)
        results.append(res)

    log.info("Enforcement applied to %d events", len(results))

    # Report & persist
    _print_summary(results, cfg)
    _save_results(results, cfg)

    return results

# Reporting helpers
def _print_summary(results: list[EnforcementResult], cfg: EnforcerConfig):
    # Pretty-print a compact enforcement report to the console.
    n = len(results)
    tp = sum(1 for r in results if r.true_label == 1)
    fp = sum(1 for r in results if r.true_label == 0)
    total_spikes = sum(r.n_spikes_detected for r in results)
    total_delta = sum(r.counterfactual_total_delta for r in results)
    mean_atten = float(np.mean([r.peak_attenuation for r in results])) if results else 1.0

    border = "═" * 56
    print(f"\n╔{border}╗")
    print(f"║{'ENFORCER SUMMARY':^56}║")
    print(f"╠{border}╣")
    print(f"║  Events enforced           : {n:<25} ║")
    print(f"║    ├─ True-positive alarms  : {tp:<25} ║")
    print(f"║    └─ False-positive alarms : {fp:<25} ║")
    print(f"║  Decay constant (λ)        : {cfg.lambda_decay:<25.6f} ║")
    print(f"║  Floor fraction            : {cfg.floor_fraction:<25.4f} ║")
    print(f"║  Mean peak attenuation     : {mean_atten:<25.6f} ║")
    print(f"║  Total spikes detected     : {total_spikes:<25} ║")
    print(f"║  Counterfactual Σ|Δ|       : {total_delta:<25.2f} ║")
    print(f"╚{border}╝\n")

    # Per-event detail
    print(f"{'#':>3}  {'Machine':<14} {'Win':>4} {'Seg':>4} {'Label':>6}"
          f"  {'Spikes':>6}  {'PeakAtt':>8}  {'Δ':>12}")
    print("─" * 70)
    for i, r in enumerate(results):
        print(
            f"{i:3d}  {r.machine_id:<14} {r.window_idx:4d} {r.alarm_segment:4d}"
            f"  {'TP' if r.true_label == 1 else 'FP':>5}"
            f"  {r.n_spikes_detected:6d}"
            f"  {r.peak_attenuation:8.5f}"
            f"  {r.counterfactual_total_delta:12.2f}"
        )
    print()


def _save_results(results: list[EnforcementResult], cfg: EnforcerConfig):
    # Persist enforcement results for downstream analysis.
    os.makedirs(cfg.output_dir, exist_ok=True)

    # pkl (full objects)
    pkl_path = os.path.join(cfg.output_dir, "enforcement_results.pkl")
    joblib.dump(results, pkl_path)
    log.info("Full results saved → %s", pkl_path)

    # json (summary)
    summary: list[dict] = []
    for r in results:
        entry = {
            "machine_id": r.machine_id,
            "window_idx": r.window_idx,
            "alarm_segment": r.alarm_segment,
            "true_label": r.true_label,
            "n_spikes_detected": r.n_spikes_detected,
            "peak_attenuation": round(r.peak_attenuation, 6),
            "enforcement_duration_seconds": round(r.enforcement_duration_seconds, 1),
            "counterfactual_total_delta": round(r.counterfactual_total_delta, 4),
            "signals": {},
        }
        for sig_name, sig_data in r.signals.items():
            entry["signals"][sig_name] = {
                "s0": round(sig_data["s0"], 6),
                "floor_value": round(sig_data["floor_value"], 6),
                "n_samples_attenuated": sig_data["n_samples_attenuated"],
                "n_spikes_in_original": len(sig_data["spikes_in_original"]),
                "counterfactual_delta": round(sig_data["counterfactual_delta"], 4),
            }
        summary.append(entry)

    json_path = os.path.join(cfg.output_dir, "enforcement_report.json")
    with open(json_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    log.info("JSON report saved  → %s", json_path)


# CLI
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Enforcer – gradual signal attenuation for pre-anomaly alarms",
    )
    p.add_argument(
        "--payload", type=str,
        default="/kaggle/working/enforcer_inputs/alarm_events.pkl",
        help="Path to alarm_events.pkl produced by optimized.py.",
    )
    p.add_argument(
        "--lambda-decay", type=float, default=0.001,
        help="Exponential decay constant λ  (default: 0.001).",
    )
    p.add_argument(
        "--floor-frac", type=float, default=0.05,
        help="Minimum signal fraction  (default: 0.05 = 5%%).",
    )
    p.add_argument(
        "--spike-z", type=float, default=3.0,
        help="Z-score threshold for spike detection  (default: 3.0).",
    )
    p.add_argument(
        "--no-counterfactual", action="store_true",
        help="Disable counterfactual logging to save memory.",
    )
    p.add_argument(
        "--output-dir", type=str,
        default="/kaggle/working/enforcer_outputs",
        help="Directory for enforcement outputs.",
    )
    return p.parse_args()


def main():
    args = _parse_args()
    cfg = EnforcerConfig(
        lambda_decay=args.lambda_decay,
        floor_fraction=args.floor_frac,
        spike_z_threshold=args.spike_z,
        counterfactual=not args.no_counterfactual,
        output_dir=args.output_dir,
    )
    run_enforcer(args.payload, cfg)


if __name__ == "__main__":
    main()
