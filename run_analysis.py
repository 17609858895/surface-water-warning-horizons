"""Run the published analysis in dependency order; list stages without training by default."""
from pathlib import Path
import argparse
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STAGES = {
    "prepare": ["00_profile.py", "01_cnemc_panel.py", "02_build_landmarks.py", "02b_obs_windows.py"],
    "core": ["05_forecast_threshold.py", "12_rerun_core.py"],
    "recurrence": ["07_decomposition_dml.py", "08_relapse_window.py", "10_framework_checks.py", "14_relapse_ci_dml.py", "14c_relapse_binned.py", "14d_overlap_trim.py", "24b_rebreach_km.py", "25_relapse_null.py", "25b_relapse_regime.py"],
    "calibration": ["16_conformal_surveillance.py", "23_mondrian.py", "26_conformal_robustness.py", "26b_conformal_frontier.py"],
    "spatial": ["13_spatial_rerun.py", "24_spatial_aggregates.py"],
    "sensitivity": ["15_shap.py", "17_masking.py", "17b_masking_oracle3s.py", "18_seq_models.py", "19_thresholds.py", "21_si_misc.py", "rerun_per_indicator_corrected.py"],
    "reconcile": ["reconcile_evidence.py"],
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=[*STAGES, "all"])
    parser.add_argument("--data", type=Path, help="Path to china_full.csv from Figshare release v4")
    args = parser.parse_args()
    if args.stage is None:
        for stage, scripts in STAGES.items():
            print(stage + ": " + ", ".join(scripts))
        return
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    if args.data:
        if not args.data.is_file():
            parser.error("The supplied data file does not exist")
        env["WATER_QUALITY_DATA"] = str(args.data.resolve())
    (ROOT / "results").mkdir(exist_ok=True)
    selected = STAGES if args.stage == "all" else [args.stage]
    for stage in selected:
        for script in STAGES[stage]:
            print(f"Running {stage}: {script}", flush=True)
            subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts" / script)], cwd=ROOT, env=env, check=True)

if __name__ == "__main__":
    main()
