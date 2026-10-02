"""Matched-window MSE diagnostic for Step 10E.

Compares three MSE variants under matched conditions (same model checkpoint,
same initial states/episodes, same channel realizations, same SNR):

    MSE_initial — single-step MSE at t=0, all episodes active
                  (matches the structural definition of recon_mse_diag)
    MSE_full    — trajectory mean over t=0..T-1, all episodes
                  (same formula as aggregate_task.mse_mean in eval JSONs)
    MSE_active  — trajectory mean, non-exited episodes only
                  (quantifies the post-exit zero-error bypass contribution)

BLOCKER POLICY
--------------
If torch.load(checkpoint_path) raises any exception (missing file,
architecture mismatch, CUDA unavailable), the script prints 'BLOCKER: ...'
to stdout and exits with code 2. Do not install packages or fabricate
results. Required environment: Python venv with PyTorch matching the
training environment.

Usage:
    .venv/Scripts/python.exe diagnostics/matched_mse_diagnostic.py [options]

Options:
    --checkpoint PATH   path to final_ckpt.pt (default: auto-discover under
                        results/M1/M1_task_awgn_k1_seed42__*/final_ckpt.pt)
    --k INT             channel uses (default: 1)
    --channel STR       awgn or rayleigh (default: awgn)
    --snr_db FLOAT      SNR in dB (default: 0.0)
    --n_episodes INT    episode count for diagnostic (default: 500)
    --seed INT          episode/noise generation seed (default: config.TEST_SEED)
    --t_max INT         episode horizon (default: config.T_MAX)
    --train_report PATH path to train_report.json to show recon_mse_diag_final
                        (default: same directory as checkpoint)
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import pathlib
import sys

# ---------------------------------------------------------------------------
# Resolve workspace root and inject into sys.path so project imports work
# ---------------------------------------------------------------------------
WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    import torch
except ImportError as e:
    print(f"BLOCKER: cannot import torch — {e}")
    print("Required: activate the project venv (.venv) which contains PyTorch.")
    sys.exit(2)

try:
    from project import config
    from project.data.normalization import normalize
    from project.evaluation.evaluate import (
        Condition,
        _neural_estimate_factory,
        generate_realizations,
        run_method,
    )
    from project.evaluation.metrics import aggregate_episode_metrics
    from project.models.reconstruction_jscc import ReconstructionDeepJSCC
except ImportError as e:
    print(f"BLOCKER: cannot import project modules — {e}")
    print("Required: run from the workspace root with the project venv active.")
    sys.exit(2)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Matched-window MSE diagnostic (Step 10E, read-only)"
    )
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to final_ckpt.pt")
    p.add_argument("--k", type=int, default=1,
                   help="Number of channel uses (default: 1)")
    p.add_argument("--channel", type=str, default="awgn",
                   choices=["awgn", "rayleigh"],
                   help="Channel type (default: awgn)")
    p.add_argument("--snr_db", type=float, default=0.0,
                   help="SNR in dB (default: 0.0)")
    p.add_argument("--n_episodes", type=int, default=config.N_TEST_EPISODES_SMOKE,
                   help=f"Episodes for diagnostic (default: {config.N_TEST_EPISODES_SMOKE})")
    p.add_argument("--seed", type=int, default=config.TEST_SEED,
                   help=f"RNG seed (default: {config.TEST_SEED})")
    p.add_argument("--t_max", type=int, default=config.T_MAX,
                   help=f"Episode horizon (default: {config.T_MAX})")
    p.add_argument("--train_report", type=str, default=None,
                   help="Path to train_report.json (default: same dir as checkpoint)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_checkpoint(k: int, channel: str, seed: int) -> pathlib.Path:
    """Auto-discover the task checkpoint for the given configuration."""
    pattern = str(WORKSPACE_ROOT / "results" / "M1" /
                  f"M1_task_{channel}_k{k}_seed{seed}__*" / "final_ckpt.pt")
    matches = glob.glob(pattern)
    if not matches:
        # Try the fetch_m1b mirror
        pattern2 = str(WORKSPACE_ROOT / "fetch_m1b" / "M1" /
                       f"M1_task_{channel}_k{k}_seed{seed}__*" / "final_ckpt.pt")
        matches = glob.glob(pattern2)
    if not matches:
        return None
    return pathlib.Path(sorted(matches)[0])


# ---------------------------------------------------------------------------
# Main diagnostic
# ---------------------------------------------------------------------------

def run_diagnostic(args: argparse.Namespace) -> None:
    k = args.k
    channel = args.channel
    snr_db = args.snr_db
    n_episodes = args.n_episodes
    seed = args.seed
    t_max = args.t_max

    # ---- Resolve checkpoint path ----
    if args.checkpoint is not None:
        ckpt_path = pathlib.Path(args.checkpoint)
    else:
        ckpt_path = find_checkpoint(k, channel, seed=42)  # seed42 is the primary
        if ckpt_path is None:
            print(f"BLOCKER: no checkpoint found for k={k}, channel={channel}, seed=42")
            print("Pattern searched: results/M1/M1_task_{channel}_k{k}_seed42__*/final_ckpt.pt")
            print("Required: M1 training must be complete.")
            sys.exit(2)

    print("=" * 72)
    print("=== Matched-Window MSE Diagnostic (Step 10E) ===")
    print("=" * 72)
    print(f"Checkpoint : {ckpt_path}")
    print(f"k={k}  channel={channel}  snr_db={snr_db}  "
          f"n_episodes={n_episodes}  seed={seed}  t_max={t_max}")
    print()

    if not ckpt_path.exists():
        print(f"BLOCKER: checkpoint file not found: {ckpt_path}")
        print("Required: M1 training must be complete and checkpoint present.")
        sys.exit(2)

    # ---- Load train_report for recon_mse_diag reference ----
    train_report_path = (pathlib.Path(args.train_report)
                         if args.train_report
                         else ckpt_path.parent / "train_report.json")
    recon_mse_diag_final = float("nan")
    if train_report_path.exists():
        try:
            with open(train_report_path) as f:
                tr = json.load(f)
            recon_mse_diag_final = tr.get("recon_mse_diag_final", float("nan"))
            print(f"train_report.json found: recon_mse_diag_final = {recon_mse_diag_final:.4f}")
        except Exception as e:
            print(f"WARNING: could not read train_report.json: {e}")
    else:
        print(f"WARNING: train_report.json not found at {train_report_path}")
    print()

    # ---- Load model checkpoint ----
    # The checkpoint format is a wrapped dict with keys:
    #   "format_version", "step", "seed", "metadata", "history",
    #   "model_state", "torch_rng_state"
    # (see project/training/common.py:save_checkpoint)
    try:
        model = ReconstructionDeepJSCC(k=k)
        payload = torch.load(str(ckpt_path), map_location="cpu",
                             weights_only=False)
        # Support both bare state_dict and the wrapped format
        if isinstance(payload, dict) and "model_state" in payload:
            state_dict = payload["model_state"]
        else:
            state_dict = payload   # bare state_dict (fallback)
        model.load_state_dict(state_dict)
        model.eval()
        print("Checkpoint loaded successfully.")
    except Exception as e:
        print(f"BLOCKER: checkpoint load failed: {e}")
        print("Required: PyTorch with matching architecture; run from venv.")
        sys.exit(2)

    # ---- Generate matched realizations (same as evaluation harness) ----
    cond = Condition(k=k, channel=channel, snr_db=snr_db,
                     t_max=t_max, episodes=n_episodes, seed=seed)
    real = generate_realizations(cond, n_episodes)
    print(f"Realizations generated: {n_episodes} episodes × {t_max} steps × "
          f"k={k} (seed={seed}, channel={channel}, snr_db={snr_db} dB).")
    print()

    # ================================================================
    # MSE_initial — single-step MSE at t=0, all episodes active
    # Definition: identical structure to recon_mse_diag in train_task_oriented.py
    #   recon_mse_diag = ((s_bar_hat - normalize(s0))^2).sum(dim=-1).mean() / 6.0
    # Here we use the paired noise realization for t=0 from the eval harness.
    # ================================================================
    print("--- MSE_initial (single step, t=0, all episodes active) ---")
    with torch.no_grad():
        s0 = torch.cat([real["p0"], real["v0"], real["goal"]], dim=-1)  # [B, 6]
        s0_bar = normalize(s0)
        noise_t0 = real["noise"][:, 0, :]        # [B, k] complex, step 0
        h_t0 = (real["h"][:, 0].unsqueeze(-1)
                if "h" in real else None)         # [B, 1] complex or None

        # Run one forward pass; denormalize_output=False gives normalized estimate
        s_hat_bar_t0 = model.forward(s0, snr_db, channel=channel,
                                     noise=noise_t0, h=h_t0,
                                     denormalize_output=False)
        # MSE averaged over state dimensions and episodes
        mse_initial = float(
            ((s_hat_bar_t0 - s0_bar).pow(2).sum(dim=-1).mean() / 6.0)
        )

    print(f"MSE_initial = {mse_initial:.6f}")
    if not math.isnan(recon_mse_diag_final):
        print(f"  recon_mse_diag_final (training) = {recon_mse_diag_final:.6f}")
        print(f"  Ratio MSE_initial / recon_mse_diag_final = "
              f"{mse_initial / recon_mse_diag_final:.4f}")
        print(f"  Note: recon_mse_diag uses random SNR ~ U(0,20 dB); "
              f"MSE_initial uses fixed SNR={snr_db} dB. "
              f"Direct comparison is informative but not exact.")
    print()

    # ================================================================
    # MSE_full and MSE_active — closed-loop trajectory (T steps)
    # Uses run_method() from the evaluation harness with the same
    # paired realizations. This exactly replicates aggregate_task.mse_mean.
    # ================================================================
    print("--- MSE_full / MSE_active (closed-loop trajectory, t=0..T-1) ---")
    with torch.no_grad():
        est_fn = _neural_estimate_factory(model, channel, snr_db)
        metrics = run_method(cond, real, estimate_fn=est_fn)

    mse_per_ep = metrics["mse"]          # [B] — per-episode trajectory MSE
    exited = metrics["exited"]           # [B] bool
    active_mask = ~exited

    mse_full = float(mse_per_ep.mean())
    n_exited = int(exited.sum())
    n_active = int(active_mask.sum())

    if active_mask.any():
        mse_active = float(mse_per_ep[active_mask].mean())
    else:
        mse_active = float("nan")

    print(f"MSE_full   = {mse_full:.6f}  (mean over all {n_episodes} episodes)")
    print(f"n_exited   = {n_exited} / {n_episodes}  ({100.0*n_exited/n_episodes:.1f}%)")
    print(f"n_active   = {n_active} / {n_episodes}  ({100.0*n_active/n_episodes:.1f}%)")

    # Compare to stored aggregate_task.mse_mean
    snr_int = int(round(snr_db))
    eval_json_path = (WORKSPACE_ROOT / "results" / "M1" / "eval" /
                      f"k{k}_{channel}_seed42" / f"eval_{snr_int}.json")
    stored_mse = float("nan")
    if eval_json_path.exists():
        try:
            with open(eval_json_path) as f:
                stored = json.load(f)
            stored_mse = stored["aggregate_task"]["mse_mean"]
            print(f"Stored aggregate_task.mse_mean (N=5000, seed42, {snr_int} dB) = "
                  f"{stored_mse:.6f}")
            diff = mse_full - stored_mse
            print(f"Difference MSE_full - stored = {diff:+.6f}  "
                  f"[n_episodes differ: {n_episodes} vs 5000]")
        except Exception as e:
            print(f"WARNING: could not read eval JSON: {e}")
    else:
        print(f"(eval JSON not found at {eval_json_path}; no stored comparison)")
    print()

    print(f"MSE_active = {mse_active:.6f}  (mean over {n_active} non-exited episodes)")
    if not math.isnan(mse_active) and mse_full > 0:
        bypass_effect = mse_full - mse_active
        print(f"MSE_full - MSE_active = {bypass_effect:+.6f}")
        print(f"  Post-exit bypass: exited episodes contribute zero MSE after exit,")
        print(f"  pulling MSE_full {'below' if bypass_effect < 0 else 'above'} MSE_active.")
    print()

    # ================================================================
    # Summary ratios
    # ================================================================
    print("--- Summary ---")
    if mse_full > 0 and not math.isnan(mse_initial):
        ratio_init_full = mse_initial / mse_full
        print(f"MSE_initial / MSE_full = {ratio_init_full:.4f}")
        if ratio_init_full > 1.0:
            print(f"  -> MSE_initial > MSE_full: single-step initial MSE is larger than "
                  f"trajectory average (expected — trajectory improves as the UAV approaches goal)")
        else:
            print(f"  -> MSE_initial <= MSE_full: trajectory-averaged MSE is larger than "
                  f"single-step at t=0")
    if not math.isnan(mse_active) and mse_full > 0:
        ratio_active_full = mse_active / mse_full
        print(f"MSE_active / MSE_full  = {ratio_active_full:.4f}")
        print(f"  -> {'Active episodes have higher per-step MSE than the full-set mean' if ratio_active_full > 1.0 else 'Active episodes have lower per-step MSE than the full-set mean'}")
    print()

    print("=" * 72)
    print("Diagnostic complete. No files modified.")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = parse_args()
    run_diagnostic(args)
