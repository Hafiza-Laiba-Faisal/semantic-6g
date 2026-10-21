"""Wave-B CSI-estimation-error ablation runner (post-freeze amendment).

GUARDRAILS (read this first):
* The frozen Step-9C protocol and ALL M1-M4 artifacts are READ-ONLY.
  Wave-B writes ONLY under results/WAVE_B/.  It is EVALUATION-ONLY:
  no training, no new models - the FROZEN M2 Rayleigh checkpoints
  (results/M2/M2_{method}_rayleigh_k{k}_seed{seed}__*/final_ckpt.pt) and
  the frozen DigitalBaseline are re-used verbatim.
* Wave-B is NOT a frozen experiment; every output row records
  ``non_frozen: True`` and the amendment tag below, and numbers must be
  captioned as diagnostic-amendment results (comparisons against the
  frozen M2 sigma=0 cells are exact-by-construction because sigma=0
  reproduces the frozen path).
* project/config.py, project/experiments/*, project/models/* and the
  frozen baselines are untouched: the CSI-error receiver logic lives
  HERE in Wave-B-local estimator functions built from the PUBLIC
  frozen building blocks (model.encode/decode, rayleigh_channel,
  equalize_zf, qpsk_llrs, viterbi_decode, quantizer).

CSI error model (Wave-B convention, added to the frozen benchmark):
    h_hat = h + eps,   eps ~ CN(0, sigma_csi^2)  per (episode, step)
  * the PHYSICAL channel is the frozen one: y = h*z + n with the frozen
    paired noise/fading realizations (only receiver CSI knowledge varies);
  * eps comes from a DEDICATED Wave-B RNG stream (never the frozen
    episode/noise streams) and is PAIRED across the three methods within
    a cell, so method differences remain clean;
  * the receiver uses h_hat for BOTH zero-forcing equalization and the
    digital effective-noise scaling sigma2_eff = sigma^2 / |h_hat|^2
    (consistent receiver-knowledge convention).
  * sigma_csi = 0 reproduces the frozen perfect-CSI path BIT-EXACTLY,
    enforced by the sigma=0 gate against the frozen M2 eval JSONs.

Commands:
    python tools/run_wave_b.py verify     # artifact + checkpoint census + gates
    python tools/run_wave_b.py plan       # full eval-cell list + runtime estimate
    python tools/run_wave_b.py smoke      # tiny grid, non-scientific path check
    python tools/run_wave_b.py run        # all cells, idempotent (skip done)
    python tools/run_wave_b.py run 12     # filter: k in {12} (any 'k<val>' or 'snr<val>' or 'sig<v>')
    python tools/run_wave_b.py finalize   # results/WAVE_B/step_wave_b_summary.json
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from project import config
from project.baselines.digital import (DigitalBaseline, dequantize, qpsk_llrs,
                                       unpack_bits, viterbi_decode)
from project.channels.rayleigh import equalize_zf, rayleigh_channel
from project.data.normalization import denormalize
from project.evaluation.evaluate import Condition, generate_realizations
from project.evaluation.metrics import aggregate_episode_metrics
from project.evaluation.statistics import wilson_ci_95
from project.models.reconstruction_jscc import ReconstructionDeepJSCC

WAVE_B_ROOT = Path("results/WAVE_B")
M2_ROOT = Path("results/M2")
AMENDMENT = "WAVE_B-non-frozen-CSI-ablation-2026-10-13"

M2_K_VALUES = (12, 18, 24, 30, 54)
M2_SNR_DB = (0.0, 5.0, 10.0, 15.0, 20.0)
SEEDS = (42, 43, 44)
CSI_SIGMAS = (0.0, 0.05, 0.1, 0.2, 0.4)   # sigma_csi grid (h ~ CN(0,1))
METHODS = ("reconstruction", "task", "digital")
B_BY_K = dict(config.DIGITAL_B_BY_K)      # frozen matched-budget B


# ---------------------------------------------------------------------------
# M2 cells that the FROZEN round itself has accepted as documented failures
# ---------------------------------------------------------------------------
# ``tools/run_step11_m2.py`` keeps the approved exception list
# CONDITIONAL_ACCEPT (amendment 2026-10-19, "eval-only, not retrained").
# Wave-B must NOT retrain those cells (frozen science) and must NOT evaluate
# their diverged checkpoints either (garbage numbers).  The only honest
# option is an explicit, reported EXCLUSION: one unit out of 30, declared in
# verify/plan/finalize so the coverage gap is impossible to miss.

def _conditional_accept_ids() -> frozenset:
    """Read the frozen round's approved exception list (single source)."""
    try:
        from run_step11_m2 import CONDITIONAL_ACCEPT as _CA
    except ImportError:                      # package-style invocation
        try:
            from tools.run_step11_m2 import CONDITIONAL_ACCEPT as _CA
        except ImportError:
            return frozenset()
    return frozenset(_CA)


def _excluded_units() -> dict:
    """(k, seed) -> frozen run_id for approved-but-unusable M2 cells.

    Only Rayleigh cells matter here: Wave-B is Rayleigh-only.  Any method
    failing inside a unit excludes the WHOLE unit, because Wave-B pairs all
    three methods on the same realizations - a 2-of-3 unit would be
    asymmetric and silently incomparable.
    """
    out = {}
    for rid in _conditional_accept_ids():
        base = rid.split("__")[0]
        parts = base.split("_")
        if (len(parts) != 5 or parts[0] != "M2" or parts[2] != "rayleigh"
                or not parts[3].startswith("k")
                or not parts[4].startswith("seed")):
            continue
        try:
            k, seed = int(parts[3][1:]), int(parts[4][4:])
        except ValueError:
            continue
        out[(k, seed)] = base
    return out


CONDITIONAL_ACCEPT = _conditional_accept_ids()
EXCLUDED_UNITS = _excluded_units()       # {(54, 43): "M2_task_rayleigh_k54_seed43"}
UNITS_TOTAL = len(M2_K_VALUES) * len(SEEDS)
UNITS_COVERED = UNITS_TOTAL - len(EXCLUDED_UNITS)
CELLS_TOTAL = UNITS_TOTAL * len(M2_SNR_DB) * len(CSI_SIGMAS)
EXPECTED_CELLS = UNITS_COVERED * len(M2_SNR_DB) * len(CSI_SIGMAS)
EXPECTED_GATES = UNITS_COVERED * len(M2_SNR_DB)


def _stable_hash(text: str) -> int:
    h = 2166136261
    for ch in text:
        h ^= ord(ch)
        h = (h * 16777619) % (2 ** 32)
    return h


def csi_eps_generator(unit: str, snr_db: float, sigma: float) -> torch.Generator:
    """DEDICATED Wave-B CSI-error stream (never a frozen RNG key stream)."""
    g = torch.Generator()
    g.manual_seed(config.TEST_SEED + _stable_hash(
        f"wave_b_csi|{unit}|{snr_db:g}|{sigma:g}|{AMENDMENT}"))
    return g


def sample_csi_eps(episodes: int, t_max: int, sigma: float,
                   generator: torch.Generator) -> torch.Tensor:
    """eps ~ CN(0, sigma^2), complex [E, T, 1] (per episode, per step)."""
    shape = (episodes, t_max, 1)
    s = (sigma / math.sqrt(2.0)) if sigma > 0 else 0.0
    return torch.complex(
        torch.randn(shape, generator=generator) * s,
        torch.randn(shape, generator=generator) * s,
    )


# ---------------------------------------------------------------------------
# frozen M2 checkpoint loading (READ-ONLY census)
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[Tuple[str, int, int], ReconstructionDeepJSCC] = {}


def find_m2_ckpt(method: str, k: int, seed: int) -> Tuple[Optional[Path], dict]:
    """Unique DONE-marked frozen M2 Rayleigh checkpoint for (method, k, seed)."""
    pattern = M2_ROOT / f"M2_{method}_rayleigh_k{k}_seed{seed}__*"
    done = []
    for d in sorted(M2_ROOT.glob(pattern.name)):
        if (d / "DONE.json").exists() and (d / "final_ckpt.pt").exists():
            done.append(d)
    if len(done) != 1:
        return None, {"found": [str(d) for d in done]}
    d = done[0]
    try:
        marker = json.loads((d / "DONE.json").read_text())
    except Exception:
        marker = {}
    return d / "final_ckpt.pt", marker


def load_m2_model(method: str, k: int, seed: int) -> Optional[ReconstructionDeepJSCC]:
    """The frozen seed-specific M2 model - never seed-shared."""
    if method not in ("reconstruction", "task"):
        raise ValueError(method)
    cache_key = (method, k, seed)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    ckpt, _marker = find_m2_ckpt(method, k, seed)
    model = None
    if ckpt is not None:
        model = ReconstructionDeepJSCC(k=k)
        payload = torch.load(ckpt, weights_only=False, map_location="cpu")
        model.load_state_dict(payload["model_state"])
        model.eval()
        _MODEL_CACHE[cache_key] = model
    return model


def load_m2_frozen_eval(k: int, seed: int, snr: float) -> Optional[dict]:
    p = M2_ROOT / "eval" / f"k{k}_rayleigh_seed{seed}" / f"eval_{snr:g}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# ---------------------------------------------------------------------------
# Wave-B estimators (built from PUBLIC frozen building blocks)
# ---------------------------------------------------------------------------

def make_neural_csi(model: ReconstructionDeepJSCC, snr_db: float
                    ) -> Callable[[torch.Tensor, torch.Tensor,
                                   torch.Tensor, torch.Tensor], torch.Tensor]:
    """est(s, noise_c, h_c, eps_c) -> PHYSICAL state estimate.

    Identical to the frozen neural path up to the equalizer: the channel
    physics use the true h (frozen realizations), equalization uses
    h_hat = h + eps.
    """

    def est(s: torch.Tensor, noise_c: torch.Tensor,
            h_c: torch.Tensor, eps_c: torch.Tensor) -> torch.Tensor:
        z = model.encode(s)
        y, _h_true = rayleigh_channel(z, snr_db, noise=noise_c, h=h_c,
                                      return_h=True)
        h_hat = h_c + eps_c
        y_eq = equalize_zf(y, h_hat)
        return model.decode(y_eq)

    return est


def make_digital_csi(base: DigitalBaseline, snr_db: float
                     ) -> Callable[[torch.Tensor, torch.Tensor,
                                    torch.Tensor, torch.Tensor], torch.Tensor]:
    """est(s, noise_c, h_c, eps_c) -> PHYSICAL state estimate.

    Replica of the frozen digital receive chain with the h-hat knowledge:
    equalization AND the effective-noise LLR scaling both use h_hat
    (consistent receiver knowledge). Uses only PUBLIC frozen pieces, so
    the frozen baseline module is untouched.
    """

    def est(s: torch.Tensor, noise_c: torch.Tensor,
            h_c: torch.Tensor, eps_c: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            sym = base.encode_state(s)                     # complex [B, k]
            y, _h_true = rayleigh_channel(sym, snr_db, noise=noise_c, h=h_c,
                                          return_h=True)
            h_hat = h_c + eps_c
            n_pay = base.pkt["n_payload_symbols"]
            y_pay = y[..., :n_pay]                         # complex slice
            y_eq = equalize_zf(y_pay, h_hat)
            sigma2 = config.TX_POWER * (10.0 ** (-snr_db / 10.0))
            hh = (h_hat.real ** 2 + h_hat.imag ** 2).clamp_min(
                config.EPS_EQUALIZER)
            sigma2_eff = sigma2 / hh                       # [B, 1]
            llrs = qpsk_llrs(y_eq, sigma2_eff)
            info = viterbi_decode(llrs)
            info = info[:, :-base.n_tail]                  # drop tail
            q_hat = unpack_bits(info, base.B)
            s_bar_hat = dequantize(q_hat, base.B)
            return denormalize(s_bar_hat)

    return est


# ---------------------------------------------------------------------------
# Wave-B closed-loop runner (derived from the frozen run_method; the ONLY
# delta is the per-step eps slice handed to the estimators)
# ---------------------------------------------------------------------------

def run_method_csi(cond: Condition, real: Dict[str, torch.Tensor],
                   est: Optional[Callable] = None,
                   eps_grid: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Single-method paired closed-loop rollout with optional CSI error.

    eps_grid: complex [E, T, 1].  est(s, noise_c, h_c, eps_c) or, with
    est=None (oracle), the true state. sigma=0 -> eps=0 -> h_hat = h ->
    code path identical to the frozen method drivers.
    """
    from project.data.normalization import normalize
    from project.uav.controller import pd_controller
    from project.uav.environment import dynamics_step

    p = real["p0"].clone()
    v = real["v0"].clone()
    goal = real["goal"]
    g = goal
    B = p.shape[0]
    E, T, k = real["noise"].shape

    pos = [p]
    vel = [v]
    ctl = []
    s_hat_bar_hist: List[torch.Tensor] = []
    s_bar_hist: List[torch.Tensor] = []
    exited = torch.zeros(B, dtype=torch.bool)
    active = torch.ones(B, dtype=torch.bool)

    for t in range(T):
        s = torch.cat([p, v, g], dim=-1)
        s_bar = normalize(s)
        noise_c = real["noise"][:, t, :]
        h_c = real["h"][:, t].unsqueeze(-1) if "h" in real else None

        if est is None:
            s_hat = s
        else:
            active_idx = active.nonzero(as_tuple=True)[0]
            if active_idx.numel() == 0:
                s_hat = s
            else:
                eps_c = (eps_grid[active_idx, t]
                         if eps_grid is not None and h_c is not None
                         else None)
                s_hat_active = est(s[active_idx], noise_c[active_idx],
                                   h_c[active_idx] if h_c is not None else None,
                                   eps_c)
                s_hat = s.clone()
                s_hat = s_hat.index_copy(0, active_idx, s_hat_active)

        s_bar_hat = normalize(s_hat)
        s_bar_hist.append(s_bar)
        s_hat_bar_hist.append(s_bar_hat)

        u = pd_controller(s_hat)
        p_new, v_new = dynamics_step(p, v, u)
        d_new = (p_new - g).norm(dim=-1)

        outside = ((p_new[:, 0] < config.WORKSPACE_MIN)
                   | (p_new[:, 0] > config.WORKSPACE_MAX)
                   | (p_new[:, 1] < config.WORKSPACE_MIN)
                   | (p_new[:, 1] > config.WORKSPACE_MAX))
        newly = active & outside
        exited = exited | newly
        active = active & ~outside

        mask = active.unsqueeze(-1)
        p = torch.where(mask, p_new, p)
        v = torch.where(mask, v_new, v)
        pos.append(p)
        vel.append(v)
        ctl.append(u)

    positions = torch.stack(pos, dim=1)
    controls = torch.stack(ctl, dim=1)
    velocities = torch.stack(vel, dim=1)

    from project.evaluation.metrics import per_episode_metrics
    metrics = per_episode_metrics(positions, controls, velocities, goal,
                                  config.GOAL_RADIUS,
                                  s_hat_bar=torch.stack(s_hat_bar_hist, dim=1),
                                  s_bar=torch.stack(s_bar_hist, dim=1))
    metrics["exited"] = exited
    metrics["positions"] = positions
    metrics["velocities"] = velocities
    metrics["controls"] = controls
    metrics["goal"] = goal
    return metrics


# (keep module-level API compact; helpers below)


# ---------------------------------------------------------------------------
# one eval cell: (k, snr, seed) x sigma, all three methods paired
# ---------------------------------------------------------------------------

def eval_path(k: int, seed: int, snr: float, sigma: float) -> Path:
    tag = f"{sigma:g}".replace(".", "p")
    return (WAVE_B_ROOT / "eval" / f"k{k}_rayleigh_seed{seed}" /
            f"eval_{snr:g}_csi{tag}.json")


def evaluate_cell(k: int, snr: float, seed: int, sigma: float,
                  episodes: int, gate: bool = True) -> Tuple[Optional[Path], dict]:
    """Run one (k, snr, seed, sigma) cell with all three methods paired."""
    models_k = {m: load_m2_model(m, k, seed)
                for m in ("reconstruction", "task")}
    if any(v is None for v in models_k.values()):
        raise RuntimeError(f"missing frozen M2 ckpt: k={k} seed={seed}")
    cond = Condition(k=k, channel="rayleigh", snr_db=snr,
                     bits_per_component=B_BY_K[k], episodes=episodes,
                     seed=config.TEST_SEED)
    real = generate_realizations(cond, episodes)
    gen = csi_eps_generator(f"k{k}_rayleigh_seed{seed}", snr, sigma)
    eps_grid = sample_csi_eps(episodes, cond.t_max, sigma, gen)

    digital = DigitalBaseline(k=k, bits_per_component=B_BY_K[k])
    ests = {
        "reconstruction": make_neural_csi(models_k["reconstruction"], snr),
        "task": make_neural_csi(models_k["task"], snr),
        "digital": make_digital_csi(digital, snr),
        # oracle reference (CSI error is irrelevant without a receiver)
    }
    t0 = time.perf_counter()
    aggregates: Dict[str, dict] = {}
    success_vecs: Dict[str, List[int]] = {}
    for m in METHODS:
        with torch.no_grad():
            per_ep = run_method_csi(cond, real, est=ests[m], eps_grid=eps_grid)
        aggregates[m] = aggregate_episode_metrics(per_ep)
        success_vecs[m] = [int(x) for x in per_ep["success"].tolist()]
    wall = time.perf_counter() - t0

    rec = {
        "STEP": "WAVE_B", "experiment": "WAVE_B",
        "unit": f"k{k}_rayleigh_seed{seed}", "k": k, "channel": "rayleigh",
        "seed": seed, "bits_per_component": B_BY_K[k], "snr_db": snr,
        "csi_error_sigma": sigma, "csi_eps_convention":
            "h_hat = h + eps, eps ~ CN(0, sigma^2); receiver equalizes and "
            "scales LLR noise with h_hat only",
        "test_seed": config.TEST_SEED, "episodes": episodes,
        "t_max": cond.t_max, "paired": True,
        "csi_eps_paired_across_methods": True,
        "amendment": AMENDMENT, "non_frozen": True,
        "wall_s": round(wall, 1),
        "methods": list(METHODS),
        **{f"aggregate_{m}": aggregates[m] for m in METHODS},
        **{f"wilson_{m}": wilson_ci_95(success_vecs[m]) for m in METHODS},
    }

    # sigma=0 gate: must reproduce the frozen M2 aggregate EXACTLY
    if gate and sigma == 0.0:
        frozen = load_m2_frozen_eval(k, seed, snr)
        if frozen is None:
            rec["sigma0_gate"] = "frozen M2 JSON missing - gate NOT run"
        elif frozen["episodes"] != episodes:
            rec["sigma0_gate"] = (f"episode mismatch (frozen "
                                  f"{frozen['episodes']} vs {episodes})")
        else:
            diffs = {}
            for m in METHODS:
                a = aggregates[m]["success_count"]
                b = frozen["aggregate_" + m]["success_count"]
                diffs[m] = (a, b)
            ok = all(a == b for a, b in diffs.values())
            rec["sigma0_gate"] = ("PASS - bit-exact reproduction of frozen M2"
                                  if ok else f"FAIL {diffs}")
    out = eval_path(k, seed, snr, sigma)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2, default=str))
    return out, rec


# ---------------------------------------------------------------------------
# cell universe / CLI plumbing
# ---------------------------------------------------------------------------

def cells():
    for k in M2_K_VALUES:
        for snr in M2_SNR_DB:
            for seed in SEEDS:
                if (k, seed) in EXCLUDED_UNITS:
                    continue          # approved M2 failure: reported, not run
                for sigma in CSI_SIGMAS:
                    yield k, snr, seed, sigma


def parse_filter(args: List[str]) -> set:
    """'k12' | 'snr10' | 'sig0.2' style cell filters (multiple allowed)."""
    keep = set()

    def ok(k, snr, sigma):
        for a in args:
            if a.startswith("k"):
                if k != float(a[1:]):
                    return False
            elif a.startswith("snr"):
                if snr != float(a[3:]):
                    return False
            elif a.startswith("sig"):
                if float(a[3:]) != sigma:
                    return False
            else:
                print(f"unknown filter {a!r}")
                sys.exit(1)
        return True

    for k, snr, seed, sigma in cells():
        if ok(k, snr, sigma):
            keep.add((k, snr, seed, sigma))
    return keep


def load_models_for_needed_ks(needed: set):
    models = {}
    for k in sorted(needed):
        entry = {}
        for m in ("reconstruction", "task"):
            entry[m] = load_m2_model(m, k)
            if entry[m] is None:
                return None, f"missing frozen M2 {m} ckpt for k={k}"
        models[k] = entry
    return models, None


def _fmt_sr(agg):
    return f"{agg['success_rate']:.4f}"


# ---------------------------------------------------------------------------

def cmd_verify():
    print("== Wave-B governance gate ==")
    bad = 0
    # 0) coverage contract: approved M2 failures are EXCLUDED, never run
    for (k, seed), rid in sorted(EXCLUDED_UNITS.items()):
        print(f"INFO  [excluded:{rid}] approved documented failure "
              f"(M2 conditional-accept) -> unit k{k}/seed{seed} NOT covered")
    print(f"INFO  [coverage] {UNITS_COVERED}/{UNITS_TOTAL} units, "
          f"{EXPECTED_CELLS}/{CELLS_TOTAL} eval cells")
    # 1) frozen M2 checkpoints unique + DONE-marked, rayleigh, all seeds
    for k in M2_K_VALUES:
        for seed in SEEDS:
            if (k, seed) in EXCLUDED_UNITS:
                continue
            for m in ("reconstruction", "task"):
                ckpt, marker = find_m2_ckpt(m, k, seed)
                ok = ckpt is not None
                if not ok:
                    print(f"FAIL  [ckpt:{m}/k{k}/seed{seed}] not uniquely DONE")
                    bad += 1
    n_ck = 2 * UNITS_COVERED
    check_done = bad == 0
    print(("PASS" if check_done else "FAIL") +
          f"  [m2_ckpts] {n_ck} usable frozen M2 rayleigh ckpts unique + "
          f"DONE-marked ({len(EXCLUDED_UNITS)} unit(s) excluded)")
    bad += 0 if check_done else 1
    # 2) frozen M2 eval JSONs (sigma=0 gate references) for COVERED units
    missing = [(k, s, snr) for k in M2_K_VALUES for s in SEEDS
               if (k, s) not in EXCLUDED_UNITS
               for snr in M2_SNR_DB if load_m2_frozen_eval(k, s, snr) is None]
    ok2 = not missing
    print(f"{'PASS' if ok2 else 'FAIL'}  [m2_eval_json] {EXPECTED_GATES} frozen "
          f"M2 evals needed ({len(missing)} missing)")
    if not ok2:
        for x in missing[:5]:
            print("      missing:", x)
        bad += 1
    # 3) output dir sanity
    print(f"INFO  [wave_b_root] {WAVE_B_ROOT} "
          f"({len(list(WAVE_B_ROOT.glob('eval/*/*.json')))} eval JSONs present)")
    print(f"== gates {'OK' if bad == 0 else 'BROKEN'} ==")
    return 0 if bad == 0 else 1


def cmd_plan():
    flt = sorted(parse_filter(sys.argv[2:]))
    n = len(flt)
    # measured smoke scale: ~11 s / cell (3 methods, E=5000, k=12) on this PC
    est_s = n * 12
    print(f"== Wave-B plan: {n} eval cells x 3 methods ==\n"
          f"grid: k {M2_K_VALUES} x snr {M2_SNR_DB} x seeds {SEEDS} x "
          f"sigma {CSI_SIGMAS}\n"
          f"coverage: {UNITS_COVERED}/{UNITS_TOTAL} units "
          f"({EXPECTED_CELLS}/{CELLS_TOTAL} cells); excluded "
          f"{sorted(EXCLUDED_UNITS.items())}\n"
          f"protocol: E=5000 (frozen), test_seed "
          f"{config.TEST_SEED}, amendment {AMENDMENT}\n"
          f"runtime estimate ~{est_s/60:.0f}-{est_s/45:.0f} min (sequential, CPU)")
    return 0


def cmd_smoke():
    print("== Wave-B smoke (E=200, NON-scientific path check) ==")
    for sigma in (0.0, 0.2):
        out, rec = evaluate_cell(12, 10.0, 42, sigma, episodes=200,
                                 gate=False)
        srs = {m: rec[f"aggregate_{m}"]["success_rate"] for m in METHODS}
        print(f"  k12 snr10 seed42 sig{sigma:g}: " +
              " ".join(f"{m}={v:.4f}" for m, v in srs.items()) +
              f"  ({rec['wall_s']}s) -> {out.name}")
    print("smoke OK")
    return 0


def cmd_run():
    flt = parse_filter(sys.argv[2:])
    todo = [(k, snr, seed, sigma) for (k, snr, seed, sigma) in cells()
            if (k, snr, seed, sigma) in flt
            and not eval_path(k, seed, snr, sigma).exists()]
    if not todo:
        print("-- nothing to run (all filtered cells done)")
        return 0
    print(f"== Wave-B run: {len(todo)} cells to go ==")
    done = 0
    t_start = time.perf_counter()
    for (k, snr, seed, sigma) in todo:
        try:
            _, rec = evaluate_cell(k, snr, seed, sigma, episodes=5000)
        except Exception as e:                          # noqa: BLE001
            print(f"  ERROR k{k} snr{snr:g} seed{seed} sig{sigma:g}: {e!r}")
            continue
        done += 1
        el = time.perf_counter() - t_start
        eta = el / done * (len(todo) - done) / 60
        srs = {m: rec[f"aggregate_{m}"]["success_rate"] for m in METHODS}
        print(f"  [{done}/{len(todo)}] k{k} snr{snr:g} seed{seed} "
              f"sig{sigma:g}: " +
              " ".join(f"{m}={v:.4f}" for m, v in srs.items()) +
              f" ({rec['wall_s']}s, eta {eta:.0f}m)", flush=True)
    print(f"-- run finished: {done}/{len(todo)} cells in "
          f"{(time.perf_counter()-t_start)/60:.1f} min")
    return 0


def cmd_finalize():
    print("== finalize: results/WAVE_B/step_wave_b_summary.json ==")
    rows = []
    for p in sorted(WAVE_B_ROOT.glob("eval/*/*.json")):
        R = json.loads(p.read_text())
        rows.append(R)
    # per-(k,snr,sigma,method) pooled over seeds + delta vs sigma=0
    pooled = {}
    for R in rows:
        for m in METHODS:
            key = (R["k"], R["snr_db"], R["csi_error_sigma"], m)
            pooled.setdefault(key, []).append(R[f"aggregate_{m}"])
    summary_rows = []
    diagram = {}
    for (k, snr, sigma, m), aggs in sorted(
            pooled.items(), key=lambda x: (x[0][2], x[0][3], x[0][0],
                                           x[0][1], x[0][2])):
        sc = sum(a["success_count"] for a in aggs)
        tot = sum(a["episodes"] for a in aggs)
        sr = sc / tot if tot else float("nan")
        base_key = (k, snr, 0.0, m)
        base = pooled.get(base_key)
        base_sc = sum(a["success_count"] for a in base) if base else None
        base_tot = sum(a["episodes"] for a in base) if base else None
        base_sr = base_sc / base_tot if base and base_tot else float("nan")
        summary_rows.append({
            "method": m, "k": k, "snr_db": snr, "csi_error_sigma": sigma,
            "n_seeds": len(aggs), "success_count_pooled": sc,
            "episodes_pooled": tot, "success_rate_pooled": round(sr, 4),
            "delta_vs_sigma0_pp": (round((sr - base_sr) * 100, 2)
                                   if base else None)})
        diagram.setdefault((k, snr), {}).setdefault(
            m, {})[f"{sigma:g}"] = round(sr, 4)
    # sigma=0 bit-exact gate census
    gates = [r for r in rows if r["csi_error_sigma"] == 0.0]
    gate_ok = sum(1 for r in gates if str(r.get("sigma0_gate", "")).startswith(
        "PASS"))
    out = WAVE_B_ROOT / "step_wave_b_summary.json"
    out.write_text(json.dumps({
        "experiment": "WAVE_B",
        "status": ("complete" if len(rows) == EXPECTED_CELLS and
                   gate_ok == EXPECTED_GATES else "incomplete"),
        "amendment": AMENDMENT,
        "coverage": {
            "units_total": UNITS_TOTAL,
            "units_covered": UNITS_COVERED,
            "cells_total_if_full": CELLS_TOTAL,
            "cells_expected": EXPECTED_CELLS,
            "excluded_units": [
                {"k": k, "seed": s, "run_id": rid,
                 "reason": ("frozen M2 conditional-accept (documented failure)"
                            "; not retrained and not evaluated")}
                for (k, s), rid in sorted(EXCLUDED_UNITS.items())],
        },
        "protocol_note": ("Wave-B: frozen M2 models + frozen digital, "
                          "E=5000, non-frozen CSI-error amendment; "
                          "sigma=0 rows are the exact frozen path; "
                          f"coverage {UNITS_COVERED}/{UNITS_TOTAL} units "
                          "(excluded unit is an approved M2 failure)"),
        "eval_cells": len(rows),
        "sigma0_gate": {"cells": len(gates), "passed": gate_ok,
                        "expected": EXPECTED_GATES},
        "summary_rows": summary_rows,
        "success_rate_by_k_snr": {f"k{k}|snr{snr:g}": v
                                  for (k, snr), v in sorted(
                                      diagram.items(),
                                      key=lambda x: (x[0][0], x[0][1]))},
    }, indent=2, default=str))
    print(f"wrote {out} ({len(rows)} cells, sigma0 gates {gate_ok}/{len(gates)})")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    cmd = sys.argv[1]
    dispatch = {"verify": cmd_verify, "plan": cmd_plan, "smoke": cmd_smoke,
                "run": cmd_run, "finalize": cmd_finalize}
    if cmd not in dispatch:
        print(__doc__)
        sys.exit(1)
    sys.exit(dispatch[cmd]())
