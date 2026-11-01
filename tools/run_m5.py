"""M5: the deferred ablation set, executed under amendment A-16 (2026-11-01).

M5 was declared in the frozen Step-9C protocol and then deferred because its
three components were under-specified.  A-16 supplies the specification before
the first cell runs.  Nothing frozen is edited: M5 writes only under
results/M5/ and reuses the Wave-A machinery by patching its universe and root
(the same isolation pattern as tools/run_wave_c.py).

DESIGN (fyp_md_files/PROTOCOL_AMENDMENTS.md A-16):

  M5a  task-loss component removal, 12 training cells
       arms    : m5_no_u (lambda_u = 0), m5_no_v (lambda_v = 0),
                 m5_no_T (lambda_T = 0) - each one term removed from
                 the lambda-heavy arm (1.0, 0.1, 0.1, 2.0)
       k       : 12, 18          channels : awgn, rayleigh    seed : 42
       steps   : config.TASK_TRAIN_STEPS (frozen 3000)
       eval    : E=1000, 5-SNR grid -> 6 units / 30 slots
       compare : STORED lam_ctl_heavy (Wave-A k=12, Wave-C k=18) - not retrained

  M5b  uncoded digital reference, 20 evaluation cells (no training)
       variants: uncoded_bmax  (B = floor(k/3), full bandwidth, no code)
                 uncoded_bcoded (B = floor((k-6)/6), same payload as the
                                 coded baseline, code removed)
       k       : 12, 18, 24, 30, 54    channels : awgn, rayleigh
       eval    : E=1000, 5-SNR grid

  M5c  cross-channel transfer, 10 evaluation slots (no training)
       source  : Wave-C lam_ctl_heavy k=18 seed42, trained on awgn / rayleigh
       test    : the other channel
       eval    : E=1000, 5-SNR grid

Usage (repo root, inside the venv):
    python tools/run_m5.py verify            # pre-flight gates
    python tools/run_m5.py smoke             # 1 cell, 100 steps, no artifacts
    python tools/run_m5.py train             # idempotent M5a queue
    python tools/run_m5.py evaluate          # M5a paired eval (E=1000)
    python tools/run_m5.py uncoded           # M5b (evaluation only)
    python tools/run_m5.py cross             # M5c (evaluation only)
    python tools/run_m5.py status
    python tools/run_m5.py finalize          # results/M5/step_m5_summary.json
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

import tools.run_wave_a as rw  # noqa: E402
import tools.run_wave_c as rwc  # noqa: E402
from project import config  # noqa: E402
from project.baselines.digital import (  # noqa: E402
    dequantize,
    pack_bits,
    qpsk_llrs,
    qpsk_modulate,
    quantize,
    unpack_bits,
)
from project.channels.awgn import awgn_channel  # noqa: E402
from project.channels.complex_utils import check_codeword  # noqa: E402
from project.channels.rayleigh import equalize_zf, rayleigh_channel  # noqa: E402
from project.data.normalization import denormalize, normalize  # noqa: E402
from project.evaluation.evaluate import (  # noqa: E402
    Condition,
    aggregate,
    evaluate_condition,
    generate_realizations,
    run_method,
)

M5_ROOT = ROOT / "results" / "M5"
WAVE_C_ROOT = rwc.WAVE_C_ROOT
AMENDMENT = "M5-A16-2026-11-01"

# tools.run_wave_c rewrites rw.WAVE_A_ROOT at import time; restore the M5 root
# BEFORE any M5 helper runs.
rw.WAVE_A_ROOT = M5_ROOT
rw.WAVE_LABEL = "M5"
rw.WAVE_EXPERIMENT = "M5"
rw.WAVE_AMENDMENT = AMENDMENT

CHANNELS = ("awgn", "rayleigh")
SNR_GRID = rw.MATCH_SNR_DB
EVAL_EPISODES = rw.EVAL_EPISODES


# ---------------------------------------------------------------------------
# M5a - task-loss component removal
# ---------------------------------------------------------------------------

M5A_ARMS = {
    # one term removed at a time from lam_ctl_heavy = (1.0, 0.1, 0.1, 2.0)
    "m5_no_u": (1.0, 0.00, 0.10, 2.0),
    "m5_no_v": (1.0, 0.10, 0.00, 2.0),
    "m5_no_T": (1.0, 0.10, 0.10, 0.00),
}
M5A_KS = (12, 18)
M5A_SEED = 42


class M5Cell(rw.WaveCell):
    """Wave-A cell with an M5 identity and its own amendment label."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.wave_id = self.wave_id.replace("WAVE_A_", "M5_")

    def spec(self):
        spec = super().spec()
        spec["experiment"] = "M5"
        spec["amendment"] = AMENDMENT
        return spec


def m5a_cells():
    out = []
    for tag, lam in M5A_ARMS.items():
        for k in M5A_KS:
            for ch in CHANNELS:
                out.append(M5Cell("A16a", "task", ch, k, M5A_SEED,
                                  tag=tag, lambdas=lam))
    return out


def m5a_units():
    seen = {}
    for c in m5a_cells():
        key = (c.k, c.channel, c.seed)
        seen.setdefault(key, {"unit": f"k{c.k}_{c.channel}_seed{c.seed}",
                              "k": c.k, "channel": c.channel, "seed": c.seed})
    return [seen[key] for key in sorted(seen)]


def m5a_eval_path(unit, snr):
    return M5_ROOT / "eval" / unit["unit"] / f"eval_{snr:g}.json"


# ---------------------------------------------------------------------------
# M5b - uncoded QPSK reference
# ---------------------------------------------------------------------------

M5B_KS = (12, 18, 24, 30, 54)


def uncoded_payloads(k):
    """Payload sizes in bits/component for the two uncoded variants."""
    b_max = k // 3                       # 2 bits/symbol, 6 components
    b_coded = (k - 6) // 6               # same payload as the coded baseline
    out = {"uncoded_bmax": b_max}
    if b_coded != b_max:
        out["uncoded_bcoded"] = b_coded
    return out


class UncodedQPSK:
    """Quantize -> pack -> Gray-QPSK, no channel code. Evaluation only.

    Same ``forward`` signature as :class:`DigitalBaseline` so it can be driven
    by the frozen paired-evaluation harness without touching it.
    """

    def __init__(self, k, bits_per_component):
        self.k = k
        self.B = int(bits_per_component)
        self.n_bits = 6 * self.B
        if self.n_bits % 2:
            raise ValueError("uncoded QPSK needs an even payload bit count")
        self.n_sym = self.n_bits // 2
        if self.n_sym > k:
            raise ValueError(f"uncoded payload needs {self.n_sym} > k={k} symbols")

    def encode_state(self, s):
        s_bar = normalize(s)
        q, _ = quantize(s_bar, self.B)
        bits = pack_bits(q, self.B).to(torch.long)
        sym = qpsk_modulate(bits)
        pad_sym = self.k - self.n_sym
        if pad_sym:
            pad_bits = 2 * pad_sym
            pad = qpsk_modulate(torch.zeros(*sym.shape[:-1], pad_bits,
                                            dtype=torch.long, device=sym.device))
            sym = torch.cat([sym, pad], dim=-1)
        assert sym.shape[-1] == self.k
        return sym

    def decode_symbols(self, y, channel, snr_db, h=None):
        check_codeword(y)
        y_pay = y[..., :self.n_sym]
        if channel == "rayleigh":
            if h is None:
                raise ValueError("Rayleigh decoding needs the CSI tensor h")
            y_pay = equalize_zf(y_pay, h)
        sigma2 = config.TX_POWER * (10.0 ** (-snr_db / 10.0))
        if channel == "rayleigh":
            hh = (h.real ** 2 + h.imag ** 2).clamp_min(config.EPS_EQUALIZER)
            sigma2_eff = sigma2 / hh
        else:
            sigma2_eff = torch.tensor(sigma2, dtype=y_pay.real.dtype)
        llrs = qpsk_llrs(y_pay, sigma2_eff)
        bits = (llrs < 0).to(torch.long)      # LLR > 0 means bit 0
        return dequantize(unpack_bits(bits, self.B), self.B)

    def forward(self, s, snr_db, channel="awgn", generator=None,
                noiseless=False, noise=None, h=None):
        if channel not in ("awgn", "rayleigh"):
            raise ValueError(f"unknown channel {channel!r}")
        with torch.no_grad():
            sym = self.encode_state(s)
            if noiseless:
                return denormalize(self.decode_symbols(sym, "awgn", snr_db))
            if channel == "awgn":
                y = awgn_channel(sym, snr_db, generator=generator, noise=noise)
                return denormalize(self.decode_symbols(y, "awgn", snr_db))
            y, h_used = rayleigh_channel(sym, snr_db, generator=generator,
                                         noise=noise, h=h, return_h=True)
            return denormalize(
                self.decode_symbols(y, "rayleigh", snr_db, h=h_used))


def m5b_conditions():
    out = []
    for k in M5B_KS:
        for variant, B in uncoded_payloads(k).items():
            for ch in CHANNELS:
                out.append({"variant": variant, "k": k, "channel": ch,
                            "bits_per_component": B})
    return out


def m5b_path(cond, snr):
    return (M5_ROOT / "uncoded" / cond["variant"]
            / f"k{cond['k']}_{cond['channel']}" / f"eval_{snr:g}.json")


# ---------------------------------------------------------------------------
# M5c - cross-channel transfer
# ---------------------------------------------------------------------------

M5C_K = 18
M5C_SEED = 42
M5C_TAG = "lam_ctl_heavy"


def m5c_units():
    return [{"trained_on": t,
             "eval_channel": ("rayleigh" if t == "awgn" else "awgn")}
            for t in CHANNELS]


def m5c_source_cell(trained_on):
    for c in rwc.cells():
        if (c.tag == M5C_TAG and c.k == M5C_K and c.seed == M5C_SEED
                and c.channel == trained_on):
            return c
    return None


def m5c_source_dir(cell):
    """Wave-C run directory (NOT the patched M5 root)."""
    return WAVE_C_ROOT / cell.run_dirname()


def m5c_path(unit, snr):
    return (M5_ROOT / "cross"
            / f"train_{unit['trained_on']}_eval_{unit['eval_channel']}"
            / f"eval_{snr:g}.json")


# ---------------------------------------------------------------------------
# expected universe
# ---------------------------------------------------------------------------

EXPECTED_TRAIN = len(m5a_cells())
EXPECTED_UNITS = len(m5a_units())
EXPECTED_SLOTS = EXPECTED_UNITS * len(SNR_GRID)
EXPECTED_UNCODED = len(m5b_conditions())
EXPECTED_UNCODED_SLOTS = EXPECTED_UNCODED * len(SNR_GRID)
EXPECTED_CROSS = len(m5c_units())
EXPECTED_CROSS_SLOTS = EXPECTED_CROSS * len(SNR_GRID)


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

def cmd_verify():
    print("== M5 pre-flight gates (A-16) ==")
    bad = 0

    def gate(name, ok, detail=""):
        nonlocal bad
        print(f"{'PASS' if ok else 'FAIL'}  [{name}]"
              + (f"  ({detail})" if detail else ""))
        if not ok:
            bad += 1

    gate("isolated_root",
         M5_ROOT.name == "M5" and M5_ROOT.parent == ROOT / "results",
         str(M5_ROOT.relative_to(ROOT)))
    frozen_roots = {ROOT / "results" / n for n in
                    ("M1", "M2", "M3", "M4", "WAVE_A", "WAVE_B", "WAVE_C")}
    gate("not_a_frozen_root", M5_ROOT not in frozen_roots)

    ids = [c.wave_id for c in m5a_cells()]
    gate("unique_ids", len(set(ids)) == len(ids), f"{len(ids)} cells")
    gate("id_prefix", all(i.startswith("M5_") for i in ids))
    gate("design",
         (EXPECTED_TRAIN, EXPECTED_UNITS, EXPECTED_SLOTS) == (12, 4, 20),
         f"{EXPECTED_TRAIN} cells / {EXPECTED_UNITS} units / {EXPECTED_SLOTS} slots")
    gate("step_budget",
         {c.steps for c in m5a_cells()} == {config.TASK_TRAIN_STEPS},
         str(config.TASK_TRAIN_STEPS))
    gate("one_seed", {c.seed for c in m5a_cells()} == {M5A_SEED})
    gate("k_grid", tuple(sorted({c.k for c in m5a_cells()})) == M5A_KS)
    gate("three_arms", len(M5A_ARMS) == 3)
    gate("uncoded_design", EXPECTED_UNCODED == 20,
         f"{EXPECTED_UNCODED} cells / {EXPECTED_UNCODED_SLOTS} slots")
    gate("cross_design", EXPECTED_CROSS == 2,
         f"{EXPECTED_CROSS} units / {EXPECTED_CROSS_SLOTS} slots")

    # every M5b payload must be encodable within k
    ok_payload = all(c["bits_per_component"] >= 1
                     and 3 * c["bits_per_component"] <= c["k"]
                     for c in m5b_conditions())
    gate("uncoded_payload_fits", ok_payload)

    # the stored comparator must exist (M5a) and the source checkpoints must
    # exist (M5c).  Both are read-only inputs.
    wa_ok = (ROOT / "results" / "WAVE_A").is_dir()
    wc_ok = all(m5c_source_cell(t) is not None for t in CHANNELS)
    gate("comparator_wave_a_present", wa_ok)
    gate("comparator_wave_c_present", wc_ok)
    ckpts = []
    for t in CHANNELS:
        c = m5c_source_cell(t)
        if c is None:
            ckpts.append(False)
            continue
        ckpts.append((m5c_source_dir(c) / "final_ckpt.pt").exists())
    gate("wave_c_checkpoints_present", all(ckpts), f"{sum(ckpts)}/2")

    gate("frozen_config_intact",
         (config.RECON_TRAIN_STEPS, config.TASK_TRAIN_STEPS) == (5000, 3000)
         and (config.LAMBDA_DISTANCE, config.LAMBDA_CONTROL,
              config.LAMBDA_VELOCITY, config.LAMBDA_TERMINAL)
         == (1.0, 0.0, 0.0, 2.0)
         and config.TRAINING_SEEDS == (42, 43, 44)
         and config.TEST_SEED == 10042)

    print("== gates OK ==" if bad == 0 else f"== {bad} gate(s) FAILED ==")
    return 0 if bad == 0 else 1


# ---------------------------------------------------------------------------
# smoke / train / evaluate (M5a, delegated to the Wave-A machinery)
# ---------------------------------------------------------------------------

def cmd_smoke(k=None):
    pool = [c for c in m5a_cells() if k is None or c.k == k]
    c = pool[0]
    c.steps = 100
    print(f"== M5 smoke: {c.wave_id} steps={c.steps} arm={c.tag} ==")
    t0 = time.perf_counter()
    model, rep, grads_finite, summary = rw._train_one(c)
    wall = time.perf_counter() - t0
    loss = summary.get("task_loss_final")
    ok = (rep["nan_batches"] == 0 and grads_finite
          and isinstance(loss, float) and loss == loss
          and loss != float("inf"))
    print(f"   wall={wall:.1f}s nan_batches={rep['nan_batches']} "
          f"grads_finite={grads_finite} task_loss_final={loss}")
    print(f"   projection: 12 cells x {wall * 3000 / 100 / 60:.1f} min "
          f"= {12 * wall * 3000 / 100 / 3600:.1f} h")
    print("   smoke wrote NOTHING (no markers, no checkpoints)")
    return 0 if ok else 1


def _select(args):
    out = m5a_cells()
    tags = [a.split("=", 1)[1] for a in args if a.startswith("--tag=")]
    ks = [int(a.split("=", 1)[1]) for a in args if a.startswith("--k=")]
    seeds = [int(a.split("=", 1)[1]) for a in args if a.startswith("--seed=")]
    shard = next((a.split("=", 1)[1] for a in args
                  if a.startswith("--shard=")), None)
    if tags:
        out = [c for c in out if c.tag in tags]
    if ks:
        out = [c for c in out if c.k in ks]
    if seeds:
        out = [c for c in out if c.seed in seeds]
    if shard:
        i, n = (int(x) for x in shard.split("/"))
        out = [c for j, c in enumerate(out) if j % n == i]
    return out


def cmd_train(lane=None, args=()):
    sel = _select(args)
    if len(sel) != len(m5a_cells()):
        print(f"== M5 selection: {len(sel)} of {len(m5a_cells())} cells ==",
              flush=True)
        for c in sel:
            print(f"   {c.wave_id}", flush=True)
    rw.wave_cells = lambda: sel
    rw.DOCUMENTED_FAILURES = frozenset()
    return rw.cmd_train(lane)


def cmd_evaluate(lane=None, args=()):
    sel = _select(args) if args else m5a_cells()
    rw.wave_cells = lambda: sel
    return rw.cmd_evaluate(lane)


# ---------------------------------------------------------------------------
# M5b - uncoded evaluation (evaluation only, no training)
# ---------------------------------------------------------------------------

def cmd_uncoded():
    print("== M5b uncoded QPSK reference (E=1000, evaluation only) ==",
          flush=True)
    n_new, total_wall = 0, 0.0
    for cond in m5b_conditions():
        B = cond["bits_per_component"]
        try:
            link = UncodedQPSK(cond["k"], B)
        except ValueError as exc:
            print(f"  SKIP k={cond['k']} {cond['variant']}: {exc}", flush=True)
            continue
        for snr in SNR_GRID:
            out = m5b_path(cond, snr)
            if out.exists():
                continue
            c = Condition(k=cond["k"], channel=cond["channel"], snr_db=snr,
                          t_max=config.T_MAX, bits_per_component=B,
                          episodes=EVAL_EPISODES, seed=config.TEST_SEED)
            real = generate_realizations(c, EVAL_EPISODES)
            t0 = time.perf_counter()

            def est(s, noise_c, h_c, _link=link, _snr=snr,
                    _ch=cond["channel"]):
                return _link.forward(s, _snr, channel=_ch,
                                     noise=noise_c, h=h_c)

            with torch.no_grad():
                metrics = run_method(c, real, estimate_fn=est)
            agg = aggregate(metrics)
            wall = time.perf_counter() - t0
            total_wall += wall
            rec = {
                "experiment": "M5", "step": "M5b",
                "variant": cond["variant"], "k": cond["k"],
                "channel": cond["channel"], "snr_db": snr,
                "bits_per_component": B,
                "n_payload_symbols": link.n_sym, "k_symbols": cond["k"],
                "episodes": EVAL_EPISODES, "test_seed": config.TEST_SEED,
                "t_max": config.T_MAX, "non_frozen": True,
                "amendment": AMENDMENT,
                "protocol_note": ("uncoded QPSK reference; evaluation only; "
                                  "same episodes/noise as the paired harness"),
                "aggregate_uncoded": agg,
                "wall_s": round(wall, 1),
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=2, default=str))
            n_new += 1
            print(f"  EVAL {cond['variant']} k={cond['k']:>2} "
                  f"{cond['channel']:<8} snr={snr:5.1f} B={B:>2} "
                  f"SR={agg['success_rate']:.3f} ({wall:.1f}s)", flush=True)
    print(f"-- M5b finished: {n_new} new slots, wall={total_wall:.1f}s",
          flush=True)
    return 0


# ---------------------------------------------------------------------------
# M5c - cross-channel transfer (evaluation only)
# ---------------------------------------------------------------------------

def cmd_cross():
    print("== M5c cross-channel transfer (E=1000, evaluation only) ==",
          flush=True)
    n_new, total_wall = 0, 0.0
    for unit in m5c_units():
        src = m5c_source_cell(unit["trained_on"])
        if src is None:
            print(f"  SKIP train={unit['trained_on']}: source cell missing",
                  flush=True)
            continue
        ckpt = m5c_source_dir(src) / "final_ckpt.pt"
        if not ckpt.exists():
            print(f"  SKIP train={unit['trained_on']}: no checkpoint at "
                  f"{ckpt.relative_to(ROOT)}", flush=True)
            continue
        model = rw.ReconstructionDeepJSCC(k=M5C_K)
        payload = torch.load(ckpt, weights_only=False)
        model.load_state_dict(payload["model_state"])
        model.eval()
        for snr in SNR_GRID:
            out = m5c_path(unit, snr)
            if out.exists():
                continue
            c = Condition(k=M5C_K, channel=unit["eval_channel"], snr_db=snr,
                          t_max=config.T_MAX, bits_per_component=None,
                          episodes=EVAL_EPISODES, seed=config.TEST_SEED)
            t0 = time.perf_counter()
            with torch.no_grad():
                res = evaluate_condition(c, methods=["task"], task_model=model)
            wall = time.perf_counter() - t0
            total_wall += wall
            agg = res["task"]["aggregate"]
            rec = {
                "experiment": "M5", "step": "M5c",
                "trained_on": unit["trained_on"],
                "eval_channel": unit["eval_channel"],
                "k": M5C_K, "seed": M5C_SEED, "snr_db": snr,
                "episodes": EVAL_EPISODES, "test_seed": config.TEST_SEED,
                "t_max": config.T_MAX, "non_frozen": True,
                "amendment": AMENDMENT,
                "source_artifact": str(
                    (m5c_source_dir(src) / "final_ckpt.pt").relative_to(ROOT)),
                "protocol_note": ("evaluation-only cross-channel transfer; the "
                                  "checkpoint is not fine-tuned"),
                "aggregate_task": agg,
                "wall_s": round(wall, 1),
            }
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(rec, indent=2, default=str))
            n_new += 1
            print(f"  EVAL train={unit['trained_on']:<8} "
                  f"eval={unit['eval_channel']:<8} snr={snr:5.1f} "
                  f"SR={agg['success_rate']:.3f} ({wall:.1f}s)", flush=True)
    print(f"-- M5c finished: {n_new} new slots, wall={total_wall:.1f}s",
          flush=True)
    return 0


# ---------------------------------------------------------------------------
# status / finalize
# ---------------------------------------------------------------------------

def _state_no_mkdir(c):
    d = M5_ROOT / c.run_dirname()
    if not d.is_dir():
        return "missing"
    for kind, name in (("done", "DONE.json"), ("failed", "FAILED.json"),
                       ("running", "RUNNING.json")):
        if (d / name).exists():
            return kind
    return "missing"


def cmd_status():
    states = {}
    for c in m5a_cells():
        st = _state_no_mkdir(c)
        states[st] = states.get(st, 0) + 1
    n_a = sum(1 for u in m5a_units() for s in SNR_GRID
              if m5a_eval_path(u, s).exists())
    n_b = sum(1 for c in m5b_conditions() for s in SNR_GRID
              if m5b_path(c, s).exists())
    n_c = sum(1 for u in m5c_units() for s in SNR_GRID
              if m5c_path(u, s).exists())
    print(f"== M5 status ({M5_ROOT.relative_to(ROOT)}) ==")
    print(f"-- M5a train : {states} of {EXPECTED_TRAIN}")
    print(f"-- M5a eval  : {n_a}/{EXPECTED_SLOTS} slots")
    print(f"-- M5b eval  : {n_b}/{EXPECTED_UNCODED_SLOTS} slots")
    print(f"-- M5c eval  : {n_c}/{EXPECTED_CROSS_SLOTS} slots")
    return 0


def cmd_finalize():
    M5_ROOT.mkdir(parents=True, exist_ok=True)

    train_rows = []
    for c in m5a_cells():
        rep = M5_ROOT / c.run_dirname() / "train_report.json"
        rec = json.loads(rep.read_text()) if rep.exists() else {
            "exit_status": "missing"}
        train_rows.append({
            "wave_id": c.wave_id, "arm": c.tag, "k": c.k,
            "channel": c.channel, "seed": c.seed,
            "lambdas": list(c.lambdas),
            "exit_status": rec.get("exit_status", "missing"),
            "task_loss_final": rec.get("task_loss_final"),
            "nan_batches": rec.get("nan_batches"),
            "grads_finite": rec.get("grads_finite"),
            "wall_seconds": rec.get("wall_seconds"),
            "eval_after_success_rate": rec.get("eval_after_success_rate"),
        })

    pooled = {}
    n_a = 0
    for unit in m5a_units():
        for snr in SNR_GRID:
            p = m5a_eval_path(unit, snr)
            if not p.exists():
                continue
            n_a += 1
            rec = json.loads(p.read_text())
            for key, agg in rec.items():
                if key.startswith("aggregate_"):
                    tag = key[len("aggregate_"):]
                    row = pooled.setdefault(
                        (tag, unit["k"], unit["channel"], snr),
                        {"success_count": 0, "episodes": 0})
                    row["success_count"] += agg["success_count"]
                    row["episodes"] += agg["episodes"]
    pooled_rows = [{"arm": tag, "k": k, "channel": ch, "snr_db": snr,
                    "success_count": v["success_count"],
                    "episodes": v["episodes"],
                    "success_rate": (v["success_count"] / v["episodes"]
                                     if v["episodes"] else None)}
                   for (tag, k, ch, snr), v in sorted(pooled.items())]

    uncoded_rows = []
    n_b = 0
    for cond in m5b_conditions():
        for snr in SNR_GRID:
            p = m5b_path(cond, snr)
            if not p.exists():
                continue
            n_b += 1
            rec = json.loads(p.read_text())
            agg = rec["aggregate_uncoded"]
            uncoded_rows.append({
                "variant": cond["variant"], "k": cond["k"],
                "channel": cond["channel"], "snr_db": snr,
                "bits_per_component": cond["bits_per_component"],
                "success_count": agg["success_count"],
                "episodes": agg["episodes"],
                "success_rate": agg["success_rate"]})

    cross_rows = []
    n_c = 0
    for unit in m5c_units():
        for snr in SNR_GRID:
            p = m5c_path(unit, snr)
            if not p.exists():
                continue
            n_c += 1
            rec = json.loads(p.read_text())
            agg = rec["aggregate_task"]
            cross_rows.append({
                "trained_on": unit["trained_on"],
                "eval_channel": unit["eval_channel"], "snr_db": snr,
                "success_count": agg["success_count"],
                "episodes": agg["episodes"],
                "success_rate": agg["success_rate"]})

    done = sum(1 for r in train_rows if r["exit_status"] == "complete")
    out = M5_ROOT / "step_m5_summary.json"
    out.write_text(json.dumps({
        "experiment": "M5",
        "amendment": AMENDMENT,
        "status": ("complete"
                   if done == EXPECTED_TRAIN
                   and n_a == EXPECTED_SLOTS
                   and n_b == EXPECTED_UNCODED_SLOTS
                   and n_c == EXPECTED_CROSS_SLOTS else "incomplete"),
        "design": {
            "m5a": {"cells": EXPECTED_TRAIN, "units": EXPECTED_UNITS,
                    "slots": EXPECTED_SLOTS, "arms": M5A_ARMS,
                    "k": list(M5A_KS), "seed": M5A_SEED,
                    "steps": config.TASK_TRAIN_STEPS},
            "m5b": {"cells": EXPECTED_UNCODED, "slots": EXPECTED_UNCODED_SLOTS,
                    "k": list(M5B_KS)},
            "m5c": {"units": EXPECTED_CROSS, "slots": EXPECTED_CROSS_SLOTS,
                    "k": M5C_K, "source_tag": M5C_TAG},
        },
        "protocol_note": (f"M5 diagnostic: E={EVAL_EPISODES} per unit; NOT "
                          "1:1 comparable to frozen M1/M2 (E=5000). Steps, "
                          "architecture, seeds and evaluation stream frozen; "
                          "only the loss weighting (M5a) and the reference "
                          "links (M5b/c) are new."),
        "m5a_train_rows": train_rows,
        "m5a_pooled_rows": pooled_rows,
        "m5b_uncoded_rows": uncoded_rows,
        "m5c_cross_rows": cross_rows,
        "counts": {"m5a_train_done": done, "m5a_train_expected": EXPECTED_TRAIN,
                   "m5a_slots": n_a, "m5a_slots_expected": EXPECTED_SLOTS,
                   "m5b_slots": n_b,
                   "m5b_slots_expected": EXPECTED_UNCODED_SLOTS,
                   "m5c_slots": n_c,
                   "m5c_slots_expected": EXPECTED_CROSS_SLOTS},
    }, indent=2, default=str))
    print(f"wrote {out.relative_to(ROOT)} ({done}/{EXPECTED_TRAIN} train, "
          f"{n_a}/{EXPECTED_SLOTS} M5a slots, "
          f"{n_b}/{EXPECTED_UNCODED_SLOTS} M5b slots, "
          f"{n_c}/{EXPECTED_CROSS_SLOTS} M5c slots)")
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    simple = {"verify": cmd_verify, "status": cmd_status,
              "uncoded": cmd_uncoded, "cross": cmd_cross,
              "finalize": cmd_finalize}
    if cmd in simple:
        return simple[cmd]()
    if cmd == "smoke":
        return cmd_smoke(int(rest[0]) if rest else None)
    if cmd in ("train", "evaluate"):
        lane = next((a for a in rest if not a.startswith("--")), None)
        return {"train": cmd_train, "evaluate": cmd_evaluate}[cmd](
            lane, [a for a in rest if a.startswith("--")])
    print(f"unknown command {cmd!r}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
