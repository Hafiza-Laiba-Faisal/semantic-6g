"""Step 7 evaluation harness tests (Gates A-H).

Run from the repository root:
    .venv/Scripts/python.exe -m tests.test_evaluation
"""

from __future__ import annotations

import math
import sys

import torch

from project import config
from project.baselines.digital import DigitalBaseline
from project.evaluation.evaluate import (
    Condition,
    evaluate_condition,
    generate_realizations,
    run_method,
)
from project.evaluation.metrics import (
    aggregate_episode_metrics,
    per_episode_metrics,
)
from project.evaluation.statistics import (
    mcnemar_test,
    mean_std,
    paired_t_test,
    success_count,
    wilson_ci_95,
)
from project.models.reconstruction_jscc import ReconstructionDeepJSCC

SUMMARY = []


def check(name, ok, detail=""):
    SUMMARY.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  [{name}]" +
          (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
# Gate A - hand-computed tiny case
# ---------------------------------------------------------------------------

def test_gate_a():
    # two episodes, T = 3; positions walked by hand
    pos = torch.tensor([
        [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [2.5, 0.0]],   # ep1
        [[6.0, 0.0], [5.0, 0.0], [4.0, 0.0], [3.2, 0.0]],   # ep2
    ])
    u = torch.tensor([
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        [[-1.0, 0.0], [-1.0, 0.0], [-0.5, 0.0]],
    ])
    vel = torch.zeros(2, 4, 2)
    goal = torch.tensor([[2.6, 0.0], [3.0, 0.0]])
    r_g = 0.5
    m = per_episode_metrics(pos, u, vel, goal, r_g)

    # ep1: d0=2.6, d1=1.6, d2=0.6, d3=0.1 -> success at t=3, ttg=3
    #      avg = (1.6+0.6+0.1)/3 = 2.3/3 ; Eu = 3 ; dT = 0.1
    # ep2: d0=3.0, d1=2.0, d2=1.0, d3=0.2 -> success at t=3, ttg=3
    #      avg = (2+1+0.2)/3 = 3.2/3 ; Eu = 1+1+0.25 = 2.25 ; dT = 0.2
    ok_success = m["success"].tolist() == [True, True]
    ok_ttg = m["time_to_goal"].tolist() == [3, 3]
    ok_fd = all(math.isclose(float(m["final_distance"][i]), v, abs_tol=1e-7)
                for i, v in enumerate([0.1, 0.2]))
    ok_avg = all(math.isclose(float(m["avg_distance"][i]), v, abs_tol=1e-7)
                 for i, v in enumerate([2.3 / 3, 3.2 / 3]))
    ok_eu = all(math.isclose(float(m["control_effort"][i]), v, abs_tol=1e-7)
                for i, v in enumerate([3.0, 2.25]))
    # MSE with a perfectly estimated trajectory = 0; with +0.3 offset on the
    # normalized state every step: (1/(6*3)) * 3 * 6 * 0.09 = 0.09
    s_bar = torch.rand(2, 3, 6)
    mse0 = per_episode_metrics(pos, u, vel, goal, r_g,
                               s_hat_bar=s_bar, s_bar=s_bar)["mse"]
    mse_off = per_episode_metrics(pos, u, vel, goal, r_g,
                                  s_hat_bar=s_bar + 0.3, s_bar=s_bar)["mse"]
    ok_mse = (float(mse0.abs().max()) < 1e-7
              and all(math.isclose(float(v), 0.09, abs_tol=1e-6) for v in mse_off))
    # failure sentinel: a trajectory that never reaches the goal
    pos_miss = pos.clone()
    pos_miss[1, -1, 0] = 2.4            # d3 = |2.4-3.0| = 0.6 > r_g; d0..d2 also > r_g
    m_miss = per_episode_metrics(pos_miss, u, vel, goal, r_g)
    ok_sentinel = (not bool(m_miss["success"][1])) \
        and int(m_miss["time_to_goal"][1]) == 4      # T+1 = 4
    check("A", ok_success and ok_ttg and ok_fd and ok_avg and ok_eu
          and ok_mse and ok_sentinel,
          f"success={ok_success}, ttg={ok_ttg}, fd={ok_fd}, avg={ok_avg}, "
          f"Eu={ok_eu}, mse={ok_mse}, sentinel={ok_sentinel}")


# ---------------------------------------------------------------------------
# Gate B - Wilson CI vs hand computation
# ---------------------------------------------------------------------------

def test_gate_b():
    z = 1.959963984540054
    for k, n in ((7, 50), (0, 10), (10, 10)):
        w = wilson_ci_95([1] * k + [0] * (n - k))
        p = k / n
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        ok = (math.isclose(w["wilson_low"], max(center - half, 0.0), abs_tol=1e-12)
              and math.isclose(w["wilson_high"], min(center + half, 1.0), abs_tol=1e-12))
        check(f"B/k={k},n={n}", ok,
              f"CI=[{w['wilson_low']:.6f}, {w['wilson_high']:.6f}]")
    w0 = wilson_ci_95([])
    check("B/n=0", math.isnan(w0["wilson_low"]), "n=0 handled safely")


# ---------------------------------------------------------------------------
# Gate C - mean/std vs reference
# ---------------------------------------------------------------------------

def test_gate_c():
    x = [0.5, 1.25, -2.0, 3.75, 0.0]
    r = mean_std(x)
    ref_m = sum(x) / len(x)
    ref_v = sum((xi - ref_m) ** 2 for xi in x) / (len(x) - 1)
    check("C", math.isclose(r["mean"], ref_m, abs_tol=1e-12)
          and math.isclose(r["std"], math.sqrt(ref_v), abs_tol=1e-12),
          f"mean={r['mean']:.6f}, std={r['std']:.6f}")


# ---------------------------------------------------------------------------
# Gate D - paired realization identity (HARD GATE)
# ---------------------------------------------------------------------------

def test_gate_d():
    # PAIRED condition: ALL methods share the SAME k (matched budget, audit
    # 8.6) - so the per-symbol noise width matches for neural and digital.
    torch.manual_seed(0)
    cond = Condition(k=54, channel="rayleigh", snr_db=10.0, t_max=6,
                     episodes=8, bits_per_component=8, seed=555)
    real = generate_realizations(cond, 8)
    recon = ReconstructionDeepJSCC(k=54)
    task = ReconstructionDeepJSCC(k=54)
    digital = DigitalBaseline(k=54, bits_per_component=8)
    # wrap each method to CAPTURE the realizations it received
    captures = {}

    def make_cap(name, fn):
        def est(s, noise_c, h_c):
            captures.setdefault(name, []).append(
                (noise_c.clone(), None if h_c is None else h_c.clone()))
            return fn(s, noise_c, h_c)
        return est

    from project.evaluation.evaluate import _digital_estimate_factory, \
        _neural_estimate_factory
    from project.evaluation import evaluate as ev

    orig = ev.run_method

    def cap_run(cond_, real_, estimate_fn):
        return orig(cond_, real_, estimate_fn)

    # run each method through run_method with capturing wrappers
    for name, model in (("recon", recon), ("task", task)):
        est0 = _neural_estimate_factory(model, cond.channel, cond.snr_db)
        cap_run(cond, real, make_cap(name, est0))
    est_d = _digital_estimate_factory(digital, cond.channel, cond.snr_db)
    cap_run(cond, real, make_cap("digital", est_d))
    cap_run(cond, real, None)  # oracle (captures nothing; noise unused)

    n_ref, h_ref = captures["recon"][0]
    for name in ("task", "digital"):
        n_i, h_i = captures[name][0]
        same_n = bool((n_i == n_ref).all())
        same_h = bool((h_i == h_ref).all()) if h_ref is not None else (h_i is None)
        check(f"D/{name}", same_n and same_h,
              f"noise bitwise={same_n}, h bitwise={same_h}")
    # oracle path: verify the materialized realizations themselves are stable
    real2 = generate_realizations(cond, 8)
    check("D/materialization-deterministic",
          bool((real["noise"] == real2["noise"]).all())
          and bool((real["h"] == real2["h"]).all()))


# ---------------------------------------------------------------------------
# Gate E - determinism of the full harness
# ---------------------------------------------------------------------------

def test_gate_e():
    torch.manual_seed(1)
    cond = Condition(k=2, channel="awgn", snr_db=8.0, t_max=8,
                     episodes=12, seed=777)
    recon = ReconstructionDeepJSCC(k=2)
    r1 = evaluate_condition(cond, methods=["oracle", "reconstruction"],
                            recon_model=recon)
    r2 = evaluate_condition(cond, methods=["oracle", "reconstruction"],
                            recon_model=recon)
    same = all(
        all(torch.equal(r1[m]["metrics"][key], r2[m]["metrics"][key])
            for key in ("success", "final_distance", "avg_distance",
                        "control_effort", "time_to_goal", "mse"))
        for m in ("oracle", "reconstruction"))
    check("E", same, "bitwise-identical metric tables across reruns")


# ---------------------------------------------------------------------------
# Gate F - interface parity + infeasible status
# ---------------------------------------------------------------------------

def test_gate_f():
    torch.manual_seed(2)
    cond = Condition(k=3, channel="awgn", snr_db=10.0, t_max=6,
                     episodes=8, seed=888)
    res = evaluate_condition(
        cond,
        methods=["oracle", "reconstruction", "task", "digital"],
        recon_model=ReconstructionDeepJSCC(k=3),
        task_model=ReconstructionDeepJSCC(k=3),
        digital=DigitalBaseline(k=54, bits_per_component=8))
    keys = ("success", "final_distance", "avg_distance", "control_effort",
            "min_distance", "time_to_goal", "mse")
    # k = 3 with no B -> digital is correctly reported as infeasible (frozen
    # framing); the schema check applies to every method that produced metrics
    ok_inf = res["digital"].get("status") == "infeasible"
    metric_methods = [m for m in res if "metrics" in res[m]]
    ok_schema = (len(metric_methods) == 3 and
                 all(key in res[m]["metrics"] for m in metric_methods
                     for key in keys))
    ok_agg = all(all(k2 in res[m]["aggregate"] for k2 in
                     ("success_count", "success_rate", "ttg_success_mean"))
                 for m in metric_methods)
    check("F/schema-parity", ok_schema and ok_agg,
          f"metric methods: {metric_methods}; digital k=3 -> {res['digital'].get('status')}")
    # explicit infeasible status at k in {1,2,3}
    cond_low = Condition(k=2, channel="awgn", snr_db=10.0, t_max=4,
                         episodes=4, bits_per_component=1, seed=999)
    res_low = evaluate_condition(cond_low, methods=["oracle", "digital"],
                                 recon_model=ReconstructionDeepJSCC(k=2))
    d = res_low["digital"]
    check("F/infeasible-status",
          d.get("status") == "infeasible" and "12" in d.get("reason", ""),
          str(d.get("reason", ""))[:60])


# ---------------------------------------------------------------------------
# Gate G - statistical pairing
# ---------------------------------------------------------------------------

def test_gate_g():
    # McNemar: pairing matters - only discordant pairs count
    a = [1, 1, 0, 0, 1, 0, 1, 0]
    b = [1, 0, 0, 0, 1, 1, 1, 0]
    m = mcnemar_test(a, b)
    # discordant: (1,0) at idx1, (0,1) at idx5 -> n01=1, n10=1 -> chi2=0, p=1
    ok_mcn = (m["n01"] == 1 and m["n10"] == 1
              and math.isclose(m["p_value"], 1.0, abs_tol=1e-12))
    # asymmetric discordance gives the expected chi2
    a2 = [1] * 9 + [0] * 1
    b2 = [0] * 9 + [1] * 1
    m2 = mcnemar_test(a2, b2)
    chi2_ref = (8 - 1) ** 2 / 10       # |9-1|-1 = 7, /10
    ok_mcn2 = (m2["n01"] == 9 and m2["n10"] == 1
               and math.isclose(m2["chi2"], chi2_ref, abs_tol=1e-9)
               and 0.0 < m2["p_value"] < 0.05)
    # paired t: pairing preserved - identical lists with opposite orderings
    # MUST produce exactly negated statistics
    x = [1.0, 2.0, 3.0, 4.0]
    y = [1.5, 2.5, 2.0, 5.0]
    t1 = paired_t_test(x, y)
    t2 = paired_t_test(y, x)
    ok_t = (math.isclose(t1["t_stat"], -t2["t_stat"], abs_tol=1e-12)
            and math.isclose(t1["mean_diff"], -t2["mean_diff"], abs_tol=1e-12)
            and math.isclose(t1["p_value"], t2["p_value"], abs_tol=1e-12))
    # constant nonzero difference -> p ~ 0
    t3 = paired_t_test([1.0] * 6, [2.0] * 6)
    ok_t2 = t3["t_stat"] == float("inf") and t3["p_value"] == 0.0
    check("G", ok_mcn and ok_mcn2 and ok_t and ok_t2,
          f"mcnemar={ok_mcn and ok_mcn2}, paired-t={ok_t and ok_t2}")


# ---------------------------------------------------------------------------
# Gate H - regression (in-process, previous suites)
# ---------------------------------------------------------------------------

def test_gate_h():
    import tests.test_channels as tc
    tc.SUMMARY.clear()
    tc.test_conversion(); tc.test_power_normalization(); tc.test_awgn_variance()
    tc.test_awgn_snr(); tc.test_rayleigh_stats(); tc.test_block_fading()
    tc.test_equalization(); tc.test_repro_and_conventions()
    check("REG/Step-3", all(tc.SUMMARY), f"{sum(tc.SUMMARY)}/{len(tc.SUMMARY)}")

    import tests.test_step02 as ts
    fns = [ts.test_config, ts.test_normalization_roundtrip,
           ts.test_normalization_endpoints_and_clip, ts.test_dynamics_closed_form,
           ts.test_dynamics_autograd, ts.test_velocity_clip,
           ts.test_controller_formula, ts.test_controller_saturation_and_gradient,
           ts.test_env_success_semantics, ts.test_env_exit_freeze_and_determinism,
           ts.test_env_sampling_and_nans]
    try:
        for fn in fns:
            fn()
        check("REG/Step 0-2", True, "11/11")
    except AssertionError:
        check("REG/Step 0-2", False)

    import tests.test_deepjscc as td
    td.SUMMARY.clear()
    td.test_channel_use_count(); td.test_decoder_input_order_parity()
    td.test_shapes(); td.test_power_through_model(); td.test_gradient_flow()
    check("REG/Step-4 structural", all(td.SUMMARY),
          f"{sum(td.SUMMARY)}/{len(td.SUMMARY)}")

    import tests.test_task_oriented as tt
    tt.SUMMARY.clear()
    tt.test_a_loss_algebra(); tt.test_b_rollout_dims(); tt.test_c_zero_error()
    tt.test_d_gradient_flow(); tt.test_e_controller_dynamics()
    check("REG/Step-5 core", all(tt.SUMMARY), f"{sum(tt.SUMMARY)}/{len(tt.SUMMARY)}")

    import tests.test_digital as tg
    tg.SUMMARY.clear()
    tg.test_gate_a(); tg.test_gate_c()   # B/D covered by channel suite; E in Gate A/F here
    check("REG/Step-6 quantizer+viterbi", all(tg.SUMMARY),
          f"{sum(tg.SUMMARY)}/{len(tg.SUMMARY)}")


if __name__ == "__main__":
    print("=" * 72)
    print("STEP 7 EVALUATION HARNESS TESTS (Gates A-H)")
    print("=" * 72)
    test_gate_a()
    test_gate_b()
    test_gate_c()
    test_gate_d()
    test_gate_e()
    test_gate_f()
    test_gate_g()
    test_gate_h()
    n_pass, n_all = sum(SUMMARY), len(SUMMARY)
    print("\n" + "=" * 72)
    print(f"SUMMARY: {n_pass}/{n_all} checks passed")
    sys.exit(0 if n_pass == n_all else 1)
