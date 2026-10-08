def _smoke_test() -> int:
    """Minimal deterministic end-to-end verification (NO experiments run)."""
    print("[smoke] device:", get_device())
    set_global_seed(0)

    # -- SECTION 1: frozen configuration self-consistency -------------------
    validate()
    assert digital_packet(12, 1)["n_padding_symbols"] == 0
    try:
        digital_packet(3, 1)
        raise AssertionError("k=3,B=1 must be infeasible")
    except InfeasibleDigitalConfig:
        pass
    print("[smoke] sec1 frozen config + digital packet arithmetic OK")

    # -- SECTION 2: normalization round-trip ---------------------------------
    s = torch.tensor([[0.0, 0.0, 0.0, 0.0, 4.0, -4.0],
                      [-4.0, 4.0, 1.5, -1.5, 0.0, 0.0]])
    s_bar = normalize(s)
    assert torch.isfinite(s_bar).all()
    assert torch.allclose(denormalize(s_bar), s, atol=1e-6)
    print("[smoke] sec2 normalization round-trip OK")

    # -- SECTION 3: complex conversion, power norm, AWGN, Rayleigh -----------
    k = 6
    x = torch.randn(4, 2 * k)
    assert torch.allclose(to_real(to_complex(x)), x, atol=1e-6)
    z = power_normalize(to_complex(x))
    assert torch.allclose(
        z.abs().pow(2).sum(dim=-1),
        torch.full((4,), float(k) * TX_POWER), atol=1e-4)
    y_awgn = awgn_channel(z, snr_db=10.0)
    assert y_awgn.shape == z.shape and torch.isfinite(y_awgn).all()
    y_ray, h_used = rayleigh_channel(z, snr_db=10.0, return_h=True)
    x_eq = equalize_zf(y_ray, h_used)
    assert torch.isfinite(x_eq).all()
    print("[smoke] sec3 AWGN + Rayleigh + power normalization OK")

    # -- SECTION 4: DeepJSCC full path (shared task/recon container) ---------
    model = ReconstructionDeepJSCC(k=k)
    s_hat = model.forward(s, snr_db=10.0, channel="awgn",
                          noise=torch.complex(torch.randn(2, k),
                                              torch.randn(2, k)))
    assert s_hat.shape == (2, 6) and torch.isfinite(s_hat).all()
    print("[smoke] sec4 DeepJSCC encoder->channel->decoder forward OK")

    # -- SECTION 7: losses ----------------------------------------------------
    assert float(reconstruction_loss(s_bar, s_bar)) == 0.0
    pos = torch.randn(2, 4, 2)
    ctl = torch.randn(2, 3, 2)
    vel = torch.randn(2, 4, 2)
    goal0 = torch.zeros(2, 2)
    tl = task_loss(pos, ctl, vel, goal0)
    assert torch.isfinite(tl["total"]) and torch.isfinite(tl["L_T"])
    print("[smoke] sec7 task + reconstruction losses OK")

    # -- SECTION 5: digital baseline encode -> channel -> decode -------------
    base = DigitalBaseline(k=12, bits_per_component=1)
    s_phys = denormalize(s_bar)
    out_d = base.forward(s_phys, snr_db=10.0, channel="awgn",
                         noise=torch.complex(torch.randn(2, 12),
                                             torch.randn(2, 12)))
    assert out_d.shape == (2, 6) and torch.isfinite(out_d).all()
    print("[smoke] sec5 digital baseline forward path OK")

    # -- SECTION 6: UAV dynamics, controller, environment ---------------------
    p = torch.zeros(2, 2)
    v = torch.zeros(2, 2)
    u = torch.tensor([[2.0, -2.0], [2.0, -2.0]])
    p1_, v1_ = step(p, v, u)
    assert torch.allclose(p1_, p + DT * v + 0.5 * DT ** 2 * u, atol=1e-6)
    u_pd = pd_controller(torch.cat([p, v, torch.zeros(2, 2)], dim=-1))
    assert (u_pd.abs() <= U_MAX + 1e-6).all()
    p0, v0, g0 = NavigationEnv().sample_starts(
        4, generator=derive_generator("episodes", 0))
    assert p0.shape == (4, 2) and g0.shape == (4, 2)
    print("[smoke] sec6 dynamics + PD controller + environment OK")

    # -- SECTION 9: paired oracle rollout + statistics sanity -----------------
    cond = Condition(k=2, channel="awgn", snr_db=10.0, episodes=4,
                     seed=TEST_SEED)
    real = generate_realizations(cond, 4)
    met = run_method(cond, real, estimate_fn=None)
    assert met["success"].dtype == torch.bool and met["success"].numel() == 4
    ci = wilson_ci_95(met["success"])
    assert 0.0 <= ci["wilson_low"] <= ci["wilson_high"] <= 1.0
    mc = mcnemar_test(torch.tensor([1, 1, 1, 0, 0]),
                      torch.tensor([1, 1, 0, 0, 0]))
    assert mc["n01"] == 1 and mc["n10"] == 0 and 0.0 <= mc["p_value"] <= 1.0
    assert mcnemar_test(torch.tensor([1, 0]),
                        torch.tensor([0, 1]))["p_value"] == 1.0
    assert paired_t_test(torch.tensor([1.0, 2.0]),
                         torch.tensor([1.0, 2.0]))["p_value"] == 1.0
    print("[smoke] sec9 paired oracle rollout + Wilson/McNemar/paired-t OK")

    # -- SECTION 10: frozen experiment matrices --------------------------------
    cells = expected_cells()
    assert cells["M1_training_runs"] == 36
    assert cells["M1_eval_conditions"] == 132
    assert cells["M2_neural_training_runs"] == 60
    assert cells["M2_digital_eval_only_seeds"] == 30
    assert cells["M2_eval_conditions"] == 150
    assert cells["M3_infeasible_cells"] == 3
    assert cells["M4_secondary_eval_conditions"] == 10
    print("[smoke] sec10 frozen experiment matrices:", dict(cells))

    print("[smoke] ALL CHECKS PASSED")
    return 0
