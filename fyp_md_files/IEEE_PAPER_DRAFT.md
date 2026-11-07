# When Does Task-Oriented DeepJSCC Beat a Coded Digital Link? A Matched-Budget Study for UAV Navigation

**AUTHORS:** Areesha Sohail, Amna Shahid, and Hafiza Laiba Faisal

**AFFILIATION:** Department of Electrical Engineering, Lahore College for Women University (LCWU), Lahore, Pakistan. Supervisor: Ismat Hira.

**ABSTRACT:** Task-oriented joint source-channel coding promises to spend scarce channel uses on mission-relevant information rather than on reconstruction. Whether that promise survives a matched-budget comparison with a modern coded link, and whether it transfers to a realistic aerial channel, is less clear. We present a controlled, simulation-only study in which a six-dimensional UAV kinematic state is transmitted over AWGN and slow Rayleigh fading at k = 12–54 channel uses per control step. Three links share identical episodes, noise, fading, and bandwidth: a convolutional-coded QPSK link, a reconstruction-oriented DeepJSCC autoencoder, and a task-oriented DeepJSCC trained on the navigation rollout. We find a bandwidth crossover: neural joint source-channel coding wins at k ≤ 18, the coded link wins at k ≥ 24, and the two are effectively tied at k = 54. The frozen task loss, which omits control and velocity penalties, fails at 2–5% success; adding both penalties recovers 0.997 at k = 12 and 0.993–0.999 at k = 18 and 54. A term-removal study localises the effect to the control-effort penalty alone, and gradient measurements show a conditioning effect of roughly fourteen orders of magnitude rather than a capacity effect. On a 3GPP UMa-aerial link the shaped policy needs 15–21 dB less transmit SNR than the digital link across twelve geometries. A Clarke/Jakes Doppler sweep and a 5G NR LDPC BG2 comparator do not move the conclusion.

**INDEX TERMS:** DeepJSCC, task-oriented communication, UAV navigation, joint source-channel coding, Rayleigh fading, 3GPP UMa-aerial, 5G NR LDPC.

## I. INTRODUCTION

Small unmanned aerial vehicles (UAVs) rely on a wireless link to a ground station for the state information that drives navigation. That link is rarely ideal: it carries thermal noise and, because the platform moves through multipath, small-scale fading. Classical bit-level pipelines separate source compression from channel coding and aim to reproduce the transmitted signal. Their performance is flat until the channel falls below a design threshold and then collapses abruptly, the well-known cliff effect [4].

Deep joint source-channel coding (DeepJSCC) replaces that separation with a pair of jointly trained networks that map source symbols directly to complex channel symbols, and it degrades gracefully under AWGN and slow fading [4], [5]. Most of this literature is reconstruction-oriented: the pair minimises a distortion measure, independent of what the receiver will do with the recovered signal. Task-oriented communication takes the opposite view. When the purpose of transmission is a decision or a control action, the link should be optimised against that downstream task [6], [7]. Task-oriented source-channel coding has been demonstrated for closed-loop control in autonomous driving [8]–[10] and studied for UAV command transmission, waypoint delivery, and swarm tracking [11]–[14], [19].

Two questions remain open. First, does a task-oriented DeepJSCC link actually beat a *modern* digital link when the comparison is matched on episodes, channel, SNR, and channel uses, including the packet-feasibility constraint that a separated design must satisfy? Second, does the result survive a realistic aerial large-scale link and a time-correlated fading process? This paper answers both with a controlled simulation study, and it keeps the negative results visible.

Our contributions are:

- **A matched-budget benchmark for UAV state transmission.** Three links share bitwise-identical episodes, noise, fading, and bandwidth. The digital packet invariant 6B + 6 ≤ k is enforced, never relaxed: at k = 1–3 no matched-budget digital packet exists (M3), and k = 12 is the first feasible point.
- **A measured bandwidth crossover, defended against four comparators.** The neural link wins at k ≤ 18 and the coded link at k ≥ 24. The ordering survives a convolutional code, an SC polar code, the full 5G NR polar chain, and 5G NR LDPC base graph 2; a payload-maximal uncoded link improves the low-bandwidth end but does not overtake the neural arm below k = 54.
- **A diagnosis of why task-oriented training collapses without shaping.** The frozen loss fails at 2–5% success; control and velocity penalties recover 0.993–0.997. Removing the control-effort term alone reproduces the failure, and the encoder gradient falls from about 10^14 to O(1), localising the mechanism to optimisation conditioning.
- **A 3GPP UMa-aerial assessment and two robustness checks.** Across twelve geometries the shaped policy needs 15–21 dB less transmit SNR for 0.90 success. A Clarke/Jakes Doppler sweep moves the neural arm by at most 0.003, and the LDPC comparator is statistically indistinguishable from the convolutional baseline at k = 12–30.

The claim is deliberately conditional. We do not propose a new communication paradigm; we characterise when a task-oriented joint code is the better use of a constrained link, and we state where it is not. The study is simulation-only, two-dimensional, and CPU-based; no flight test or software-defined radio is involved.

## II. RELATED WORK

DeepJSCC was introduced for wireless image transmission and shown to outperform separation-based digital transmission at low SNR and small bandwidth under AWGN and slow Rayleigh fading [4]. Bandwidth-agile variants adapt the channel-use allocation to the available bandwidth [5], and later work extends the idea to MIMO, OFDM, and attention-based architectures [15]–[17]. These systems optimise reconstruction fidelity. Our reconstruction baseline is deliberately of this type, so that the comparison isolates the training objective rather than the architecture.

Task-oriented communication formalises the objective through an information bottleneck or a task loss [6], [7]. For control systems, task-oriented source-channel coding has been applied to autonomous driving with camera imagery [8] and extended to joint communication–computing–control co-design [10] and to the alignment of task- and reconstruction-oriented objectives [9]. A doctoral study develops the edge-robotics framing in detail [24]. UAV-specific work includes semantics-aware command and control [13], goal-oriented waypoint transmission with value-of-information scheduling [14], decentralised semantic communication for swarm tracking [12], and visual navigation with edge–aerial collaboration [11]. Shiri et al. study learned UAV path planning over a wireless link [19]. None of these provides a matched-budget comparison of a task-trained DeepJSCC link against a packet-feasible coded link on a compact kinematic state, which is the comparison we run.

For the aerial channel we use the large-scale layer of 3GPP TR 36.777 and TR 38.901 [1], [2]; the surveys of Khuwaja et al. [18] motivate the LoS/NLoS mixture and the altitude dependence. The digital comparators follow the 5G NR channel-coding specification [3] for polar and LDPC constructions. Our 3GPP layer is a link-budget and dispersion model, not a full channel simulator, and we say so explicitly.

## III. SYSTEM MODEL AND METHOD

### A. UAV Kinematics and Controller

The navigation state at discrete step t is

*s*(t) = [*x*(t), *y*(t), *v_x*(t), *v_y*(t), *x_g*, *y_g*] ∈ R^6,

where the first four components are position and velocity and the last two are the fixed goal. The plant is a two-dimensional double integrator,

*p*(t+1) = *p*(t) + Δt *v*(t) + ½ Δt² *u*(t),
*v*(t+1) = *v*(t) + Δt *u*(t),

with Δt = 0.1 s, horizon T = 100 steps, workspace ±8 m, success radius 0.5 m, speed limit 5 m/s, and acceleration limit 2 m/s². The controller is fixed for every method,

*u*(t) = clip(−K_p (*p*(t) − *p*_g) − K_v *v*(t), ±*u*_max),

with K_p = 1.0, K_v = 2.0, and u_max = 2 m/s²; the damping ratio is unity. Fixing the controller ensures that a performance difference is attributable to the communication link, not to a co-adapted control policy. Each state component is mapped to [−1, 1] by a fixed affine transform whose ranges come from the configuration, never from data.

### B. Channel Models

At every step the transmitter maps *s*(t) to k complex symbols *z*(t) under an average power constraint P = 1, normalising the encoder output to unit average symbol power. The AWGN channel is *y* = *z* + *n* with *n* ~ CN(0, σ²I) and σ² = P·10^(−SNR/10). The Rayleigh channel is *y* = *h z* + *n* with one gain *h* ~ CN(0, 1) per codeword, held over its k symbols; the receiver has perfect CSI and applies zero-forcing equalisation. These idealisations are standard in the DeepJSCC literature and are applied identically to all three links.

The aerial extension uses the 3GPP UMa-aerial large-scale layer [1], [2]. For height h and ground distance d_2D, with d_3D = sqrt(d_2D² + h²), the LoS probability is

P_LoS = d_1/d_2D + exp(−d_2D/p_1)(1 − d_1/d_2D),

with p_1 = 4300 log10(h) − 3800 and d_1 = max(460 log10(h) − 700, 18), and the path losses are

PL_LoS = 28.0 + 22 log10(d_3D) + 20 log10(f_c),
PL_mean = P_LoS·PL_LoS + (1 − P_LoS)·PL_NLoS,

with the NLoS branch as specified in TR 38.901 [2] and f_c in GHz. At the pinned geometry d_2D = 500 m, h = 50 m, f_c = 2 GHz, the model gives P_LoS = 0.8887 and PL_mean = 95.63 dB. Under perfect-CSI zero forcing a deterministic gain is exactly an effective-SNR shift, so the evaluation needs no retraining: SR(SNR_tx) = P_LoS·SR(SNR_tx − PL_LoS) + (1 − P_LoS)·SR(SNR_tx − PL_NLoS). The dispersion-training variant instead draws one LoS/NLoS state, one log-normal shadow term (4 dB LoS, 6 dB NLoS), and one block-fading gain per episode. The layer contains no Doppler, no time-correlated shadowing, and no altitude-dependent small-scale fading.

### C. Three Communication Links

**Digital link.** Each state component is scalar-quantised to B bits, framed with a zero tail, convolutionally encoded (K = 7, rate 1/2), Gray-mapped to QPSK, and decoded with soft-decision Viterbi. The frozen pairs satisfy 6B + 6 = k exactly, so B = 1, 2, 3, 4, 8 at k = 12, 18, 24, 30, 54. A pair whose payload would exceed k is declared infeasible rather than widened. At k = 1–3 even B = 1 needs 12 uses, so the budget is short by 11, 10, and 9 uses respectively; this is the M3 boundary, a structural statement rather than a measured failure.

**Reconstruction DeepJSCC.** A 6 → 128 → 128 → 2k encoder with PReLU activations produces k complex symbols; the decoder mirrors it. It is trained on L_recon = (1/6)·E[||ŝ − s||²].

**Task-oriented DeepJSCC.** The same architecture, trained end-to-end through the differentiable channel on the navigation rollout loss.

### D. Task Loss and Shaping

The task loss is L = λ_d L_d + λ_u L_u + λ_v L_v + λ_T L_T, where L_d and L_T are the mean squared distance to the goal and the terminal squared distance, L_u is the mean squared control effort, and L_v the mean squared velocity. The frozen protocol used (λ_d, λ_u, λ_v, λ_T) = (1, 0, 0, 2). The diagnostic arm uses (1, 0.1, 0.1, 2). The control and velocity penalties are the subject of the ablation in Section IV-B.

### E. Protocol, Metrics, and Statistics

The reconstruction model trains for 5,000 steps with batch 4,096; the task model for 3,000 steps with 256 episodes of horizon 100. Both use Adam [20] with learning rate 10⁻³ and gradient clipping at 1.0; the training SNR is drawn uniformly from 0 to 20 dB; the final checkpoint is evaluated, not a best-validation one. Training seeds are {42, 43, 44} and the test seed is 10042. The main matrix evaluates k ∈ {12, 18, 24, 30, 54} at SNR ∈ {0, 5, 10, 15, 20} dB with 5,000 episodes per unit on AWGN and Rayleigh fading. Episodes are paired: initial states, goals, noise, and fading are generated once per unit and shared by all methods. Success is a binary indicator (final distance ≤ 0.5 m within T), compared with the McNemar test [22]; Wilson 95% intervals [23] are reported and continuous metrics use paired t-tests. Extension blocks run at 1,000 episodes per unit and are labelled separately rather than pooled. All code is CPU-only, implemented in PyTorch [21], and deterministic; the suite contains 111 passing tests. Every mean reported below is recomputed from the stored per-SNR JSON artifacts rather than transcribed by hand.

## IV. RESULTS

### A. The Bandwidth Crossover

Table I reports the main AWGN result, pooled over three seeds (15,000 episodes per condition). At k = 12 the reconstruction link reaches 0.242 against 0.062 for the coded link, roughly a fourfold advantage. At k = 18 the gap narrows (0.254 against 0.164). The ordering reverses between k = 18 and k = 24: at k = 24 the coded link reaches 0.306 against 0.233 for reconstruction, and by k = 54 it reaches 0.995 while reconstruction stays near 0.246. On Rayleigh fading the same ordering holds, with the k = 54 digital link at 0.941 and reconstruction at 0.258. The reconstruction MSE is flat in k (0.049–0.054), which shows that the neural decoder is limited by its objective, not by the number of channel uses.

**Table I.** Success rate versus bandwidth on AWGN, pooled over three seeds (E = 5,000 per unit).

| k | Reconstruction | Task (frozen) | Coded digital |
|---:|---:|---:|---:|
| 12 | **0.242** | 0.028 | 0.062 |
| 18 | **0.254** | 0.018 | 0.164 |
| 24 | 0.233 | 0.034 | **0.306** |
| 30 | 0.279 | 0.021 | **0.566** |
| 54 | 0.246 | 0.016 | **0.995** |

![Fig. 1. Success rate versus channel uses on AWGN. The crossover lies between k = 18 and k = 24; the neural arms are limited by their objective, not by bandwidth.](fyp_md_files/figures/fig4_bandwidth_crossover.png)

Two boundary results sharpen the reading. At k = 1–3 the matched-budget digital link cannot be built at all (M3), so the low-bandwidth comparison is structural. At k = 54 a separated design with B = 8 reaches 0.974 on AWGN and 0.718 on Rayleigh at 0 dB (M4), whereas the shaped neural link reaches 0.999 at k = 18. The crossover is therefore a statement about how much bandwidth separation needs, not about neural networks being universally stronger.

The comparator matters. Removing the channel code and spending the whole budget on payload (uncoded QPSK, 3B ≤ k) lifts the low-bandwidth digital link from 0.062 to 0.526 at k = 12 (B = 4) and to 0.880 at k = 18 (B = 6). That is a large improvement, and it relocates the low-bandwidth ceiling onto the frozen framing rather than onto separation itself. Even so, the uncoded link reaches only 0.884 at k = 54 and does not overtake the shaped neural arm at any measured bandwidth below k = 54. The headline "digital wins at k ≥ 24" is true of the coded baseline specifically; the comparator must travel with the sentence.

### B. Why the Frozen Loss Fails, and What Fixes It

With λ_u = λ_v = 0 the task arm sits at 2–5% success and is flat across SNR at every k (Table I). A width-256 model still fails (0.027 AWGN, 0.041 Rayleigh), and its reconstruction MSE explodes to about 10³ while the task loss falls: the failure is a loss pathology, not a capacity limit.

Adding control and velocity penalties fixes it (Table II). The λ-heavy arm reaches 0.997 on both channels at k = 12, and 0.993–0.999 at k = 18 and k = 54 across the SNR grid. The λ-light arm (0.01) reaches 0.995 on AWGN but collapses to 0.019 on Rayleigh, so the strength of the shaping matters: the fix is not "any positive λ". A reconstruction auxiliary loss (0.243), fixed-SNR training (0.050), and longer training (0.035) do not recover the task, and an untrained model scores 0.016, so the metric is not trivially exploitable.

**Table II.** Wave-A loss-component ablation at k = 12 (E = 1,000).

| Arm | AWGN | Rayleigh |
|---|---:|---:|
| Frozen task loss (λ_u = λ_v = 0) | 0.020–0.048 | 0.033 |
| λ-light (0.01, 0.01) | 0.995 | 0.019 |
| **λ-heavy (0.1, 0.1)** | **0.997** | **0.997** |
| Reconstruction auxiliary loss | 0.243 | 0.244 |
| Untrained random model | 0.016 | — |

![Fig. 2. Wave-A arms at k = 12. Only the control-penalty arms recover the task; the light penalty is fragile on Rayleigh.](fyp_md_files/figures/fig_wave_a_arms.png)

The mechanism is visible in the gradients. The frozen arm's encoder gradient has p50 ≈ 3.0 × 10^14 and p99 ≈ 2.4 × 10^14, with 75–100 non-finite steps and no convergence. The λ-heavy arm has p50 ≈ 3.3–4.1 and p99 ≈ 57–113, with zero non-finite steps and convergence in 10/10 cells. The separation is roughly fourteen orders of magnitude, and it persists in wider networks, so capacity is not the explanation. Adding the penalties changes the conditioning of the optimisation, not the representational budget.

A term-removal study localises the effect further. Removing the control-effort penalty alone collapses success to 0.016–0.027 and, at k = 18 AWGN, destroys training outright (final loss 5,213.5 with non-finite gradients at the final step). Removing the velocity or terminal penalty changes nothing (0.992–0.999). The active term is λ_u, which is also the only term that is a direct function of the action applied to the plant and therefore the only one that shapes the encoder gradient through the closed loop. The experiment localises the effect; it does not prove that causal chain.

![Fig. 3. Training stability. λ-shaping bounds the encoder gradient by about fourteen orders of magnitude and removes non-finite steps.](fyp_md_files/figures/fig_training_stability.png)

### C. The 3GPP UMa-Aerial Link

On the pinned geometry (500 m, 50 m, 2 GHz; PL_mean = 95.63 dB) the shaped policy needs 87 dB of transmit SNR for 0.90 at k = 54 and 102 dB for 0.99; the coded link needs 108 dB and 117 dB. At k = 18 the shaped policy needs 91 dB for 0.90 while the coded link cannot reach 0.90 at any transmit SNR (its ceiling is 0.17). The advantage is therefore 21 dB at 0.90 and 15 dB at 0.99 for k = 54, and unbounded at k = 18 where the classical link is payload-limited.

Table III repeats the evaluation over twelve geometries. The advantage does not collapse: it stays between 15 and 21 dB for 0.90 and between 12 and 15 dB for 0.99 across a 5× span in distance, a 2× span in altitude, and a 1.75× span in carrier frequency. The 500/50/2 row reproduces the pinned result exactly (87 dB against 108 dB), which confirms that the sweep measures the same quantity. Distance dominates at roughly 29 dB per 5×, steeper than free space because the aerial NLoS branch adds a height-dependent exponent; raising altitude from 50 to 100 m recovers up to 9 dB at 1000 m as P_LoS rises from 0.772 to 0.853; carrier frequency adds a flat 5 dB from 2.0 to 3.5 GHz, shifting the budget without changing the comparison.

**Table III.** A-17 multi-geometry link budget: transmit SNR (dB) for SR ≥ 0.90 at k = 54 (E = 1,000).

| d_2D (m) | h (m) | f_c (GHz) | P_LoS | PL_mean (dB) | λ-heavy | Digital | Adv. |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 50 | 2.0 | 0.967 | 85.4 | 74 | 89 | +15 |
| 200 | 50 | 3.5 | 0.967 | 90.3 | 79 | 94 | +15 |
| 500 | 50 | 2.0 | 0.889 | 95.6 | 87 | 108 | +21 |
| 500 | 100 | 3.5 | 0.945 | 99.2 | 88 | 103 | +15 |
| 1000 | 50 | 2.0 | 0.772 | 105.3 | 103 | 123 | +20 |
| 1000 | 100 | 3.5 | 0.853 | 107.4 | 99 | 119 | +20 |

![Fig. 4. A-10/A-17 required transmit SNR on the 3GPP UMa-aerial link. The shaped policy retains a 15–21 dB advantage across twelve geometries.](fyp_md_files/figures/fig_aerial_required_snr.png)

Training under the full dispersion is a negative result and is reported as such. Both cells (k = 18 and 54) complete and both collapse, with final losses of 4,320 and 210 and 311 and 203 non-finite gradient steps; the checkpoints sit at 0.02–0.03 success. Resampling the >20 dB outage tail (3.78% of episodes) restores stability (loss 4,643 → 16.2) and moves the operating point by at most 2 dB relative to the Rayleigh-trained arm, inside sampling error. The honest claim is that 3GPP training neither helps nor hurts at this operating point, not that it recovers a specific number of decibels.

### D. Robustness: Code Family and Temporal Correlation

Table IV adds 5G NR LDPC base graph 2, implemented with the specified lifting sets, quasi-cyclic lifting, systematic encoding, and offset min-sum decoding (15 iterations, offset 0.5), without the 38.212 rate-matching chain [3]. No ordering changes. At k = 12, 18, and 24 the two families are statistically indistinguishable; at k = 54 the only visible gap runs against LDPC on AWGN (0.978 against 0.995) and disappears on Rayleigh (0.944 against 0.940). The gap is structural: at Z = 5 the BG2 mother code is 50/260 ≈ 0.19, so the 108-bit budget is obtained by puncturing to an effective rate of 0.44, which costs belief propagation exactly at 0 dB. The full 5G NR polar chain behaves the same way: identical to convolution at k = 12–30, and weaker at k = 54 (0.909 against 0.995). Four comparators — convolutional, SC polar, full 5G polar, and LDPC BG2 — therefore leave the crossover intact.

**Table IV.** 5G NR LDPC BG2 versus the convolutional baseline, mean success over 0–20 dB (E = 1,000).

| k | LDPC AWGN | Conv. AWGN | LDPC Rayleigh | Conv. Rayleigh |
|---:|---:|---:|---:|---:|
| 12 | 0.0656 | 0.0654 | 0.0646 | 0.0620 |
| 18 | 0.1518 | 0.1470 | 0.1522 | 0.1574 |
| 24 | 0.2910 | 0.2818 | 0.2914 | 0.2878 |
| 30 | 0.5546 | 0.5436 | 0.6154 | 0.6082 |
| 54 | 0.9780 | 0.9952 | 0.9438 | 0.9396 |

Temporal correlation is the second axis. Every headline number above draws one independent gain per control step, which is correct only if the platform moves many wavelengths between steps. Replacing that convention with a Clarke/Jakes sum-of-sinusoids process (32 oscillators, exact J0 autocorrelation) and sweeping the displacement per step Δ = v·Δt/λ from 0.033 to 3.34 wavelengths changes the mean fade duration from 1.10 to 4.08 steps at the most correlated point, while the marginal distribution stays Rayleigh (deep-fade fraction 9.2–9.6%). The shaped arm is insensitive: the largest deviation is 0.003 at k = 12 and 0.001 at k = 54. The digital k = 54 link loses up to 0.028 (0.912 against 0.940) because a convolutional code without interleaving has no time diversity and a four-step fade erases a run of symbols. The pre-registered rule was triggered by that digital change, and the required follow-up arm was run rather than argued away: retraining under correlated fading at Δ = 0.033 gives 0.993 against 0.994 for the stored checkpoint, inside the seed spread. The i.i.d. convention is therefore validated for the neural arms rather than assumed. Above Δ ≈ 0.33 the process is indistinguishable from the frozen model.

A CSI-error sweep completes the robustness picture: success rate is flat up to σ = 0.4, while reconstruction MSE degrades monotonically by about 18%. Success rate is too coarse to reveal that degradation, which is why both task and fidelity metrics are reported.

## V. DISCUSSION

The data supports a conditional design rule rather than a universal one. Task-oriented joint coding wins when the channel is the binding constraint: low bandwidth (k ≤ 18), low SNR, and fading. There the neural link allocates its few symbols to the subspace that the controller needs, while the digital link is either payload-limited or cliff-limited. At high bandwidth the coded link has enough symbols to protect all six components; the neural decoder's objective, not the channel, becomes the bottleneck, and the coded link overtakes it at k = 54, where the two are effectively tied.

Two results qualify the headline. First, the low-bandwidth digital ceiling is partly a framing artefact: an uncoded payload-maximal link reaches 0.526 at k = 12 against 0.062 for the frozen coded packet. The crossover is thus about payload efficiency and objective alignment together, and its location depends on which classical comparator is used. Second, the shaped neural arm is at the task ceiling (0.99+), so the benchmark proves that the policy works but cannot resolve small differences between good methods. A harder task is needed for that.

The 3GPP results add a power dimension. The absolute budget is large — 87–108 dB of transmit SNR — because it is a link-budget statement relative to a unit-gain benchmark, with no antenna gain, noise figure, or bandwidth. What matters for the comparison is that the ordering is stable: 15–21 dB at 0.90 and 12–15 dB at 0.99 over twelve geometries. The dispersion-training negative result is a limitation of the training recipe, not evidence against the mechanism: rare deep outages destabilise the rollout gradient, and resampling the tail restores stability without moving the operating point.

The main threats to validity are disclosed rather than hidden. The study is simulation-only and two-dimensional, with a fixed controller, perfect receiver CSI in the main matrix, and binary success as the primary metric. The 3GPP layer has no Doppler or time-correlated shadowing, and the LDPC arm omits the 38.212 rate-matching chain. None of these converts the result into a flight-certified claim. They do, however, mean that the crossover should be read as a property of the frozen framing, the objective, and the matched budget — precisely the quantities a system designer controls.

## VI. CONCLUSION

We asked whether a DeepJSCC link trained on UAV goal-reaching performance beats a reconstruction-oriented neural link and a matched-budget coded digital link. The answer is conditional and empirically grounded. At k ≤ 18 the neural link wins clearly; at k ≥ 24 the coded link overtakes it and ties at k = 54. At k = 1–3 the matched-budget digital link is structurally infeasible, and at k = 54 a separated B = 8 design needs 54 uses to reach 0.974 at 0 dB while the shaped link reaches 0.999 at k = 18. The frozen task loss fails at 2–5% success; control and velocity penalties recover 0.993–0.997, and a term-removal study localises the effect to the control-effort penalty, with a fourteen-order-of-magnitude gradient-conditioning signature.

On a 3GPP UMa-aerial link the shaped policy needs 15–21 dB less transmit SNR than the digital link for 0.90 success across twelve geometries, and the digital link cannot reach 0.90 at k = 18. A Clarke/Jakes Doppler sweep moves the neural arm by at most 0.003 and validates the i.i.d. fading convention for these policies; 5G NR LDPC BG2 does not change the ordering. Future work should test a harder, non-saturated task; a principled training scheme for the full 3GPP dispersion; a complete 38.212 LDPC chain; time-correlated shadowing; and hardware validation.

## REFERENCES

[1] 3GPP, *Study on Enhanced LTE Support for Aerial Vehicles*, TR 36.777, 2017.

[2] 3GPP, *Study on Channel Model for Frequencies from 0.5 to 100 GHz*, TR 38.901, 2019.

[3] 3GPP, *Multiplexing and Channel Coding*, TS 38.212, 2023.

[4] E. Bourtsoulatze, D. B. Kurka, and D. Gündüz, "Deep joint source-channel coding for wireless image transmission," *IEEE Trans. Cogn. Commun. Netw.*, vol. 5, no. 3, pp. 567–579, 2019.

[5] D. B. Kurka and D. Gündüz, "Bandwidth-agile image transmission with deep joint source-channel coding," *IEEE Trans. Wireless Commun.*, vol. 20, no. 12, pp. 8081–8095, 2021.

[6] D. Gündüz et al., "Beyond transmitting bits: Context, semantics, and task-oriented communications," *IEEE J. Sel. Areas Commun.*, vol. 41, no. 1, pp. 5–41, 2023.

[7] J. Shao, Y. Mao, and J. Zhang, "Learning task-oriented communication for edge inference: An information bottleneck approach," *IEEE J. Sel. Areas Commun.*, vol. 40, no. 1, pp. 197–211, 2022.

[8] Y. Diao, Z. Meng, X. Xu, C. She, and P. G. Zhao, "Task-oriented source-channel coding enabled autonomous driving based on edge computing," in *Proc. IEEE INFOCOM Workshops*, 2024.

[9] Y. Diao, Y. Zhang, C. She, P. G. Zhao, and E. L. Li, "Aligning task- and reconstruction-oriented communications for edge intelligence," *IEEE J. Sel. Areas Commun.*, early access, 2025.

[10] Y. Diao, Y. Zhang, D. De Martini, P. G. Zhao, and E. L. Li, "Task-oriented co-design of communication, computing, and control for edge-enabled industrial cyber-physical systems," *IEEE J. Sel. Areas Commun.*, early access, 2025.

[11] Z. Fang et al., "Task-oriented communications for visual navigation with edge-aerial collaboration in low altitude economy," arXiv:2504.18317, 2025.

[12] M. Tang, C. Feng, and T. Q. S. Quek, "Decentralized semantic communication and cooperative tracking control for a UAV swarm over wireless MIMO fading channels," *IEEE Trans. Veh. Technol.*, early access, 2025.

[13] Y. Xu, H. Zhou, and Y. Deng, "Task-oriented semantics-aware communication for wireless UAV control and command transmission," *IEEE Commun. Lett.*, vol. 27, no. 8, pp. 2232–2236, 2023.

[14] W. Wu, Y. Yang, Y. Deng, and A. H. Aghvami, "Goal-oriented semantic communications for robotic waypoint transmission: The value and age of information approach," arXiv:2312.13182, 2024.

[15] H. Wu, Y. Shao, C. Bian, K. Mikolajczyk, and D. Gündüz, "Deep joint source-channel coding for adaptive image transmission over MIMO channels," *IEEE Trans. Wireless Commun.*, vol. 23, no. 10, pp. 15002–15017, 2024.

[16] M. Yang, C. Bian, and H.-S. Kim, "OFDM-guided deep joint source-channel coding for wireless multipath fading channels," *IEEE Trans. Cogn. Commun. Netw.*, vol. 8, no. 2, pp. 584–599, 2022.

[17] J. Xu, B. Ai, W. Chen, A. Yang, P. Sun, and M. Rodrigues, "Wireless image transmission using deep source channel coding with attention modules," *IEEE Trans. Circuits Syst. Video Technol.*, vol. 32, no. 4, pp. 2315–2328, 2022.

[18] A. A. Khuwaja, Y. Chen, N. Zhao, M.-S. Alouini, and P. Dobbins, "A survey of channel modeling for UAV communications," *IEEE Commun. Surveys Tuts.*, vol. 20, no. 4, pp. 2804–2821, 2018.

[19] H. Shiri, J. Park, and M. Bennis, "Remote UAV online path planning via neural network based opportunistic control," arXiv:1910.04969, 2019.

[20] D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. ICLR*, 2015.

[21] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Proc. NeurIPS*, 2019.

[22] Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, no. 2, pp. 153–157, 1947.

[23] E. B. Wilson, "Probable inference, the law of succession, and statistical inference," *J. Amer. Statist. Assoc.*, vol. 22, no. 158, pp. 209–212, 1927.

[24] Y. Diao, "Task-oriented communication for edge intelligence enabled connected robotics systems," Ph.D. dissertation, Univ. Glasgow, 2025.
