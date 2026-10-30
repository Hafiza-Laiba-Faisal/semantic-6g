# TASK-ORIENTED DEEP JOINT SOURCE-CHANNEL CODING FOR UAV NAVIGATION UNDER NOISY AND FADING WIRELESS CHANNELS

**Final Year Project Thesis**

**Department of Electrical Engineering**
**Lahore College for Women University (LCWU)**

**Submitted by:**

| Name | Roll Number |
|---|---|
| Areesha Sohail | 1005 |
| Amna Shahid | 1004 |
| Hafiza Laiba Faisal | 1015 |

**Program:** B.E. Electrical Engineering
**Supervisor:** Ismat Hira
**Date:** October 2026

---

<!-- pagebreak -->
## Declaration

I hereby declare that the work presented in this thesis, titled “Task-Oriented Deep Joint Source-Channel Coding for UAV Navigation under Noisy and Fading Wireless Channels,” is my own work. It has not been submitted previously for any degree or examination at this or any other institution. All sources of information and assistance have been acknowledged, and all quotations and paraphrases are cited.

| Name | Roll Number | Signature |
|---|---|---|
| Areesha Sohail | 1005 | |
| Amna Shahid | 1004 | |
| Hafiza Laiba Faisal | 1015 | |

Date: ______________________

<!-- pagebreak -->
## Certificate

This is to certify that the thesis titled “Task-Oriented Deep Joint Source-Channel Coding for UAV Navigation under Noisy and Fading Wireless Channels,” submitted by Areesha Sohail (1005), Amna Shahid (1004), and Hafiza Laiba Faisal (1015) in partial fulfilment of the requirements for the degree of B.E. Electrical Engineering at Lahore College for Women University (LCWU), is a record of original work carried out under my supervision. The work has not been submitted elsewhere for the award of any other degree.

Supervisor: Ismat Hira

Signature: ______________________

Date: ______________________

<!-- pagebreak -->
## Acknowledgements

We thank our supervisor, Ismat Hira, for her guidance, patience, and careful feedback throughout this project. We are grateful to the Department of Electrical Engineering at Lahore College for Women University for the resources and support that made this work possible. We also thank our families and friends for their encouragement during the long experimental runs. Finally, we acknowledge the open-source Python, PyTorch, and NumPy communities, whose tools made the simulation study reproducible.

<!-- pagebreak -->
## Abstract

Reliable and bandwidth-efficient transmission of navigation-relevant information is a core requirement for autonomous unmanned aerial vehicle (UAV) operation over noisy and fading wireless links. Conventional digital communication and reconstruction-oriented Deep Joint Source-Channel Coding (DeepJSCC) are optimised to recover the transmitted signal with minimum distortion, independent of how that signal will subsequently be used. This thesis investigates a Task-Oriented DeepJSCC system in which the encoder and decoder are trained end-to-end to optimise UAV goal-reaching performance rather than reconstruction fidelity.

A compact six-dimensional UAV kinematic state — position, velocity, and goal-relative information — is transmitted over Additive White Gaussian Noise (AWGN) and slow Rayleigh fading channels and used by a fixed navigation controller on a two-dimensional double-integrator model. Three systems are compared under identical channel, SNR, and bandwidth conditions: a conventional digital link (quantisation, convolutional coding, QPSK), a reconstruction-oriented DeepJSCC autoencoder, and a task-oriented DeepJSCC system trained on the navigation rollout loss. The study is simulation-only and implemented in Python/PyTorch on CPU with deterministic seeds.

The central empirical finding is a bandwidth crossover: neural joint source-channel coding outperforms the classical digital link at low bandwidth (k ≤ 18 channel uses), while the classical link overtakes it at k ≥ 24 and dominates at k = 54. A second finding is diagnostic: the frozen task-oriented loss, which sets the control and velocity penalties to zero, collapses to 2–5% success. Adding control/velocity shaping (λ_u = λ_v = 0.1) recovers 99.7% success at k = 12 and 99.3–99.9% at k = 18 and 54, on both channels. The mechanism is a gradient-conditioning effect of roughly fourteen orders of magnitude, not a capacity effect. The result survives a 5G NR polar comparator, a 3GPP aerial link budget, a CSI-error sweep, and a three-seed replication. The thesis contributes a matched-budget benchmark, the crossover characterisation, and a reproducible diagnosis of why task-oriented training fails without loss shaping.

**Keywords:** DeepJSCC, task-oriented communication, semantic communication, UAV navigation, joint source-channel coding, AWGN, Rayleigh fading, 6G, goal-reaching control.

<!-- toc -->

## List of Abbreviations

| Abbreviation | Meaning |
|---|---|
| 3GPP | Third Generation Partnership Project |
| 5G NR | Fifth-Generation New Radio |
| AWGN | Additive White Gaussian Noise |
| BER | Bit Error Rate |
| BLER | Block Error Rate |
| CA-SCL | CRC-Aided Successive Cancellation List |
| CI | Confidence Interval |
| CPS | Cyber-Physical System |
| CSI | Channel State Information |
| CVAE | Conditional Variational Autoencoder |
| DeepJSCC | Deep Joint Source-Channel Coding |
| JSCC | Joint Source-Channel Coding |
| KPI | Key Performance Indicator |
| LoS | Line of Sight |
| MSE | Mean Squared Error |
| NLoS | Non-Line-of-Sight |
| O-VIB | Orthogonally Constrained Variational Information Bottleneck |
| PReLU | Parametric Rectified Linear Unit |
| QPSK | Quadrature Phase-Shift Keying |
| RL | Reinforcement Learning |
| RQ | Research Question |
| SC | Successive Cancellation |
| SNR | Signal-to-Noise Ratio |
| TOSA | Task-Oriented Semantics-Aware |
| TSCC | Task-oriented Source-Channel Coding |
| UAV | Unmanned Aerial Vehicle |
| UMa | Urban Macro |
| VIB | Variational Information Bottleneck |
| ZF | Zero-Forcing |

## Nomenclature

| Symbol | Meaning |
|---|---|
| s_t | UAV state vector at step t |
| p_t | UAV position at step t |
| v_t | UAV velocity at step t |
| u_t | Control input (commanded acceleration) at step t |
| p_g | Goal position |
| k | Number of channel uses per transmission |
| n | Source dimension (n = 6) |
| B | Bits per state component in the digital baseline |
| P | Average transmit power per symbol |
| σ² | Noise variance per complex symbol |
| h | Rayleigh fading coefficient |
| λ_d, λ_u, λ_v, λ_T | Task-loss weights for distance, control, velocity, terminal distance |
| T | Episode horizon (100 steps) |
| Δt | Simulation time step (0.1 s) |
| SR | Goal-reaching success rate |
| E | Number of evaluation episodes |

<!-- pagebreak -->
# Chapter 1 — Introduction

## 1.1 Background

Small, low-power UAVs are increasingly used for surveillance, delivery, inspection, and search-and-rescue missions. In most architectures the UAV depends on a wireless link to a ground station or remote controller for navigation-relevant information exchange. This link is rarely ideal: it is subject to additive thermal noise and, because the UAV moves through space with time-varying line-of-sight conditions and multipath reflections, to small-scale fading. Under such impairments, conventional bit-level pipelines — which separate source compression from channel coding and aim to reconstruct the transmitted signal exactly — are known to suffer from the cliff effect: performance degrades gracefully until the channel SNR falls below a design threshold, after which the system fails abruptly (Bourtsoulatze, Kurka & Gündüz, 2019).

Deep Joint Source-Channel Coding (DeepJSCC) replaces separate source and channel coding with a pair of jointly trained neural networks that map source symbols directly to complex-valued channel input symbols. It has been shown under AWGN and slow Rayleigh fading to degrade gracefully rather than catastrophically as SNR deteriorates. The great majority of DeepJSCC research, however, is reconstruction-oriented: the encoder and decoder minimise a distortion metric between the transmitted and received signal, independent of what the received information will subsequently be used for. In parallel, task-oriented and goal-oriented communication argues that when the purpose of transmission is a downstream decision or control action, the communication system should be optimised directly against that downstream task's performance rather than against signal fidelity (Gündüz et al., 2023; Shao, Mao & Zhang, 2022).

This thesis investigates the intersection of these two ideas for a specific downstream task: UAV navigation toward a goal position. It asks whether training a DeepJSCC encoder/decoder pair directly on navigation performance, rather than reconstruction error, yields a communication system that is more robust to channel impairments in the sense that matters to the mission — reaching the goal — even if it is not better at reproducing the original state bit-for-bit.

## 1.2 Motivation

From a communications engineering perspective, the project is motivated by three considerations. First, UAV platforms are typically bandwidth- and power-constrained, so any reduction in channel resources needed for a given navigation reliability is operationally valuable. Second, the graceful-degradation property already demonstrated for reconstruction-oriented DeepJSCC suggests that a task-oriented variant may inherit similar robustness while concentrating representational capacity on task-relevant information. Third, task-oriented DeepJSCC has been shown to improve closed-loop control under noisy channels in autonomous driving (Diao et al., 2024, 2025), but it has not been evaluated for UAV navigation using a compact kinematic state, leaving open whether the same benefits transfer to a lower-dimensional, non-visual source and a mobile aerial channel.

## 1.3 Problem Statement

When a UAV's navigation-relevant state information (position, velocity, and goal information) is transmitted over a noisy or fading wireless channel using either a conventional digital pipeline or a reconstruction-oriented DeepJSCC scheme, the communication system is optimised to minimise a generic distortion measure on the transmitted state vector. This measure does not necessarily correspond to what matters operationally: the UAV's ability to reach its goal safely and efficiently. Under a fixed bandwidth and SNR budget, a communication system that spends its limited channel-use budget reproducing every component of the state vector with equal fidelity may perform worse, in terms of navigation outcome, than a system that allocates representational capacity toward the state information most relevant to the navigation decision.

The problem addressed in this thesis is therefore whether, and to what extent, training a DeepJSCC encoder/decoder directly against a navigation-task objective — a Task-Oriented DeepJSCC system — improves UAV navigation performance relative to conventional digital communication and reconstruction-oriented DeepJSCC, under identical AWGN and Rayleigh fading channel conditions.

## 1.4 Research Gap

The literature confirms four facts. (i) DeepJSCC under AWGN and Rayleigh fading is a mature methodology for reconstruction-oriented image transmission. (ii) Task-oriented communication, formalised through an information bottleneck objective, is well established for downstream inference and classification. (iii) UAV-specific semantic and task-oriented communication has been investigated for command-and-control scheduling, swarm tracking control, waypoint repetition, and vision-based localisation. (iv) Task-oriented DeepJSCC trained directly against a closed-loop control objective under a noisy channel has been demonstrated for autonomous driving using camera imagery.

None of the reviewed works combines a DeepJSCC neural encoder/decoder, trained end-to-end so its loss is defined directly on UAV navigation/goal-reaching performance, operating on a compact kinematic state vector rather than imagery, and evaluated systematically against conventional digital and reconstruction-oriented DeepJSCC baselines under both AWGN and Rayleigh fading at matched bandwidth and SNR. This thesis targets that specific gap: a domain transfer and systematic empirical study, not the introduction of a fundamentally new communication paradigm.

## 1.5 Research Questions

**RQ1 (Task and communication performance).** To what extent does a Task-Oriented DeepJSCC framework, trained end-to-end to optimise UAV navigation performance, outperform conventional digital communication and reconstruction-oriented DeepJSCC in transmitting UAV navigation state information over AWGN and Rayleigh fading channels?

**RQ2 (Robustness and resource sensitivity).** How do channel type (AWGN versus Rayleigh fading), signal-to-noise ratio, and bandwidth/channel-use ratio affect the navigation performance gap between the Task-Oriented DeepJSCC system and the conventional and reconstruction-oriented baselines?

## 1.6 Aim

The aim is to design, implement, and empirically evaluate a Task-Oriented DeepJSCC communication system for UAV navigation-state transmission, and to characterise, through controlled simulation experiments under AWGN and Rayleigh fading, the conditions under which optimising the communication system directly for goal-reaching performance yields measurable benefits over conventional digital communication and reconstruction-oriented DeepJSCC.

## 1.7 Objectives

**Objective 1.** Design and implement an end-to-end Task-Oriented DeepJSCC framework for transmitting a compact UAV navigation-state representation over AWGN and Rayleigh fading channels.

**Objective 2.** Integrate the communication framework with a simplified two-dimensional UAV kinematic navigation model and quantify the effect of communication-induced state degradation on goal-reaching performance.

**Objective 3.** Benchmark the Task-Oriented DeepJSCC system against a conventional digital communication baseline and a reconstruction-oriented DeepJSCC baseline under identical channel, SNR, and bandwidth conditions.

## 1.8 Key Performance Indicators

**Objective 1 KPIs.** A functional task-oriented encoder and decoder in PyTorch with a defined channel-use ratio; a validated differentiable AWGN layer (empirical noise variance within tolerance); a validated differentiable Rayleigh layer (fading statistics matching the model); stable training convergence across at least three seeds; at least three bandwidth ratios supported.

**Objective 2 KPIs.** Goal-reaching success rate over a fixed test set; mean final position error; mean goal-reaching time; control-command error relative to the noiseless baseline; episode failure rate.

**Objective 3 KPIs.** Success-rate-versus-SNR curves for all three methods; BER/BLER for the digital baseline; reconstruction MSE for the reconstruction baseline; the performance gap in percentage points at matched bandwidth; the degradation slope from AWGN to Rayleigh.

## 1.9 Scope

The project is simulation-only and implemented in Python/PyTorch. It considers a single UAV on a point-to-point goal-reaching task with a simplified two-dimensional kinematic model. Only AWGN and Rayleigh fading channels are used. There is no physical UAV, no software-defined radio, no RF hardware, no real flight test, no federated or swarm learning, no full six-degree-of-freedom aerodynamics, and no full 3GPP channel simulator in the core matrix. The navigation controller is a fixed, lightweight policy; the contribution is the communication system, not a new control algorithm.

## 1.10 Thesis Organisation

Chapter 2 reviews the literature and positions the gap. Chapter 3 defines the system model: UAV dynamics, controller, channels, encoder/decoder, the three methods, and the loss. Chapter 4 describes the methodology, experimental matrix, metrics, and statistical protocol. Chapter 5 reports the results. Chapter 6 discusses their meaning. Chapter 7 states the limitations. Chapter 8 concludes and outlines future work. Appendices provide the frozen configuration, the protocol amendments, the reproduction map, and the disclosure paragraphs.

<!-- pagebreak -->
# Chapter 2 — Literature Review

## 2.1 Foundational Deep Joint Source-Channel Coding

Bourtsoulatze, Kurka and Gündüz (2019) introduced DeepJSCC for wireless image transmission. A jointly trained convolutional encoder/decoder maps pixel values directly to channel symbols, outperforming JPEG/JPEG2000-plus-capacity-achieving-channel-coding baselines at low SNR and small bandwidth. The system avoids the cliff effect under AWGN and, under slow Rayleigh fading, learns fading-resilient representations that outperform separation-based digital transmission at all tested SNR and bandwidth values. This paper establishes both the DeepJSCC architecture and the AWGN/Rayleigh evaluation protocol adopted in this thesis.

Subsequent work extended the architecture with attention modules for content-adaptive coding (Xu et al., 2022), variable-rate and bandwidth-agile coding (Kurka & Gündüz, 2021), explicit OFDM-based multipath fading (Yang, Bian & Kim, 2022), and MIMO fading channels (Wu, Shao, Bian, Mikolajczyk & Gündüz, 2024). These works confirm that channel-aware DeepJSCC under AWGN and Rayleigh fading is a mature methodology, but they remain reconstruction-oriented: the training objective is signal fidelity, not a downstream task outcome.

## 2.2 Task-Oriented and Semantic Communication

Gündüz et al. (2023) provide the tutorial framing adopted here, distinguishing semantic, goal-oriented, and task-oriented communication and surveying information-theoretic and learning-based approaches. Shao, Mao and Zhang (2022) formalise task-oriented communication for edge inference using a variational information bottleneck (VIB) objective that jointly optimises feature extraction, source coding, and channel coding for a downstream classification task rather than data reconstruction. They also propose a dynamic, channel-adaptive feature-length scheme.

These works establish task-oriented communication as a rigorous, IB-grounded paradigm, but their downstream task is classification or inference on a held-out label, not closed-loop control or navigation, and their source is high-dimensional imagery rather than a low-dimensional kinematic state vector.

## 2.3 UAV-Specific Semantic and Task-Oriented Communication

Four UAV-specific works are directly relevant.

**Fang et al. (2025)** propose task-oriented communication for UAV visual navigation with edge-aerial collaboration in the low-altitude economy. An orthogonally constrained variational information bottleneck (O-VIB) encoder compresses multi-camera image features before transmission to an edge server for position estimation. The wireless link is modelled through a Shannon-capacity expression incorporating path loss, shadowing, and small-scale fading as a scalar channel gain. The task is localisation, not closed-loop goal-reaching, and the channel is not a differentiable symbol-level AWGN/Rayleigh layer embedded in end-to-end training. This is the closest UAV navigation reference for the present work.

**Xu, Zhou and Deng (2023)** propose a task-oriented semantics-aware (TOSA) framework for UAV command-and-control transmission. It defines the value of information using similarity and age-of-information metrics. The work addresses scheduling and importance-weighting of command-and-control messages rather than learning a joint source-channel neural code.

**Wu, Yang, Deng and Aghvami (2024)** propose goal-oriented semantic communication for robotic (including UAV) waypoint transmission. A deep reinforcement learning agent generates command-and-control data and a proactive repetition scheme (DeepPro) combats channel unreliability. This is a scheduling and repetition strategy layered on generated commands, not a trained DeepJSCC encoder/decoder.

**Tang, Feng and Quek (2025)** study decentralised semantic communication and cooperative tracking control for a UAV swarm over MIMO fading channels. A semantic triggering rule decides whether to transmit, based on tracking-error cost and channel state, combined with a Lyapunov drift-plus-penalty control law. The work addresses when to transmit, not how to jointly encode the state into channel symbols.

## 2.4 Task-Oriented JSCC for Closed-Loop Control

The methodologically closest precedent is the work of Diao and colleagues. Diao, Meng, Xu, She and Zhao (2024) propose a Task-oriented Source-Channel Coding (TSCC) framework that uses a modified conditional variational autoencoder guided by an autonomous-driving agent to jointly optimise source and channel coding under different channel conditions in a driving simulator. Diao, Zhang, De Martini, Zhao and Li (2025) extend this to a task-oriented co-design of communication, computing, and control for industrial cyber-physical systems, combining VIB-based task-oriented JSCC with delay-aware control prediction. Diao, Zhang, She, Zhao and Li (2025) propose an alignment between task-oriented and reconstruction-oriented communication for edge intelligence, and the doctoral thesis (Diao, 2025) consolidates the line of work.

This body of work demonstrates that a DeepJSCC-style system optimised directly against closed-loop control performance under a noisy channel is feasible and beneficial. It is, however, exclusively for ground-vehicle autonomous driving using camera imagery, not for UAV navigation using a compact kinematic state, and it does not provide a systematic matched-bandwidth comparison across AWGN and Rayleigh fading.

## 2.5 UAV Mobility and Channel Models

The UAV kinematic model in this thesis follows the discrete-time double-integrator formulation used by Shiri, Park and Bennis (2019) in communication-aware UAV path planning. The state is position and velocity in two dimensions, and the control input is commanded acceleration. The model deliberately omits full flight dynamics; its purpose is to provide a reproducible downstream task, not a new controller.

For the aerial link budget extension, the large-scale parameters follow the 3GPP TR 36.777 / TR 38.901 UMa-aerial model: a LoS probability, separate LoS and NLoS path-loss expressions, and their mixture. Khuwaja et al. (2018) survey the wider family of air-to-ground channel models, including Rayleigh and Rician small-scale fading and altitude-dependent K-factors. These models inform the link-budget mapping in Appendix A and the limitations in Chapter 7.

## 2.6 Positioning of the Present Work

**Table 2.1 — Positioning of the reviewed work relative to this thesis.**

| Work | Application | Communication method | Channel model | Task metric | Limitation relative to this thesis |
|---|---|---|---|---|---|
| Bourtsoulatze et al. (2019) | Image transmission | Reconstruction DeepJSCC | AWGN; slow Rayleigh | PSNR / distortion | Reconstruction-oriented; not a control task |
| Shao, Mao & Zhang (2022) | Edge inference | Task-oriented VIB | AWGN-like link | Classification accuracy | Classification, not control; image source |
| Xu, Zhou & Deng (2023) | UAV command and control | TOSA scheduling | Cellular / URLLC | Value / age of information | Scheduling, not a neural JSCC codec |
| Wu, Yang, Deng & Aghvami (2024) | Robotic / UAV waypoints | RL commands + repetition | Packet-loss link | Value / age of information | Not a DeepJSCC encoder; packet abstraction |
| Tang, Feng & Quek (2025) | UAV swarm tracking | Semantic triggering + Lyapunov | MIMO fading | Tracking error, power | Decides when to transmit, not how to encode |
| Fang et al. (2025) | UAV visual navigation | Task-oriented O-VIB | Shannon capacity (path loss, shadowing, fading) | Localisation error | Image source; capacity-level channel; localisation |
| Diao et al. (2024, 2025) | Autonomous driving / CPS | Task-oriented JSCC (CVAE/VIB) | Noisy channel, varying SNR | Driving score, bandwidth | Ground vehicle; camera source; no UAV state |
| **This thesis** | **UAV goal-reaching** | **Task-oriented DeepJSCC** | **Differentiable AWGN and Rayleigh** | **Goal-reaching success** | **Simulation-only; 2-D kinematics** |

## 2.7 Summary

The gap is specific and defensible: no reviewed work trains a DeepJSCC encoder/decoder end-to-end on a UAV goal-reaching objective using a six-dimensional kinematic state, over a differentiable symbol-level AWGN/Rayleigh channel, and compares it against both a conventional digital link and a reconstruction-oriented DeepJSCC baseline at matched bandwidth. The thesis fills that gap and reports both the positive result (the crossover) and the negative/diagnostic result (the frozen task loss fails without control shaping).

<!-- pagebreak -->
# Chapter 3 — System Model

## 3.1 End-to-End Architecture

The architecture is a UAV-to-ground-station link. The UAV is always the transmitter of state information; the ground-based navigation controller is always the receiver and decision-maker. The loop is:

**UAV state → task-oriented encoder → DeepJSCC channel symbols → AWGN / Rayleigh channel → task-oriented decoder → control-relevant representation → navigation controller → control action → UAV state update.**

The same loop is used by all three methods. They differ only in how the encoder, decoder, and training objective are defined.

## 3.2 UAV Kinematic Model

The navigation-relevant state at discrete step t is

**s_t = [x_t, y_t, v_x,t, v_y,t, x_g, y_g] ∈ R^6**

where (x_t, y_t) is position, (v_x,t, v_y,t) is velocity, and (x_g, y_g) is the fixed goal. The dynamics are a discrete double integrator:

**p_{t+1} = p_t + Δt · v_t + ½ Δt² · u_t**

**v_{t+1} = v_t + Δt · u_t**

with Δt = 0.1 s, horizon T = 100 steps (10 s), workspace ±8 m, success radius 0.5 m, velocity limit 5 m/s, and acceleration limit 2 m/s². An episode succeeds when the UAV is within 0.5 m of the goal at or before T; it fails on timeout or on leaving the workspace.

## 3.3 Navigation Controller

The controller is fixed and identical for all methods:

**u_t = clip(−K_p (p_t − p_g) − K_v v_t, ±u_max)**

with K_p = 1.0, K_v = 2.0, u_max = 2.0 m/s². The damping ratio is ζ = 1.0 (critically damped). Keeping the controller fixed ensures that any performance difference is attributable to the communication system, not to a jointly changing control policy.

## 3.4 State Normalisation

Each state component is mapped to [−1, 1] using a fixed affine transform with ranges taken from the configuration, never from data:

**s̄_i = 2 · (clip(s_i, a_i, b_i) − a_i) / (b_i − a_i) − 1**

The inverse transform is applied after decoding. Fixed ranges prevent data leakage and make the normalisation identical across methods and seeds.

## 3.5 Communication Model and Bandwidth Convention

At each step the UAV encodes s_t into k complex channel symbols z_t. The bandwidth/channel-use ratio is k/n, where n = 6 is the source dimension and k is the number of channel uses per transmission. The thesis uses k ∈ {1, 2, 3} for the low-bandwidth study and k ∈ {12, 18, 24, 30, 54} for the matched-budget study.

The transmitted symbols satisfy an average power constraint. Per codeword, the encoder output is normalised:

**z ← z · sqrt(kP / ||z||²),  P = 1**

so that the average symbol power is exactly P. The receiver observes y and produces a task-relevant representation r_t.

## 3.6 AWGN Channel

**y = z + n,  n ~ CN(0, σ² I)**

with per-symbol noise variance σ² = P · 10^(−SNR/10). The channel SNR is SNR = P/σ², expressed in dB. AWGN isolates thermal noise from multipath and is the baseline under which DeepJSCC was first established.

## 3.7 Rayleigh Fading Channel and CSI

**y = h · z + n,  h ~ CN(0, 1)**

The fading coefficient is drawn independently for each codeword and held constant over its k symbols (slow, block fading). The receiver has perfect channel state information and applies zero-forcing equalisation:

**ŷ = y / h**

with a numerical guard on |h|. The transmitter has no CSI (open-loop). These idealisations — perfect receiver CSI, ZF equalisation, packet-granularity block fading, and open-loop power control — are standard in the DeepJSCC literature and are applied identically to all three methods. Imperfect CSI is treated as a separate ablation (Wave-B, Section 4.7).

## 3.8 DeepJSCC Encoder and Decoder

The encoder maps the six-dimensional normalised state to 2k real values through two hidden layers of width 128 with PReLU activations:

**6 → 128 → 128 → 2k real → k complex symbols**

The real and imaginary parts are stacked into complex symbols and power-normalised. The decoder is the mirror image:

**2k real → 128 → 128 → 6**

Both DeepJSCC variants — reconstruction and task-oriented — use exactly the same architecture. They differ only in the training objective. This isolates the effect of the objective.

## 3.9 The Three Communication Methods

**Method A — Conventional digital.** Each state component is scalar-quantised to B bits, framed and padded, encoded with a convolutional code (K = 7, rate 1/2), modulated with Gray-labelled QPSK, transmitted over the same channel, and decoded with soft-decision Viterbi. The frozen pairs satisfy 6B + 6 = k exactly, where the six extra uses are the zero-tail termination bits. A pair whose payload would exceed k is declared infeasible and excluded, never silently widened. The B values are k = 12 → B = 1, k = 18 → B = 2, k = 24 → B = 3, k = 30 → B = 4, k = 54 → B = 8.

**Method B — Reconstruction-oriented DeepJSCC.** The encoder/channel/decoder above, trained to minimise the mean-squared reconstruction error between the decoded and original state:

**L_recon = (1/6) · E[ ||ŝ_t − s_t||² ]**

**Method C — Task-oriented DeepJSCC.** The same architecture, trained end-to-end on the navigation rollout loss defined in Section 3.10. Gradients propagate from the goal-reaching objective through the decoder, the differentiable channel, and the encoder.

## 3.10 Task-Oriented Loss

The task loss is a weighted sum of four terms:

**L_task = λ_d · L_d + λ_u · L_u + λ_v · L_v + λ_T · L_T**

where

**L_d = (1/T) Σ_t ||p_t − p_g||²**  (average squared distance to goal)

**L_u = (1/T) Σ_t ||u_t||²**  (average control effort)

**L_v = (1/T) Σ_t ||v_t||²**  (average velocity)

**L_T = ||p_T − p_g||²**  (terminal squared distance)

The frozen protocol used **(λ_d, λ_u, λ_v, λ_T) = (1, 0, 0, 2)**. The diagnostic extension uses **(1, 0.1, 0.1, 2)**. The control and velocity terms are the subject of the Wave-A ablation in Chapter 5.

## 3.11 Fairness Conditions

All three methods share the same UAV environment, initial conditions, goal configurations, channel model and SNR for a given cell, bandwidth budget, test episodes, and evaluation seeds. The two DeepJSCC variants have identical architecture and parameter count. The digital baseline is deterministic and untrained. The comparison is therefore matched on every axis except the communication method itself.

## 3.12 Disclosure of Idealizations

The Rayleigh channel is slow block fading with one complex gain per k-symbol packet. The receiver has perfect CSI and applies ZF equalisation. The transmitter has no CSI. These four idealisations — perfect receiver CSI, ZF, packet-granularity fading, and open-loop power control — are standard in the DeepJSCC literature and are applied identically to all methods. Wave-B quantifies the sensitivity to imperfect CSI by injecting estimation error ε ~ CN(0, σ²) at the receiver; success rate is found to be insensitive up to σ = 0.4, while reconstruction MSE degrades by approximately 18%.

The digital baseline uses a convolutional code with K = 7, rate 1/2, QPSK, and soft-decision Viterbi decoding. The SC polar decoder is used as a low-complexity reference in A-11; the full 5G NR chain with CRC-aided SCL (L = 8) is used in A-15. The primary comparison is between communication paradigms, not between decoder implementations.

<!-- pagebreak -->
# Chapter 4 — Methodology and Experimental Design

## 4.1 Research Design

The project follows a simulation-based, empirical machine-learning methodology in four stages. First, a two-dimensional UAV kinematic environment is implemented, providing ground-truth states and a controller that can operate on ground-truth or communication-degraded estimates. Second, the conventional digital baseline and the reconstruction-oriented DeepJSCC baseline are implemented and validated independently of the navigation task. Third, the task-oriented DeepJSCC system is implemented with the channel embedded as a differentiable layer, so gradients flow from the navigation objective to the encoder. Fourth, all three systems are evaluated under an identical matrix spanning channel type, SNR, and bandwidth, followed by the diagnostic and robustness extensions.

## 4.2 Training Protocol

The reconstruction model is trained for 5,000 steps with a batch of 4,096 states. The task model is trained for 3,000 steps with a batch of 256 episodes and a horizon of T = 100. Both use Adam with learning rate 1e-3, gradient clipping at 1.0, and no early stopping. The training SNR is sampled uniformly from 0 to 20 dB per batch. The final checkpoint, not a best-validation checkpoint, is used in all evaluations. Training seeds are {42, 43, 44}; the test seed is 10042. All training and evaluation are CPU-only, which makes execution deterministic for a given seed and configuration.

## 4.3 Experimental Matrix

The matrix spans the following dimensions.

**Table 4.1 — Experimental matrix.**

| Dimension | Levels |
|---|---|
| Channel | AWGN; slow Rayleigh fading |
| SNR | 0, 2, 4, …, 20 dB (M1); 0, 5, 10, 15, 20 dB (M2 and extensions) |
| Bandwidth k | 1, 2, 3 (M1); 12, 18, 24, 30, 54 (M2) |
| Method | Digital; reconstruction DeepJSCC; task-oriented DeepJSCC |
| Seeds | 42, 43, 44 |
| Test episodes | 5,000 per unit (M2, Wave-B); 1,000 per unit (Wave-A, Wave-C, A-11, A-15) |

## 4.4 M1 — Low-Bandwidth Study

M1 evaluates k = 1, 2, 3 with 198 paired evaluation units and three seeds. It establishes the baseline behaviour of the task and reconstruction arms at severe bandwidth limits. The result is that reconstruction reaches 24–31% success while the frozen task arm sits at 2–5% and is flat across SNR.

## 4.5 M2 — Matched-Budget Three-Way Comparison

M2 evaluates k = 12, 18, 24, 30, 54 with 150 evaluation units, three seeds, and 5,000 episodes per unit. It is the main benchmark and produces the bandwidth-crossover result. The frozen (k, B) pairs satisfy 6B + 6 = k exactly; infeasible cells are declared, not widened.

## 4.6 Wave-A — Loss-Component Ablation

Wave-A is a post-freeze diagnostic at k = 12, three seeds, both channels, with E = 1,000. It tests six arms: the frozen task loss; a light control penalty (λ_u = λ_v = 0.01); a heavy control penalty (0.1); a reconstruction auxiliary loss; fixed-SNR (10 dB) training; and longer training (12,000 steps). It also includes an untrained random model as a metric-sanity control. Wave-A completes 85 of 90 slots; two cells fail with non-finite gradients and are reported as failures rather than retrained away.

## 4.7 Wave-B — CSI-Estimation-Error Ablation

Wave-B evaluates 350 cells across five CSI-error levels (σ = 0, 0.05, 0.1, 0.2, 0.4), all k values, both channels, and E = 5,000. The estimator is **ĥ = h + ε**, ε ~ CN(0, σ²); equalisation and LLR-noise scaling use ĥ only. The error is paired across methods and across σ, so every method sees identical episodes. The σ = 0 gate reproduces the frozen M2 results bit-exactly (70/70 gates pass).

## 4.8 Wave-C — Generalisation of λ-Shaping

Wave-C is a four-cell screening study: λ-heavy at k = 18 and 54, both channels, seed 42, E = 1,000. It asks whether the k = 12 fix generalises to other bandwidths. All four cells train in parallel; wall-clock is 4,956–5,457 s per cell.

## 4.9 Extensions A-9 to A-15

**A-9 (capacity).** A width-256 task model tests whether the low-bandwidth failure is a capacity artefact.

**A-10 (aerial link).** The trained λ-heavy checkpoints are evaluated through a 3GPP UMa-aerial large-scale link budget at d_2D = 500 m, h = 50 m, f_c = 2 GHz. Under perfect-CSI ZF, a deterministic gain is exactly an effective-SNR shift, so no retraining is required. LoS and NLoS curves are mixed with P_LoS.

**A-11 (SC polar).** A CRC-8 polar code with SC decoding and no rate matching is compared with the convolutional baseline on the same episodes. It tests whether the crossover depends on the specific classical code.

**A-12 (3GPP dispersion training).** The λ-heavy arm is retrained under the full 3GPP dispersion (per-episode LoS/NLoS, TR 38.901 shadowing, block fading). It is a negative result: both cells diverge.

**A-13 (outage-tail resampling).** The same design, but the rare >20 dB outage tail (3.78% of episodes) is resampled. It tests whether the divergence is caused by those rare high-error episodes.

**A-14 (k = 24 seeds).** Six cells, three seeds, two channels, test whether the k = 24 result is a one-initialisation artefact.

**A-15 (5G NR polar).** A full 5G NR polar chain — CRC-6/11, 5G reliability construction, rate matching, CA-SCL with L = 8 — is compared with the convolutional baseline on the frozen grid and identical episodes. It is the strongest classical comparator in the thesis.

## 4.10 Evaluation Metrics

**Navigation-level.** Goal-reaching success rate (binary, final distance ≤ 0.5 m within T); mean final distance; mean average distance; minimum distance; control effort; time-to-goal; episode failure rate (timeout or workspace exit).

**Communication-level.** Operating SNR; BER/BLER for the digital baseline; bandwidth/channel-use ratio; reconstruction MSE for the reconstruction baseline.

**Trajectory-level.** Trajectory MSE against the noiseless-communication rollout, with the caveat that a low trajectory MSE does not imply successful navigation (the post-exit bypass can make MSE look small even when the episode failed).

## 4.11 Statistical Analysis

All methods are evaluated on bitwise-identical episodes: initial positions, goals, noise, and fading are pre-generated once per (k, channel, SNR, seed) unit and shared across methods. This paired design removes episode-sampling variance from pairwise comparisons. Binary success is compared with the McNemar test (two-sided, continuity-corrected χ²(1)); continuous metrics use the paired t-test. Wilson 95% confidence intervals are reported on all success rates. Three training seeds are used; pooled results aggregate 15,000 episodes per condition in M2. Per-seed means and standard deviations are reported alongside pooled estimates. The pooled Wilson interval reflects variability over training replicas, not statistically independent episodes.

## 4.12 Reproducibility and Engineering Controls

Every tunable constant lives in `project/config.py` with a provenance tag ([FROZEN], [LIT], or [PROJ]). The codebase has 57 passing tests covering the channel statistics, power normalisation, digital budget, evaluation gates, McNemar formula, architecture width, and protocol locks. The Docker image builds and the full suite runs on Linux in CI. All experiments are CPU-only and deterministic; a given (seed, configuration) pair reproduces bit-identically. Recovery from interruption retrains from the fixed seed under the frozen protocol, rather than resuming a partially saved optimizer state.

<!-- pagebreak -->
# Chapter 5 — Results

## 5.1 M1 — Low Bandwidth (k = 1, 2, 3)

**Table 5.1 — M1 low-bandwidth results (k = 1, 2, 3).**

| Method | Success rate (AWGN) | Behaviour |
|---|---|---|
| Reconstruction DeepJSCC | 0.24 – 0.31 | Learns a usable state representation |
| Task-oriented, frozen loss | 0.02 – 0.05 | Flat across SNR; does not learn the task |

The frozen task arm is not merely worse than reconstruction; it is essentially unresponsive to SNR. That pattern motivated the Wave-A diagnosis.

## 5.2 M2 — The Bandwidth Crossover (Main Result)

Pooled over three seeds and 5,000 episodes per unit, AWGN:

**Table 5.2 — M2 success rate by bandwidth on AWGN.**

| k | Reconstruction | Task (frozen λ) | Digital | Best method |
|---|---|---|---|---|
| 12 | **0.253** | 0.054 | 0.064 | Reconstruction |
| 18 | **0.237** | 0.025 | 0.165 | Reconstruction |
| 24 | 0.235 | 0.036 | **0.324** | Digital |
| 30 | 0.262 | 0.021 | **0.629** | Digital |
| 54 | 0.249 | 0.036 | **0.935** | Digital |

The crossover occurs between k = 18 and k = 24. At k = 12 the neural link is roughly four times the digital link; at k = 54 the digital link reaches 0.935 while reconstruction stays near 0.249. Reconstruction MSE is flat in k (0.049–0.054), which shows that the neural decoder is limited by its training objective, not by the number of channel uses.

![Figure 5.1 — M2 success rate versus bandwidth: the crossover between k = 18 and k = 24.](fyp_md_files/figures/fig4_bandwidth_crossover.png)

## 5.3 Wave-A — The λ-Shaping Diagnosis

At k = 12, three seeds, both channels:

**Table 5.3 — Wave-A loss-component ablation at k = 12.**

| Arm | AWGN | Rayleigh | Conclusion |
|---|---|---|---|
| Frozen task loss (λ_u = λ_v = 0) | 0.020–0.048 | 0.033 | Fails |
| λ-light (0.01, 0.01) | 0.995 | 0.019 | Fragile / degenerate on Rayleigh |
| **λ-heavy (0.1, 0.1)** | **0.997** | **0.997** | Recovers the task |
| Reconstruction auxiliary loss | 0.243 | 0.244 | Does not fix it |
| Fixed 10 dB training | 0.050 | — | Does not fix it |
| Longer training (12k steps) | 0.035 | — | Does not fix it |
| Untrained random model | 0.016 | — | Metric is not trivially exploitable |

Three controls establish that the result is real. First, an untrained model scores 0.016, so the metric cannot be gamed by a trivial policy. Second, at −20 dB the λ-heavy arm drops to 0.279, so the channel is genuinely used. Third, the λ-light Rayleigh arm is degenerate at 0.019, which shows the effect is fragile and depends on the strength of the shaping.

![Figure 5.2 — Wave-A arm ablation: only the control-penalty arms recover the task.](fyp_md_files/figures/fig_wave_a_arms.png)

## 5.4 Mechanism — Why λ-Shaping Works

The A-9 mechanism study extracts every per-step gradient measurement stored by the project. The frozen task arm's encoder gradient is enormous: p50 ≈ 3.0 × 10^14 and p99 ≈ 2.4 × 10^14, with 75–100 non-finite gradient steps and no convergence. The λ-heavy arm's gradient is O(1): p50 ≈ 3.3–4.1 and p99 ≈ 57–113, with zero non-finite steps and convergence in 10/10 cells. The separation is roughly fourteen orders of magnitude.

A quasi-2 × 2 isolates the trigger: λ off diverges (0/2); λ on with standard channels converges (10/10); λ on with the full 3GPP dispersion diverges (0/2); λ on with the outage tail resampled converges (2/2). Capacity is not the explanation: the frozen-arm histories are from the wider (256) cells and still explode, and A-9 shows a 256-wide frozen cell still fails at 0.027/0.041. The trigger is rare high-error episodes, not parameter count.

![Figure 5.3 — Training stability: λ-shaping bounds the encoder gradient by about fourteen orders of magnitude.](fyp_md_files/figures/fig_training_stability.png)

## 5.5 Wave-B — CSI-Estimation Error

**Table 5.4 — Wave-B CSI-estimation-error sweep, pooled across k.**

| σ_csi | Reconstruction SR | Task SR | Digital SR | Reconstruction MSE | Final distance |
|---|---|---|---|---|---|
| 0.0 | 0.2468 | 0.0341 | 0.3841 | 0.0488 | 3.491 |
| 0.05 | 0.2472 | 0.0342 | 0.3840 | 0.0490 | 3.493 |
| 0.1 | 0.2473 | 0.0342 | 0.3846 | 0.0494 | 3.496 |
| 0.2 | 0.2467 | 0.0341 | 0.3875 | 0.0512 | 3.515 |
| 0.4 | 0.2475 | 0.0339 | 0.3946 | **0.0576** | 3.569 |

Success rate is essentially flat up to σ = 0.4, while reconstruction MSE degrades monotonically by about 18% at every k (+18.0% at k = 12, +18.5% at k = 18, +17.8% at k = 24, +18.2% at k = 30, +17.6% at k = 54). The honest claim is therefore: closed-loop success is insensitive to CSI magnitude error up to σ = 0.4, but success rate is too coarse to reveal the degradation; MSE exposes it. The digital SR rises slightly with CSI error (0.384 → 0.395); this is physically unexpected and is reported as an anomaly, not as robustness.

![Figure 5.4 — Wave-B: success rate versus k with CSI-error bands.](fyp_md_files/figures/fig_waveb1_sr_vs_k_csi_bands.png)

## 5.6 Wave-C — Generalisation of λ-Shaping

**Table 5.5 — Wave-C λ-heavy success rate at k = 18 and k = 54.**

| Unit | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB |
|---|---|---|---|---|---|
| λ-heavy, k = 18, AWGN | 0.999 | 0.999 | 0.999 | 0.998 | 0.999 |
| λ-heavy, k = 18, Rayleigh | 0.993 | 0.997 | 0.996 | 0.997 | 0.997 |
| λ-heavy, k = 54, AWGN | 0.998 | 0.999 | 0.999 | 0.998 | 0.999 |
| λ-heavy, k = 54, Rayleigh | 0.996 | 0.998 | 0.998 | 0.998 | 0.998 |

The λ-shaped link is at the task ceiling in every condition. The baselines matter because of that ceiling:

**Table 5.6 — Wave-C λ-heavy advantage over the best baseline.**

| Condition | λ-heavy | Best baseline | Gap |
|---|---|---|---|
| k = 18, AWGN, 0 dB | 0.999 | 0.164 | +83.5 pp |
| k = 18, Rayleigh, 0 dB | 0.993 | 0.166 | +82.7 pp |
| k = 54, AWGN, 0 dB | 0.998 | 0.974 | +2.4 pp |
| k = 54, Rayleigh, 0 dB | 0.996 | 0.718 | +27.8 pp |
| k = 54, Rayleigh, 20 dB | 0.998 | 1.000 | −0.2 pp (parity) |

At k = 54 and high SNR the classical link saturates at 1.000, so λ-shaping cannot win there; it ties. The advantage lives at low SNR, low bandwidth, and fading. A policy that does nothing cannot produce a 0.125 m final distance on a randomly initialised target while spending 341 units of control effort; the success is tracking, not idling.

![Figure 5.5 — Wave-C: λ-shaping holds at k = 18 and k = 54.](fyp_md_files/figures/fig_wavec1_sr_vs_snr.png)

## 5.7 A-9 — Capacity Ablation

A width-256 task model still fails: 0.027 AWGN and 0.041 Rayleigh, versus 0.018/0.024 at width 128. The task loss falls while reconstruction MSE explodes to about 10^3. The failure is therefore a loss/objective pathology, not a capacity limitation.

## 5.8 A-10 — 3GPP Aerial Link Budget

At d_2D = 500 m, h = 50 m, f_c = 2 GHz, P_LoS = 0.8887, PL_mean = 95.63 dB:

**Table 5.7 — A-10 required transmit SNR on the 3GPP UMa-aerial link.**

| Method | Transmit SNR for SR ≥ 0.90 | Transmit SNR for SR ≥ 0.99 |
|---|---|---|
| λ-heavy, k = 18 | 91 dB | 107 dB |
| λ-heavy, k = 54 | 87 dB | 102 dB |
| Digital, k = 18 | Not reachable (ceiling 0.17) | Not reachable |
| Digital, k = 54 | 108 dB | 117 dB |

The λ policy's advantage over the classical link is 21 dB at 0.90 and 15 dB at 0.99 for k = 54, and unbounded at k = 18 where the classical link is payload-limited. The model includes LoS probability, LoS/NLoS path loss, and their mixture. It does not model Doppler, shadowing, aerial small-scale fading, antenna patterns, noise figure, or bandwidth. One geometry is evaluated.

![Figure 5.6 — A-10: required transmit SNR on the 3GPP UMa-aerial link.](fyp_md_files/figures/fig_aerial_required_snr.png)

## 5.9 A-11 and A-15 — Is the Digital Baseline Artificially Weak?

A-11 uses CRC-8 polar with SC decoding and no rate matching. At k = 12, 18, 24 the two coders are within 0.05 of each other once the payload ceiling is reached; at k = 30 polar is better at ≥10 dB; at k = 54 with B = 8, plain SC polar is worse at low SNR and reaches parity only at 10–15 dB.

A-15 uses the full 5G NR chain (CRC-6/11, 5G reliability construction, rate matching, CA-SCL with L = 8), pooled over five SNRs at E = 1,000:

**Table 5.8 — A-15 full 5G NR polar comparator versus the convolutional baseline.**

| k | Convolutional | 5G NR polar + CA-SCL | Reading |
|---|---|---|---|
| 12 | 0.0654 | 0.0656 | Identical |
| 18 | 0.1470 | 0.1504 | Identical within CI |
| 24 | 0.2818 | 0.2926 | 5G slightly better; still above reconstruction 0.235 |
| 30 | 0.5436 | 0.5596 | 5G slightly better |
| 54 | **0.9952** | 0.9092 | Convolutional still stronger |

The crossover at k = 18 → 24 survives the strongest classical comparator. The reported gap does not depend on the convolutional code being weak.

![Figure 5.7 — A-11: polar versus convolutional digital baseline.](fyp_md_files/figures/fig_polar_vs_conv.png)

## 5.10 A-12 and A-13 — Training under 3GPP Dispersion

A-12 retrains the λ-heavy arm under the full 3GPP dispersion at k = 18 and 54. Both cells complete and both training runs collapse: final losses of 4,320 and 210, with 311 and 203 non-finite gradient steps. The checkpoints are flat at 0.02–0.03 success. The result is reported as the negative result it is and is not retrained.

A-13 repeats the design with the >20 dB outage tail (3.78% of episodes) resampled. Training is stable (loss 4,643 → 16.2), and the operating point moves by at most 2 dB relative to the Rayleigh-trained arm — inside sampling error. The honest claim is that training under the dispersion does not move the operating point; it is not a claim that 3GPP training recovers a specific number of dB. A-13 trains on 96.2% of the dispersion (100% LoS, 66% NLoS) and evaluates on 100%.

## 5.11 A-14 — k = 24 Seed Variance

Six cells, three seeds, two channels: success ranges from 0.995 to 1.000 with a seed-to-seed range ≤ 0.004. The k = 24 number is not a one-initialisation artefact. Because the metric is at its ceiling, the study rules out a large initialisation dependence rather than measuring seed variance at a non-saturated point.

## 5.12 Summary of Findings

1. **Bandwidth crossover.** Neural JSCC wins at k ≤ 18; the classical digital link wins at k ≥ 24. This is the primary thesis contribution and is robust to a 5G NR polar comparator.
2. **Frozen task loss fails.** With λ_u = λ_v = 0, the task arm is at 2–5% success and flat across SNR, at every k.
3. **λ-shaping fixes it.** With λ_u = λ_v = 0.1, success reaches 99.7% at k = 12 and 99.3–99.9% at k = 18 and 54, on both channels.
4. **The mechanism is gradient conditioning.** The encoder gradient is bounded by about fourteen orders of magnitude; capacity is not the cause.
5. **CSI magnitude error is hidden by success rate.** SR is flat to σ = 0.4; MSE degrades by about 18%.
6. **Aerial link.** The λ policy needs 91 dB (k = 18) or 87 dB (k = 54) of transmit SNR for 0.90; the digital link needs 108 dB at k = 54 and cannot reach 0.90 at k = 18.
7. **Honest limits.** The 0.99+ result is at the task ceiling; high-bandwidth/high-SNR digital reaches 1.000; full-dispersion training (A-12) diverges.

<!-- pagebreak -->
# Chapter 6 — Discussion

## 6.1 When Task-Oriented Communication Wins

The data supports a conditional answer, not a universal one. Task-oriented communication wins when the channel is the binding constraint: low bandwidth (k ≤ 18), low SNR, and fading. In those regimes the neural code allocates its limited symbols to the state information that matters for control, and the digital link is either payload-limited or cliff-limited. At high bandwidth the classical link has enough symbols to send the six state components reliably; the neural decoder's objective, not the channel, becomes the bottleneck, and the digital link wins.

This is the direct 6G answer the thesis offers: semantic/task-oriented communication is not a universal replacement for classical communication. It is a regime-specific tool, and the crossover is where the regime changes.

## 6.2 Why λ-Shaping Works

The frozen task loss optimises distance and terminal distance but places no cost on control effort or velocity. The resulting gradient is enormous and non-finite, and training does not converge. Adding a small control/velocity penalty changes the conditioning of the optimisation, not the capacity of the network. The gradient falls from ~10^14 to O(1), non-finite steps disappear, and the policy converges. The effect is therefore best described as loss shaping for optimisation stability, with a task-level benefit.

The fragility of the λ-light arm on Rayleigh (0.019) is important: the fix is not "any positive λ". The strength of the shaping matters, and the mechanism is sensitive to the channel distribution. A-12 shows that it is also sensitive to rare high-error episodes in training: the same λ arm diverges when the training set contains >20 dB outages, and resampling the tail restores stability.

## 6.3 Why the Crossover Exists

At low k the digital link cannot carry six quantised components with enough error protection; the payload is too small and the code rate is fixed. The neural link can compress the six components jointly and protect the task-relevant subspace, so it degrades gracefully. At high k the digital link has enough symbols for a low-rate code with strong protection, and its performance saturates near 1.000. The neural decoder, trained on reconstruction MSE, remains limited by its objective and does not improve with k; its MSE is flat in k. The crossover is therefore a statement about the interaction between source dimension, channel uses, and training objective, not about neural networks being universally better.

## 6.4 Implications for 6G UAV Communication

The results suggest a design rule for aerial links: use task-oriented joint source-channel coding when the uplink is bandwidth-constrained or fading-limited, and use a classical digital link when bandwidth is abundant. The λ-shaping result adds an implementation rule: a task-oriented loss must include the control/effort terms that make the representation useful to the controller; a distance-only objective is not enough. The CSI result adds a measurement rule: success rate alone can hide an 18% reconstruction degradation, so report both task and fidelity metrics.

## 6.5 Threats to Validity

**Internal validity.** The controller is fixed, the episodes are paired, and the seeds are fixed, which removes the main confounds. The main internal threat is the choice of λ and the fact that the λ-heavy arm is at ceiling; both are disclosed.

**External validity.** The model is 2-D, simulation-only, single-UAV, and uses AWGN/Rayleigh plus a large-scale aerial link budget. It does not include Doppler, shadowing, aerial small-scale fading, or hardware. The results should not be read as flight-certified performance.

**Construct validity.** Success is a binary 0.5 m radius; it is coarse. MSE and final distance are reported alongside it. The trajectory MSE caveat is documented: a low trajectory MSE does not imply successful navigation because of the post-exit zero-error bypass.

**Statistical validity.** The McNemar test is paired and the Wilson intervals are reported. The pooled interval reflects training-seed variability, not independent episodes; this is stated explicitly.

<!-- pagebreak -->
# Chapter 7 — Limitations

1. **Simulation only.** No physical UAV, no software-defined radio, no flight test.
2. **Two-dimensional kinematics.** No altitude, attitude, wind, or rotor dynamics.
3. **Fixed controller.** The controller is not jointly learned, by design, to keep the comparison fair. A learned controller could change the absolute numbers.
4. **Perfect CSI in the main study.** CSI error is a separate ablation (Wave-B), not a default assumption.
5. **Flat Rayleigh fading in the main matrix.** No Doppler, shadowing, or 3GPP dispersion in the core results.
6. **λ-shaping is not a universal cure.** A-12 diverges under the full dispersion; A-13 restores stability only after resampling the outage tail and does not move the operating point.
7. **The 0.99+ result is at the task ceiling.** It proves the policy works but cannot resolve small differences between good methods.
8. **Two Wave-A cells failed** with non-finite gradients. They are reported as failures, not retrained away.
9. **CSI success-rate insensitivity is coarse.** MSE exposes the 18% degradation that SR hides.
10. **Polar baseline.** A-11 used SC polar without rate matching; A-15 adds the full 5G NR chain and confirms the crossover, but the 5G code runs at a higher rate for A ≥ 20 (CRC-11), which is a cost to 5G, not a favour.
11. **A-13 coverage.** A-13 trains on 96.2% of the dispersion (100% LoS, 66% NLoS) and evaluates on 100%; it is not a full-dispersion training result.
12. **One aerial geometry.** The A-10 link budget uses one distance, height, and carrier frequency.

<!-- pagebreak -->
# Chapter 8 — Conclusion and Future Work

## 8.1 Conclusion

This thesis asked whether a DeepJSCC system trained directly on UAV goal-reaching performance beats reconstruction-oriented DeepJSCC and a conventional digital link under matched channel and bandwidth conditions. The answer is conditional and empirically grounded.

At low bandwidth (k ≤ 18) the neural joint source-channel link clearly outperforms the digital link; at high bandwidth (k ≥ 24) the digital link overtakes and dominates. The crossover is the primary result. The frozen task-oriented loss, with zero control and velocity penalties, fails at 2–5% success; adding λ_u = λ_v = 0.1 recovers 99.7% at k = 12 and 99.3–99.9% at k = 18 and 54, on both channels. The mechanism is gradient conditioning: the encoder gradient is bounded by about fourteen orders of magnitude, and capacity is not the explanation. The result survives a 5G NR polar comparator, a 3GPP aerial link budget, a CSI-error sweep, and a three-seed replication. The thesis also reports the negative results honestly: full-dispersion training diverges, the metric is at ceiling, and the high-bandwidth/high-SNR digital link reaches 1.000.

## 8.2 Contributions

1. **A matched-budget benchmark** for UAV state transmission: three communication methods, identical episodes, identical channel, identical bandwidth, both AWGN and Rayleigh.
2. **The bandwidth-crossover characterisation:** a direct, data-backed answer to when task-oriented/semantic communication should be used in a 6G UAV link.
3. **The λ-shaping diagnosis and mechanism:** why task-oriented training collapses without control/effort terms, and how a small penalty fixes it.
4. **A reproducible testbed:** 57 passing tests, CPU-only determinism, Docker/CI, and frozen configuration with provenance tags.

## 8.3 Future Work

1. **A harder task.** The current metric is at ceiling; a tighter goal radius, shorter horizon, or cluttered workspace would let the benchmark resolve small differences between good methods.
2. **Full-dispersion training.** A-12 diverges and A-13 only resamples the tail; a principled training scheme for the full 3GPP dispersion is open.
3. **λ attribution.** Which of λ_u and λ_v is the active term, and whether a single combined penalty is sufficient, is not yet isolated.
4. **Richer channel models.** Doppler, shadowing, aerial small-scale fading, antenna patterns, and multi-UAV interference.
5. **Learned controller.** A jointly learned controller could test whether the communication gain survives control co-design.
6. **M5 ablations.** Task-loss component removal, uncoded digital, and cross-channel training were deferred by the frozen protocol and remain open.
7. **Hardware validation.** A software-defined radio or UAV flight test would test external validity.

<!-- pagebreak -->
# References

3GPP. (2017). *Technical specification group radio access network; Study on enhanced LTE support for aerial vehicles* (TR 36.777). 3rd Generation Partnership Project.

3GPP. (2019). *Technical specification group radio access network; Study on channel model for frequencies from 0.5 to 100 GHz* (TR 38.901). 3rd Generation Partnership Project.

3GPP. (2023). *Technical specification group radio access network; Multiplexing and channel coding* (TS 38.212). 3rd Generation Partnership Project.

Bourtsoulatze, E., Kurka, D. B., & Gündüz, D. (2019). Deep joint source-channel coding for wireless image transmission. *IEEE Transactions on Cognitive Communications and Networking, 5*(3), 567–579. https://doi.org/10.1109/TCCN.2019.2919300

Diao, Y. (2025). *Task-oriented communication for edge intelligence enabled connected robotics systems* (Doctoral thesis, University of Glasgow). University of Glasgow Theses Repository.

Diao, Y., Meng, Z., Xu, X., She, C., & Zhao, P. G. (2024). Task-oriented source-channel coding enabled autonomous driving based on edge computing. In *Proceedings of the 17th International Workshop on Networked Robotics and Communication Systems (NetRobiCS 2024), IEEE INFOCOM Workshops*. https://doi.org/10.1109/INFOCOMWKSHPS61880.2024.10620735

Diao, Y., Zhang, Y., De Martini, D., Zhao, P. G., & Li, E. L. (2025). Task-oriented co-design of communication, computing, and control for edge-enabled industrial cyber-physical systems. *IEEE Journal on Selected Areas in Communications* (early access). Preprint: arXiv:2503.08661.

Diao, Y., Zhang, Y., She, C., Zhao, P. G., & Li, E. L. (2025). Aligning task- and reconstruction-oriented communications for edge intelligence. *IEEE Journal on Selected Areas in Communications* (early access). https://doi.org/10.1109/JSAC.2025.3559121

Fang, Z., Liu, Z., Wang, J., Hu, S., Guo, Y., Deng, Y., & Fang, Y. (2025). *Task-oriented communications for visual navigation with edge-aerial collaboration in low altitude economy* [Preprint]. arXiv:2504.18317.

Gündüz, D., Qin, Z., Aguerri, I. E., Dhillon, H. S., Yang, Z., Yener, A., Wong, K. K., & Chae, C.-B. (2023). Beyond transmitting bits: Context, semantics, and task-oriented communications. *IEEE Journal on Selected Areas in Communications, 41*(1), 5–41. https://doi.org/10.1109/JSAC.2022.3223408

Khuwaja, A. A., Chen, Y., Zhao, N., Alouini, M.-S., & Dobbins, P. (2018). A survey of channel modeling for UAV communications. *IEEE Communications Surveys & Tutorials, 20*(4), 2804–2821.

Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization. In *Proceedings of the 3rd International Conference on Learning Representations (ICLR)*.

Kurka, D. B., & Gündüz, D. (2021). Bandwidth-agile image transmission with deep joint source-channel coding. *IEEE Transactions on Wireless Communications, 20*(12), 8081–8095.

McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika, 12*(2), 153–157.

Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S. (2019). PyTorch: An imperative style, high-performance deep learning library. In *Advances in Neural Information Processing Systems 32 (NeurIPS)*.

Shao, J., Mao, Y., & Zhang, J. (2022). Learning task-oriented communication for edge inference: An information bottleneck approach. *IEEE Journal on Selected Areas in Communications, 40*(1), 197–211. https://doi.org/10.1109/JSAC.2021.3126087

Shiri, H., Park, J., & Bennis, M. (2019). *Remote UAV online path planning via neural network based opportunistic control* [Preprint]. arXiv:1910.04969.

Tang, M., Feng, C., & Quek, T. Q. S. (2025). Decentralized semantic communication and cooperative tracking control for a UAV swarm over wireless MIMO fading channels. *IEEE Transactions on Vehicular Technology* (early access). Preprint: arXiv:2411.06136.

Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. *Journal of the American Statistical Association, 22*(158), 209–212.

Wu, H., Shao, Y., Bian, C., Mikolajczyk, K., & Gündüz, D. (2024). Deep joint source-channel coding for adaptive image transmission over MIMO channels. *IEEE Transactions on Wireless Communications, 23*(10), 15002–15017.

Wu, W., Yang, Y., Deng, Y., & Aghvami, A. H. (2024). *Goal-oriented semantic communications for robotic waypoint transmission: The value and age of information approach* [Preprint]. arXiv:2312.13182.

Xu, J., Ai, B., Chen, W., Yang, A., Sun, P., & Rodrigues, M. (2022). Wireless image transmission using deep source channel coding with attention modules. *IEEE Transactions on Circuits and Systems for Video Technology, 32*(4), 2315–2328.

Xu, Y., Zhou, H., & Deng, Y. (2023). Task-oriented semantics-aware communication for wireless UAV control and command transmission. *IEEE Communications Letters, 27*(8), 2232–2236. https://doi.org/10.1109/LCOMM.2023.3290109

Yang, M., Bian, C., & Kim, H.-S. (2022). OFDM-guided deep joint source-channel coding for wireless multipath fading channels. *IEEE Transactions on Cognitive Communications and Networking, 8*(2), 584–599.

<!-- pagebreak -->
# Appendix A — Frozen Configuration

**Table A.1 — Frozen configuration.**

| Parameter | Value | Provenance |
|---|---|---|
| State dimension n | 6 | FROZEN |
| Time step Δt | 0.1 s | PROJ |
| Horizon T | 100 steps | PROJ |
| Workspace | ±8 m | PROJ |
| Success radius | 0.5 m | PROJ |
| Initial distance | ≥4 m | PROJ |
| Velocity limit | 5 m/s | PROJ |
| Acceleration limit | 2 m/s² | PROJ |
| Controller gains | K_p = 1.0, K_v = 2.0 | FROZEN |
| Transmit power P | 1.0 | FROZEN |
| SNR training range | U(0, 20) dB | LIT + PROJ |
| SNR test grid (M1) | 0, 2, …, 20 dB | PROJ |
| SNR test grid (M2) | 0, 5, 10, 15, 20 dB | PROJ |
| Bandwidth k (M1) | 1, 2, 3 | FROZEN |
| Bandwidth k (M2) | 12, 18, 24, 30, 54 | FROZEN |
| Digital bits B | k=12→1, 18→2, 24→3, 30→4, 54→8 | FROZEN |
| Encoder width | 128 | FROZEN |
| Reconstruction steps | 5,000 | FROZEN |
| Task steps | 3,000 | FROZEN |
| Reconstruction batch | 4,096 states | FROZEN |
| Task batch | 256 episodes | FROZEN |
| Optimizer | Adam, lr = 1e-3 | FROZEN |
| Gradient clip | 1.0 | FROZEN |
| Training seeds | 42, 43, 44 | FROZEN |
| Test seed | 10042 | FROZEN |
| Test episodes | 5,000 per unit (M2, Wave-B) | FROZEN |
| Test episodes | 1,000 per unit (Wave-A, Wave-C, A-11, A-15) | PROJ |

# Appendix B — Protocol Amendments

The frozen protocol was extended only through dated, pre-registered amendments. Each amendment states its question, design, budget, and what is not changed before its first cell is run.

**Table B.1 — Protocol amendments A-1…A-15.**

| ID | Question | Status |
|---|---|---|
| A-1…A-8 | Wave-A/B/C infrastructure and design | complete |
| A-9 | Is the failure a capacity artefact? | complete (negative) |
| A-10 | Does λ hold under a 3GPP aerial link? | complete |
| A-11 | Is the digital baseline one convolutional code? | complete |
| A-12 | Does full-dispersion training help? | complete (negative, kept) |
| A-13 | Does outage-tail resampling stabilise training? | complete |
| A-14 | Is k = 24 a one-initialisation artefact? | complete |
| A-15 | Does a full 5G NR polar comparator change the crossover? | complete |

# Appendix C — Reproduction and File Map

**Table C.1 — Reproduction and file map.**

| Path | Contents |
|---|---|
| `project/config.py` | Every frozen constant with provenance tags |
| `project/uav/` | Dynamics, controller, rollouts |
| `project/channels/` | AWGN, Rayleigh, 3GPP aerial layers |
| `project/models/` | DeepJSCC encoder and decoder |
| `project/training/` | Reconstruction and task-oriented trainers |
| `project/baselines/digital.py` | Conventional digital chain |
| `project/evaluation/` | Metrics, statistics, paired harness |
| `results/M1/`, `results/M2/` | Main benchmark artifacts |
| `results/WAVE_A/`, `results/WAVE_B/`, `results/WAVE_C/` | Diagnostic and robustness waves |
| `results/ARCH_ABL/`, `results/K24_SEEDS/`, `results/AERIAL/`, `results/AERIAL_TRAIN_TRUNC/`, `results/POLAR/`, `results/POLAR5G/` | Extensions A-9…A-15 |
| `tests/` | 57 passing tests |
| `fyp_md_files/figures/` | All figures |

**Commands.** `python main.py --help` for the experiment runner; `python -m pytest tests/ -q` for the suite. Full setup is in `SETUP.md`; the architecture is in `ARCHITECTURE.md`.

# Appendix D — Disclosure Paragraphs

**D.1 Channel idealisations.** The Rayleigh channel is slow block fading with one complex gain h ~ CN(0,1) per k-symbol packet. The receiver has perfect CSI and applies ZF equalisation; the transmitter has no CSI. These idealisations are standard in the DeepJSCC literature and are applied identically to all three methods. Wave-B injects estimation error ε ~ CN(0, σ²) and finds success rate insensitive up to σ = 0.4 while reconstruction MSE degrades by about 18%.

**D.2 Digital baseline.** The digital baseline uses scalar quantisation (B bits per component), a convolutional code with K = 7 and rate 1/2, Gray QPSK, and soft-decision Viterbi decoding. The frozen (k, B) pairs satisfy 6B + 6 = k exactly; infeasible pairs are declared, not widened. The SC polar decoder is a low-complexity reference; A-15 adds the full 5G NR chain with CA-SCL L = 8.

**D.3 Statistical methodology.** All methods are evaluated on bitwise-identical episodes. Binary success uses the McNemar test; continuous metrics use the paired t-test. Wilson 95% intervals are reported. Three seeds are used; pooled M2 results aggregate 15,000 episodes per condition. The pooled interval reflects training-seed variability, not independent episodes.

**D.4 Training reproducibility.** All training and evaluation are CPU-only and deterministic. A given (seed, configuration) pair reproduces bit-identically. The final checkpoint is used, not a best-validation checkpoint. Interrupted runs are retrained from the fixed seed under the frozen protocol.

**D.5 Wave-A diagnostic.** The frozen task loss uses λ_d = 1, λ_T = 2, λ_u = λ_v = 0. Wave-A shows that enabling control/effort penalties (λ_u = λ_v > 0) recovers near-perfect success (0.997 at k = 12, both channels) from the frozen baseline of 0.02–0.05. The frozen M1/M2 results are preserved unmodified; Wave-A is presented as a diagnostic contribution.

# Appendix E — Viva Preparation

**Q: Which UAV paper did you use as the reference?**
A: Fang et al. (2025), arXiv:2504.18317, is the UAV navigation reference. Diao et al. (2024, 2025) is the method reference, and Shiri et al. (2019) is the UAV dynamics reference.

**Q: Why not use Fang et al. directly?**
A: Fang et al. transmit multi-camera image features and estimate position over a Shannon-capacity link. We transmit a six-dimensional kinematic state and close the loop through a controller to a goal, over a differentiable symbol-level AWGN/Rayleigh channel.

**Q: What is task-oriented here?**
A: The encoder and decoder are trained on the navigation rollout loss — distance, control effort, velocity, and terminal distance — not on reconstruction MSE.

**Q: Why did the task arm fail at first?**
A: The frozen loss had λ_u = λ_v = 0. The encoder gradient exploded to ~10^14, and the representation was not constrained to be useful for the controller. Setting λ_u = λ_v = 0.1 fixed it.

**Q: Is 99.7% too good to be true?**
A: We checked three controls: an untrained model scores 0.016; at −20 dB success drops to 0.279; the λ-light Rayleigh arm is degenerate at 0.019. The policy also ends two orders of magnitude closer to the goal while using less than half the control effort.

**Q: Does semantic communication always win?**
A: No. It wins at low bandwidth and under fading. At k ≥ 24 the classical link overtakes; at k = 54 and 20 dB it reaches 1.000 while we reach 0.998.

**Q: What is the main contribution?**
A: A matched-budget three-way benchmark, the bandwidth-crossover characterisation, and the λ-shaping diagnosis with a measured gradient mechanism.

**Q: What would you do next?**
A: A harder task to escape the ceiling, a principled full-dispersion training scheme, λ-term attribution, and richer channel models including Doppler.

---

*End of thesis master document.*
