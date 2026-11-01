# Lacuna Literature Map — Task-Oriented DeepJSCC for UAV Navigation over 3GPP UMa-Aerial Channels

**Prepared:** 1 November 2026  
**Method:** Direct Lacuna queries (no MCP), followed by source-page reads.  
**Scope:** Chapter 2 (Literature Review), gap analysis, and the planned IEEE paper.  
**Status:** Working map. Every item must be checked against the primary paper before it enters the thesis reference list.

---

## 0. Quick read

- The thesis gap still stands. In the queries run, **no paper surfaced** that combines all five elements at once: a UAV kinematic state, closed-loop goal-reaching, a differentiable symbol-level DeepJSCC codec, a matched-bandwidth digital baseline, and a 3GPP UMa-aerial channel.
- The closest new works are **Path-SC** (UAV/UGV cooperative path planning with task-aware semantic transmission), **SC-GIR** (invariant task-oriented representations over AWGN/Rayleigh), **JTD-SC/DSSCC** (joint task-and-data semantic communication with a deep separate source-channel code), **Robust Deep JSCC** (SNR-mismatch brittleness), and **Conformal Semantic Communication** (task-level guarantees under channel shift).
- The most useful additions to Chapter 2 are **Path-SC**, **SC-GIR**, **JTD-SC/DSSCC**, **Khina et al. (2016)** (analog JSCC for control), and **Pan et al. (2024)** (time-varying channels). These sharpen the UAV, separation-vs-joint, control, and Doppler arguments respectively.
- The IEEE paper needs one extra related-work paragraph on **joint vs separate coding at low bandwidth** and one on **robustness under channel shift**. The current thesis already has the DeepJSCC and task-oriented foundations; these additions make the positioning harder to attack.
- **Do not claim “first”** on the basis of this map. Claim only: “to our knowledge, no prior work combines …”, and only after one final exact-phrase search before submission.

---

## 1. What Lacuna is, and what it is not

Lacuna is a research map for machine learning and AI built by Tiptree Systems. At the time of access it indexed roughly **2.23 million papers**, **28,588 research directions**, **215,520 authors**, and **5,632 venues**. It groups papers into research directions, extracts concepts, and links each generated item back to its sources.

For this project it is useful because it surfaces **clusters and neighbouring literatures** that a keyword search in a communications database can miss, especially at the boundary of task-oriented communication, control, and multi-task loss weighting.

It is **not** a substitute for IEEE Xplore, the 3GPP specifications, or a primary-source read. Its summaries and concept elements are generated. Venue and year metadata can disagree across versions (for example, early access versus final publication). Treat every entry below as a **lead**, not as a verified citation.

The counts in this document are Lacuna cluster sizes, not the number of papers that satisfy a strict inclusion criterion.

---

## 2. Map to the thesis

| Thesis part | Question it answers | What this map adds | Best sources |
|---|---|---|---|
| §2.1 DeepJSCC foundations | Why joint source-channel coding at all? | A direct “separate coding can beat DJSCC at low bandwidth for joint task+data” counterpoint | Bourtsoulatze 2019; Kurka & Gündüz 2021; Huang et al. 2023 |
| §2.2 Task-oriented communication | What is already task-oriented? | Invariant representations, distortion resilience, 6G framing, SNR-mismatch robustness, channel-shift guarantees | Shao et al. 2022; Wanasekara et al. 2025; Nguyen et al. 2025; Park et al. 2025; Batra et al. 2026 |
| §2.3 UAV-specific work | Who else does UAV semantic communication? | Path-SC is the closest UAV path-planning semantic system; UAV resource-allocation and hybrid-RL work broaden the UAV context | Fang et al. 2025; Zhao et al. 2025; Fair Resource Allocation 2025; UAV-assisted Hybrid RL 2023 |
| §2.4 JSCC for control | Why is control different from classification? | A clean theoretical precedent: analog JSCC for unstable plants over AWGN, with LQG cost and stability margins | Khina et al. 2016; Diao et al. 2024/2025 |
| §2.5 UAV channel models | Why 3GPP UMa-aerial? | UAV channel survey, NTN hybrid fading, time-varying JSCC, and the quasi-static/Doppler gap | Khuwaja et al. 2018; 3GPP TR 36.777; 3GPP TR 38.901; Pan et al. 2024 |
| §5.3–5.4 λ-shaping | Why does the frozen task loss fail? | Multi-task loss-balancing literature gives the general mechanism; the thesis gives the control-specific instance | Adaptive Loss Balancing direction; DRGO; Wave-A/Wave-C |
| §6.6 and Chapter 7 | What can break the claim? | Channel shift, SNR mismatch, time-varying channels, single-geometry 3GPP model | Park et al. 2025; Batra et al. 2026; Pan et al. 2024 |

---

## 3. Closest works and the exact delta

| Work | Venue / year | What it does | Overlap with this thesis | Delta / gap | Cite in |
|---|---|---|---|---|---|
| **Bourtsoulatze, Kurka & Gündüz** `[P1]` | IEEE TCCN, 2019 | DeepJSCC for wireless image transmission; AWGN and slow Rayleigh; beats JPEG/JPEG2000 + capacity-achieving coding at low SNR / small bandwidth | Same AWGN/Rayleigh protocol; same “joint vs separate” question | Reconstruction objective, image source, no control loop | §2.1, §3.9 |
| **Kurka & Gündüz** `[P2]` | IEEE TWC, 2021 | Bandwidth-agile DeepJSCC; successive refinement and multiple descriptions | Bandwidth scaling is central to the thesis crossover | Image layers, not a task; no digital baseline at matched k | §2.1, §3.5 |
| **Shao, Mao & Zhang** `[P3]` | IEEE JSAC, 2022 | Task-oriented VIB for edge inference; channel-adaptive feature length | Task objective instead of reconstruction | Classification, image source, no closed-loop control | §2.2 |
| **Fang et al.** `[P4]` | arXiv, 2025 | UAV visual navigation with edge-aerial collaboration; O-VIB encoder; localisation at the edge | UAV navigation and task-oriented compression | Image source; capacity-level channel; localisation, not goal-reaching | §2.3 |
| **Zhao et al. (Path-SC)** `[P5]` | arXiv, 2025; IEEE TVT, 2026 | UAV/UGV cooperative path planning; DynamicViT + Local Betweenness Centrality; sends task-critical map regions; up to 54% data reduction vs JPEG; robust at low SNR | UAV navigation, task-aware transmission, bandwidth-constrained and noisy link | Image/map source; path planning on a ground vehicle; no 6-D UAV state; no symbol-level DeepJSCC; no 3GPP aerial model | §2.3, Table 2.1 |
| **Wanasekara et al. (SC-GIR)** `[P6]` | IEEE TMC, 2025 | Self-supervised invariant representation for goal-oriented semantic communication; AWGN and Rayleigh; ~85% accuracy at k/n = 0.1 under AWGN, ~80% under Rayleigh | Task-oriented representation; fading robustness; extreme bandwidth compression | Image classification/segmentation; no control; no matched digital baseline across k | §2.2, §5.5 |
| **Huang et al. (JTD-SC / DSSCC)** `[P7]` | IEEE IoT Journal, 2023/2024 | Joint task-and-data semantic communication with a deep **separate** source-channel code; compares against DJSCC; reports gains at low channel bandwidth ratio | Directly relevant to the separation-vs-joint argument and the digital baseline | Image transmission + classification; no control; no UAV channel | §2.1, §2.2, §6.3 |
| **Nguyen et al. (DRGO)** `[P8]` | IEEE TMC, 2025 | “Distortion resilience” for goal-oriented semantic communication; unified semantic-distortion metric; DRL for bandwidth/compression/power | Task metric and bandwidth trade-off; supports the thesis’s task-loss framing | Classification on CIFAR-10/MNIST; no control loop; no 3GPP aerial | §2.2, §5.3 |
| **Park et al. (Robust Deep JSCC)** `[P9]` | arXiv, 2025 | Robust DeepJSCC for task-oriented semantic communication; targets brittleness when test SNR differs from training SNR | Directly relevant to Wave-B (CSI error) and the channel-shift threat | Image task; no control; no 3GPP aerial; preprint | §2.2, §5.5, §6.6 |
| **Batra et al. (Conformal Semantic Communication)** `[P10]` | AI4NextG @ ICML, 2026 | Distribution-free, finite-sample task-level coverage guarantees under channel distribution shift | Formal language for the thesis’s channel-shift limitation | Classification/image setting; guarantees are not control-theoretic; very recent | §6.6, Chapter 7 |
| **Khina et al.** `[P11]` | arXiv / CDC, 2016 | Analog JSCC for unstable plants over AWGN; multi-rate control; LQG cost and stability margins; JSCC beats separation at short block lengths | The cleanest theoretical precedent for JSCC in a control loop | Scalar plant; analog maps; no deep learning; no UAV | §2.4 |
| **Pan et al. (DRJSCC)** `[P12]` | WCNC, 2024 | Deep refinement-based JSCC over time-varying channels; block-wise re-encoding under CSI change | Directly relevant to the missing Doppler/time-correlated-shadowing extension | Image transmission; assumes CSI feedback; no control | §2.5, Chapter 7 |
| **Calvanese Strinati et al. (6G-GOALS)** `[P13]` | EuCNC/6G Summit, 2024 | Goal-oriented and semantic communication for AI-native 6G; semantic plane, O-RAN, DeepJSCC, VoI/AoI | Strong framing citation for the 6G motivation | Vision/architecture paper, not a method or benchmark | §2.2, §1.1 |

### The one-sentence positioning

> The literature contains task-oriented image communication, UAV semantic navigation, and JSCC for control, but not a single study that trains a symbol-level DeepJSCC codec on a **6-D UAV kinematic state** for **closed-loop goal-reaching**, compares it against a **matched-bandwidth digital link** across a bandwidth sweep, and repeats the evaluation over **AWGN, Rayleigh, and a 3GPP UMa-aerial link budget**.

That sentence is the thesis’s defensible gap. Keep it, but verify it with one final search before submission.

---

## 4. Research directions worth monitoring

These are clusters, not individual papers. They are useful for the viva question “what else is happening in this area?” and for the IEEE paper’s related-work section.

| Direction | Size | Why it matters here | Link |
|---|---:|---|---|
| Task Oriented Semantic Communication | 635 papers, 1,061 concepts | The core cluster; covers SKB, task metrics, semantic communication | `lacuna.tiptreesystems.com/direction/task-oriented-semantic-communication/txn_78b3b57c4ea54109a253288ad2470105` |
| Modular Separation of Source and Channel Coding | 78 papers, 129 concepts | The classical baseline and the separation principle the thesis tests | `lacuna.tiptreesystems.com/direction/modular-separation-of-source-and-channel-coding/txn_a067c66196d247be94d3e1cf17292e14` |
| End-to-End Neural Joint Source-Channel Coding | 25 papers, 25 concepts | Foundational DeepJSCC lineage | `lacuna.tiptreesystems.com/direction/end-to-end-neural-joint-source-channel-coding/txn_9cb33d9432264352944a987c135445ef` |
| Hybrid Semantic and Bit Communication Networks | 16 papers, 25 concepts | Directly relevant to the crossover: when to use semantic vs bit pipes | `lacuna.tiptreesystems.com/direction/hybrid-semantic-and-bit-communication-networks/txn_1d8e26783dd942caa7552a05f0261abc` |
| Adaptive Loss Balancing in Multi-Task Learning | 712 papers, 940 concepts | General mechanism behind λ-shaping | `lacuna.tiptreesystems.com/direction/adaptive-loss-balancing-in-multi-task-learning/txn_d5823272f93245dab792f7f08b2eb8bb` |
| Hybrid Channel Fading Modeling for Non-Terrestrial Networks | 74 papers, 94 concepts | UAV/NTN channel modelling beyond Rayleigh/Rician | `lacuna.tiptreesystems.com/direction/hybrid-channel-fading-modeling-for-non-terrestrial-networks/txn_77acc15bb9b4498eaeed25e33f4c0c1f` |
| Overcoming Idealized Assumptions in Channel Modeling | 741 papers, 762 concepts | Supports the thesis’s “disclosure of idealizations” and limitations | `lacuna.tiptreesystems.com/direction/overcoming-idealized-assumptions-in-channel-modeling/txn_98106c980e594d08b0bd59f7e9b37027` |
| Tracking Time-Varying Channels Beyond the Quasi-Static Assumption | 758 papers | The Doppler / time-correlated-shadowing gap | `lacuna.tiptreesystems.com/direction/tracking-time-varying-channels-beyond-the-quasi-static-assumption/txn_53d4ac02cf3049b7882ba53724fe51a9` |
| Mitigating Wireless Channel Estimation Error Sensitivity | 485 papers, 516 concepts | Wave-B context: CSI error and robust receivers | `lacuna.tiptreesystems.com/direction/mitigating-wireless-channel-estimation-error-sensitivity/txn_83edf6868a734b56901308f8c8f15cec` |
| Balancing Line of Sight and Interference in Aerial Networks | 145 papers | Aerial link geometry and LoS/NLoS trade-offs | `lacuna.tiptreesystems.com/direction/balancing-line-of-sight-and-interference-in-aerial-networks/txn_7da73faffcf140ad8c70e6533c6abf15` |

---

## 5. Gap matrix

| Gap | What the thesis currently has | What is missing | Action | Priority |
|---|---|---|---|---|
| Exact combination of UAV state + closed-loop control + matched bandwidth + 3GPP aerial | A defensible gap statement in §2.6 and Table 2.1 | A final exact-phrase search and a “to our knowledge” sentence | Run the four exact-phrase queries in §8; add the sentence only if clean | P0 |
| 3GPP multi-geometry and Doppler | One geometry: d2D = 500 m, h = 50 m, fc = 2 GHz; no Doppler; no time-correlated shadowing | A-10 extension over geometry/height/frequency; Doppler model | Add A-10 multi-geometry sweep; cite Pan et al. 2024 and the NTN fading direction | P1 |
| M5 task-loss component removal | M5 specified but deferred; no `results/M5` | A-16 amendment and M5a run | Write A-16, run M5a first (remove task-loss component), then M5b/M5c if approved | P0/P1 |
| Digital baseline strength | Polar/LDPC-style digital link and A-11/A-15 checks | Uncoded digital baseline and a clearer “weak baseline” rebuttal | M5b; report both coded and uncoded digital | P1 |
| Statistical correction | McNemar and Wilson intervals | Family-wise correction across many comparisons | Add Holm–Bonferroni or a preregistered primary endpoint | P1 |
| Channel shift / domain generalisation | AWGN, Rayleigh, 3GPP; no test-time adaptation | Robustness under distribution shift; conformal/robust baselines | Cite Park et al. 2025 and Batra et al. 2026; add a shift experiment if time permits | P1/P2 |
| Task metric richness | Goal-reaching success | Continuous control cost (LQG-style), time-to-goal, safety margin | Add a secondary control-cost metric | P2 |
| Bandwidth range | k = 1, 2, 3, 6, 12, 18, 24, 54; B = 1 and B = 8 | Intermediate k and more seeds around the crossover | Add k = 9, 15, 21, 30, 36, 48; more seeds at k = 18/24 | P1 |
| λ attribution | Wave-A and Wave-C | A λ sweep and component-removal ablation | Add λ ∈ {0.01, 0.05, 0.1, 0.2, 0.5}; component removal | P1 |
| Hardware validation | Simulation only | SDR/over-the-air test | Plan a small SDR testbed after submission | P2 |

---

## 6. Suggested thesis edits

These are small, high-value edits. They do not change the results or the frozen protocol.

### 6.1 §2.1 — add the separation counterpoint

After the Kurka & Gündüz sentence, add:

> A useful counterpoint comes from Huang et al. (2023), who propose a deep **separate** source-channel code for joint task-and-data semantic communication and report gains over DJSCC at low channel bandwidth ratios. This does not contradict the present result; it sharpens the question. The crossover is not “joint always wins” versus “separate always wins,” but a regime-dependent comparison under a matched bandwidth budget.

### 6.2 §2.2 — add robustness and channel shift

Add a short paragraph:

> Recent work has begun to address the brittleness of task-oriented JSCC under channel mismatch. Park et al. (2025) show that task-oriented DeepJSCC degrades when the test SNR differs from the training SNR, and Batra et al. (2026) introduce distribution-free task-level coverage guarantees under channel distribution shift. These results frame the present Wave-B and 3GPP-dispersion experiments as a control-specific instance of a broader robustness question.

Also add SC-GIR and DRGO:

> Wanasekara et al. (2025) learn invariant representations with self-supervised contrastive learning and report strong classification accuracy at extreme compression under AWGN and Rayleigh fading. Nguyen et al. (2025) replace pure accuracy with a “distortion resilience” metric and optimise bandwidth, compression, and power jointly. Both support the thesis’s claim that the task metric, not reconstruction fidelity, should drive the communication design.

### 6.3 §2.3 — add Path-SC

Add after Fang et al.:

> Zhao et al. (2025) propose Path-SC, a task-oriented semantic framework for UAV/UGV cooperative path planning. It identifies map regions that are critical for route choice and transmits them preferentially, reducing data volume by up to 54% relative to JPEG while remaining robust at low SNR. Path-SC is the closest UAV path-planning semantic system to the present work, but it transmits image/map content rather than a 6-D kinematic state, and it does not compare against a matched-bandwidth digital link over a 3GPP UMa-aerial channel.

### 6.4 §2.4 — add the control-theoretic precedent

Add before the Diao discussion:

> The control-theoretic case for JSCC is older than deep learning. Khina et al. (2016) show that, for an unstable plant over an AWGN channel with more channel uses than source samples, analog JSCC can improve stability margins and LQG cost relative to separation or repetition. The present work can be read as a learned, high-dimensional, task-oriented version of that argument.

### 6.5 §2.5 — add the time-varying gap

Add at the end of the 3GPP paragraph:

> Time-varying extensions are also beginning to appear in the JSCC literature. Pan et al. (2024) re-encode later blocks when CSI changes and report gains over static DeepJSCC under Rayleigh fading. The present 3GPP extension is deliberately quasi-static; adding Doppler and time-correlated shadowing is the natural next step.

### 6.6 Table 2.1 — add four rows

Add rows for **Path-SC**, **SC-GIR**, **JTD-SC/DSSCC**, and **Khina et al. (2016)**. The “Limitation relative to this thesis” column should stay short: image/map source, classification not control, separate coding not a control loop, scalar plant not UAV.

### 6.7 References — APA entries

Add these after verifying the final venue, year, and DOI:

- Huang, J., Li, D., Huang, C., Qin, X., & Zhang, W. (2023/2024). Joint task and data oriented semantic communications: A deep separate source-channel coding scheme. *IEEE Internet of Things Journal*.
- Khina, A., Pettersson, G. M., Kostina, V., & Hassibi, B. (2016). Multi-rate control over AWGN channels via analog joint source-channel coding. *arXiv* / *CDC 2016*.
- Nguyen, M.-D., Vinh, Q. D., Yang, Z., Pham, Q.-V., & Hwang, W.-J. (2025). Distortion resilience for goal-oriented semantic communication. *IEEE Transactions on Mobile Computing*.
- Pan, J., Li, H., Zhang, G., Cai, Y., & Yu, G. (2024). Deep refinement-based joint source channel coding over time-varying channels. *IEEE WCNC 2024*.
- Park, T., Hong, E., Jeon, Y.-S., Lee, N., & Kim, Y. (2025). Robust deep joint source channel coding for task-oriented semantic communications. *arXiv*.
- Wanasekara, S. H., Nguyen, V.-D., Wong, K.-S., Nguyen, M.-D., Chatzinotas, S., & Dobre, O. A. (2025). SC-GIR: Goal-oriented semantic communication via invariant representation learning for image transmission. *IEEE Transactions on Mobile Computing*.
- Zhao, F., Sun, Y., Lan, J., Zhang, L., Liu, X., & Imran, M. A. (2025). Adaptive semantic communication for UAV/UGV cooperative path planning. *arXiv*; *IEEE Transactions on Vehicular Technology*, 2026.
- Calvanese Strinati, E., Di Lorenzo, P., Sciancalepore, V., Aijaz, A., Kountouris, M., Gündüz, D., Popovski, P., et al. (2024). Goal-oriented and semantic communication in 6G AI-native networks: The 6G-GOALS approach. *EuCNC/6G Summit 2024*.
- Batra, S., Karuturi, S., Bukkapatnam, K. S., Patel, L., Shastry, T. A., & Park, M. (2026). Conformal semantic communication: Distribution-free task-level coverage guarantees for goal-oriented transmission under channel shift. *AI4NextG @ ICML 2026*.

**Caution:** the Lacuna entries for some of these are preprints or early-access versions. Use the final published version if it exists; do not cite a preprint as a journal paper without checking.

---

## 7. IEEE paper plan

### 7.1 Suggested title

> **When Does Task-Oriented DeepJSCC Beat Digital Transmission? A Matched-Bandwidth Study for UAV Navigation over 3GPP UMa-Aerial Channels**

This is more precise than “Task-Oriented DeepJSCC for UAV Navigation” because the contribution is the **crossover**, not the codec alone.

### 7.2 Six-page structure

| Section | Pages | Content |
|---|---:|---|
| I. Introduction | 0.75 | 6G goal-oriented vision; UAV navigation; the matched-bandwidth question; contributions |
| II. Related work | 1.0 | DeepJSCC; task-oriented semantic; UAV semantic navigation; JSCC for control; 3GPP aerial channels |
| III. System model | 1.0 | UAV kinematics; controller; channel model (AWGN, Rayleigh, 3GPP UMa-aerial); DeepJSCC encoder/decoder; digital baseline |
| IV. Training and evaluation | 0.75 | λ-shaped task loss; matched bandwidth; metrics; statistical tests |
| V. Results | 1.5 | Crossover; M3 boundary; M4 B = 8; Wave-A mechanism; Wave-B CSI; Wave-C generalisation; 3GPP A-10/A-13 |
| VI. Discussion and limitations | 0.75 | Why the crossover exists; separation vs joint; single geometry; no Doppler; simulation-only |
| VII. Conclusion | 0.25 | What practitioners should take away |
| References | 1.0 | 20–25 numeric references |

### 7.3 Numeric citation starter set

Use this as a working mapping, not a final bibliography:

1. Bourtsoulatze et al., DeepJSCC, 2019.
2. Kurka & Gündüz, bandwidth-agile DeepJSCC, 2021.
3. Gündüz et al., beyond transmitting bits, 2023.
4. Shao, Mao & Zhang, task-oriented VIB, 2022.
5. Fang et al., UAV visual navigation, 2025.
6. Zhao et al., Path-SC, 2025/2026.
7. Wanasekara et al., SC-GIR, 2025.
8. Huang et al., JTD-SC/DSSCC, 2023/2024.
9. Park et al., robust DeepJSCC, 2025.
10. Batra et al., conformal semantic communication, 2026.
11. Khina et al., analog JSCC for control, 2016.
12. 3GPP TR 36.777, 2017.
13. 3GPP TR 38.901, 2019.
14. Khuwaja et al., UAV channel survey, 2018.
15. Pan et al., DRJSCC over time-varying channels, 2024.
16. Nguyen et al., DRGO, 2025.
17. Calvanese Strinati et al., 6G-GOALS, 2024.
18. 3GPP TS 38.212, 2023.
19. Shiri, Park & Bennis, UAV path planning, 2019.
20. Diao et al., task-oriented JSCC for driving/CPS, 2024/2025.

### 7.4 Related-work paragraph draft

> **DeepJSCC and separation.** Bourtsoulatze et al. [1] showed that a jointly trained encoder/decoder can outperform a separation-based image pipeline at low SNR and small bandwidth, and Kurka and Gündüz [2] extended the idea to bandwidth-agile layered transmission. The comparison is not universally one-sided: Huang et al. [8] report that a deep separate source-channel code can beat DJSCC for joint task-and-data semantic communication at low channel bandwidth ratios. This motivates a matched-budget comparison rather than a blanket “joint wins” claim.
>
> **Task-oriented and UAV communication.** Gündüz et al. [3] and Shao et al. [4] formalise task-oriented communication around a downstream objective rather than reconstruction. UAV-specific work includes Fang et al. [5], who compress multi-camera features for edge-aerial localisation, and Zhao et al. [6], whose Path-SC system prioritises navigation-critical map regions and reduces data volume by up to 54% versus JPEG. Both are image/map systems; neither trains a symbol-level codec on a UAV kinematic state for closed-loop goal-reaching.
>
> **Robustness and control.** Park et al. [9] show that task-oriented DeepJSCC is brittle under SNR mismatch, and Batra et al. [10] provide distribution-free coverage guarantees under channel shift. In control, Khina et al. [11] show that analog JSCC can improve stability margins and LQG cost relative to separation at short block lengths. The present work connects these lines: it asks when a learned task-oriented JSCC link beats a digital link for a closed-loop UAV task, and what happens when the channel is a 3GPP UMa-aerial link rather than AWGN or Rayleigh.

### 7.5 Claims to avoid

- Do not write “first task-oriented JSCC for UAV navigation.” Path-SC and Fang et al. are close enough that a reviewer will object.
- Do not write “3GPP-compliant.” The thesis uses a 3GPP **link-budget model**, not a full 3GPP-compliant simulator.
- Do not write “digital is always worse at low bandwidth.” The thesis shows a crossover, and M3 shows a feasibility boundary.
- Do not write “λ-shaping is novel.” Frame it as a **diagnosis** of why the frozen task loss fails, grounded in the multi-task loss-balancing literature.

---

## 8. Search log

These are the direct queries used. Counts are Lacuna’s semantic-search counts and include broad matches.

| Query | Results | Useful? |
|---|---:|---|
| task-oriented joint source-channel coding UAV | 233 | Yes; surfaced DeepJSCC, task-oriented, bandwidth-agile |
| task-oriented semantic communication UAV | 10 | Yes; core direction and recent UAV work |
| 3GPP UMa aerial channel UAV | 9,995 | Noisy; “UMA” and “channel” match generic ML terms |
| bandwidth-agile deep joint source-channel coding | 249 | Yes; Kurka & Gündüz and related |
| deep joint source-channel coding UAV navigation | 254 | Yes; foundational and UAV-adjacent |
| goal-oriented semantic communication closed-loop control | 204 | Mixed; surfaced 6G-GOALS and control-adjacent work |
| UAV channel modeling 3GPP aerial path loss | 48 | Yes; Khuwaja survey and NTN fading |
| semantic communication channel shift domain shift | 1,135 | Broad; useful for robustness framing |
| task loss weighting deep reinforcement learning | 3,947 | Broad; useful for λ-shaping context |
| bandwidth allocation semantic communication deep JSCC | 39 | Yes; resource-allocation work |
| UAV semantic communication goal reaching navigation | 31 | Yes; Path-SC and UAV semantic cluster |
| matched bandwidth deep JSCC digital baseline comparison | 2 | Too restrictive; not useful |
| task-oriented JSCC control state vector | 128 | Yes; analog JSCC for control and finite-length JSCC |

### Exact-phrase checks to run before submission

Run these as quoted phrases in Lacuna, IEEE Xplore, and Google Scholar:

1. `"UAV" "task-oriented" "joint source-channel coding" "goal-reaching"`
2. `"matched bandwidth" "deep JSCC" "digital" "UAV"`
3. `"3GPP UMa-aerial" "deep JSCC"`
4. `"kinematic state" "joint source-channel coding" "UAV"`

If any of these returns a direct match, update the gap statement. If none does, keep “to our knowledge” and cite the closest works.

---

## 9. MCP decision

For this pass, **direct query was enough**. The map above came from 13 searches and about a dozen source-page reads, with no setup.

Use the **Lacuna MCP** if we want a repeatable pipeline:

- search many queries in one run;
- pull paper context, directions, authors, and concept elements programmatically;
- build a citation database for the IEEE paper;
- re-run the gap check before submission.

The hosted endpoint is `https://lacuna.tiptreesystems.com/api/mcp` and is free / no-auth at the time of writing. Adding it to OpenCode requires a `mcp` entry in the OpenCode config and a restart. I can do that when we move from “map” to “full literature pipeline.”

**Recommendation:** keep direct queries for the next thesis edit. Add MCP only when we start the IEEE paper’s reference database, because that is where repeated structured search pays off.

---

## 10. Next actions

| Priority | Action | Output |
|---|---|---|
| P0 | Add Path-SC, SC-GIR, JTD-SC/DSSCC, Khina 2016, Pan 2024 to Chapter 2 | Revised §2.1–§2.5 + Table 2.1 + references |
| P0 | Run the four exact-phrase checks in §8 | Confirmed or revised gap statement |
| P0 | Verify venue/year/DOI for every new reference | Clean APA entries |
| P1 | Write protocol amendment A-16 and run M5a | `results/M5` + thesis subsection |
| P1 | Add A-10 multi-geometry sweep | 3GPP extension table/figure |
| P1 | Add Holm–Bonferroni or a preregistered primary endpoint | §4.11 revision |
| P1 | Draft the IEEE paper using §7 | 6-page draft |
| P2 | Add Doppler/time-correlated shadowing | A-17 or future-work section |
| P2 | Add a continuous control-cost metric | Secondary results table |
| P2 | Plan an SDR/over-the-air validation | Future-work paragraph |

---

## 11. Source key

### Directions

- `[D1]` Task Oriented Semantic Communication — 635 papers, 1,061 concepts. `lacuna.tiptreesystems.com/direction/task-oriented-semantic-communication/txn_78b3b57c4ea54109a253288ad2470105`
- `[D2]` Modular Separation of Source and Channel Coding — 78 papers, 129 concepts. `lacuna.tiptreesystems.com/direction/modular-separation-of-source-and-channel-coding/txn_a067c66196d247be94d3e1cf17292e14`
- `[D3]` End-to-End Neural Joint Source-Channel Coding — 25 papers, 25 concepts. `lacuna.tiptreesystems.com/direction/end-to-end-neural-joint-source-channel-coding/txn_9cb33d9432264352944a987c135445ef`
- `[D4]` Hybrid Semantic and Bit Communication Networks — 16 papers, 25 concepts. `lacuna.tiptreesystems.com/direction/hybrid-semantic-and-bit-communication-networks/txn_1d8e26783dd942caa7552a05f0261abc`
- `[D5]` Adaptive Loss Balancing in Multi-Task Learning — 712 papers, 940 concepts. `lacuna.tiptreesystems.com/direction/adaptive-loss-balancing-in-multi-task-learning/txn_d5823272f93245dab792f7f08b2eb8bb`
- `[D6]` Hybrid Channel Fading Modeling for Non-Terrestrial Networks — 74 papers, 94 concepts. `lacuna.tiptreesystems.com/direction/hybrid-channel-fading-modeling-for-non-terrestrial-networks/txn_77acc15bb9b4498eaeed25e33f4c0c1f`

### Papers

- `[P1]` Bourtsoulatze, Kurka & Gündüz (2019), DeepJSCC, IEEE TCCN. `lacuna.tiptreesystems.com/work/deep-joint-source-channel-coding-for-wireless-image-qygv`
- `[P2]` Kurka & Gündüz (2021), Bandwidth-agile DeepJSCC, IEEE TWC. `lacuna.tiptreesystems.com/work/bandwidth-agile-image-transmission-with-deep-joint-source`
- `[P3]` Shao, Mao & Zhang (2022), Task-oriented VIB, IEEE JSAC. (Thesis reference; no Lacuna page used.)
- `[P4]` Fang et al. (2025), UAV visual navigation, arXiv:2504.18317. (Thesis reference.)
- `[P5]` Zhao et al. (2025), Path-SC, arXiv; IEEE TVT 2026. `lacuna.tiptreesystems.com/work/adaptive-semantic-communication-for-uav-ugv-cooperative-path`
- `[P6]` Wanasekara et al. (2025), SC-GIR, IEEE TMC. `lacuna.tiptreesystems.com/work/sc-gir-goal-oriented-semantic-communication-via-invariant-48tq`
- `[P7]` Huang et al. (2023/2024), JTD-SC/DSSCC, IEEE IoT Journal. `lacuna.tiptreesystems.com/work/joint-task-and-data-oriented-semantic-communications`
- `[P8]` Nguyen et al. (2025), DRGO, IEEE TMC. `lacuna.tiptreesystems.com/work/distortion-resilience-for-goal-oriented-semantic-6pm7`
- `[P9]` Park et al. (2025), Robust Deep JSCC, arXiv. `lacuna.tiptreesystems.com/work/robust-deep-joint-source-channel-coding-for-task`
- `[P10]` Batra et al. (2026), Conformal Semantic Communication, AI4NextG @ ICML. `lacuna.tiptreesystems.com/work/conformal-semantic-communication-distribution-free-task`
- `[P11]` Khina et al. (2016), Analog JSCC for control, arXiv / CDC. `lacuna.tiptreesystems.com/work/multi-rate-control-over-awgn-channels-via-analog`
- `[P12]` Pan et al. (2024), DRJSCC over time-varying channels, IEEE WCNC. `lacuna.tiptreesystems.com/work/deep-refinement-based-joint-source-channel-coding`
- `[P13]` Calvanese Strinati et al. (2024), 6G-GOALS, EuCNC/6G Summit. `lacuna.tiptreesystems.com/work/goal-oriented-and-semantic-communication-in-6g-ai`

### Standards

- 3GPP TR 36.777 (2017), Study on enhanced LTE support for aerial vehicles.
- 3GPP TR 38.901 (2019), Study on channel model for frequencies from 0.5 to 100 GHz.
- 3GPP TS 38.212 (2023), Multiplexing and channel coding.

---

## 12. One-line answer for the viva

> “The literature has task-oriented image communication, UAV semantic navigation, and JSCC for control, but not a matched-bandwidth, symbol-level, closed-loop UAV study over a 3GPP UMa-aerial link. That is the gap this thesis fills, and the crossover is the result.”
