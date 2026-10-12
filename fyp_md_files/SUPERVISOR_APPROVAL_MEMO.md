# Memo — Supervisor Review Response & Requested Approvals

**Project:** Task-Oriented DeepJSCC for UAV Navigation under Noisy and Fading Wireless Channels
**Date:** October 2026
**Status requested from supervisor:** approval items A1–A4 below.

---

## 1._responses to the three review points

**1.1 "UAV khud ko message send kar rahi hai?" — Architecture clarification.**
The implemented system is **UAV-to-ground-station (uplink)**, not UAV-to-self. The UAV is the
transmitter of its 6-D kinematic state `s_t = [x, y, vx, vy, xg, yg]`; the **ground station** is a
separate receiver node that decodes a task-relevant representation and computes the control input
`u_t`, which is applied back to the UAV model. This models the standard UAV C&C scenario where
navigation computation is offloaded from a compute-constrained drone to the ground segment
(the same directional structure as the UAV command-and-control literature the proposal cites,
Section 14). The original topic statement described the symmetric downlink (ground sends a
command; the UAV infers intent); the implemented, defensible scope is the **state-report uplink**,
which is the bandwidth-critical direction of that same C&C loop. **Amendment requested (A3):**
add one explicit sentence to the system model — *"the return command downlink is assumed ideal and
out of scope; the wireless bottleneck under study is the uplink state transmission."*

**1.2 Research Question wording.**
The reviewer is right that RQ1 ("to what extent does task-oriented ... outperform ...") presumes the
direction of the result, and contradicts Section 22 of the same proposal ("no direction is presumed
in advance"). **Amendment requested (A1):** replace RQ1 with the neutral, conditional form:

> *RQ1 (revised): Under what conditions — channel type (AWGN vs Rayleigh), SNR, and bandwidth
> ratio — does task-oriented DeepJSCC trained directly on navigation performance outperform, match,
> or underperform conventional digital communication and reconstruction-oriented DeepJSCC for UAV
> state transmission?*

This reframing is not cosmetic: our completed M1/M2 experiments (150/150 cells, 3 seeds,
5000 paired test episodes per unit; McNemar + Wilson-CI statistics) show the answer is
**regime-dependent** — reconstruction outperforms task-oriented training at low bandwidth
(k ∈ {1,2,3}), and digital communication wins decisively once bandwidth is sufficient. The neutral
RQ turns this from a "failed confirmation" into the project's central scientific finding: a characterized
regime boundary with a diagnosis, not a promoted method.

**1.3 Novelty.**
The defensible novelty claim (matches the detailed literature audit already in the repo) is:

1. **Domain transfer + closed-loop extension:** Diao et al.'s task-oriented JSCC-for-control is
   image-based autonomous driving; this project transfers it to a low-dimensional UAV kinematic
   state AND backpropagates end-to-end through the closed-loop UAV plant dynamics, which the
   source literature does not specify for UAVs (documented as project-specific in the audit).
2. **First systematic matched-bandwidth three-way comparison** (digital vs reconstruction vs
   task-oriented) for UAV navigation under both AWGN and Rayleigh fading — not present in the
   reviewed literature.
3. **Regime-mapping result:** where task-orientation helps vs hurts as a function of bandwidth/SNR,
   with a root-cause diagnosis (ablation waves A1–A3, below).
4. **Reproducible frozen-protocol testbed** (pre-registered configs, seeds, paired evaluation).

We do NOT claim a new coding paradigm. The conservative, evidence-backed claim is exactly what
supervisors should want for an FYP that will also become a paper.

## 2._requested approvals (extension waves; frozen M1–M4 results are untouched)

| Item | What | Why it matters |
|---|---|---|
| **A1** | Adopt revised RQ1 (above) | Removes the internal inconsistency flagged in review |
| **A2** | Adopt the MVP-aligned title/problem statement: Task-Oriented DeepJSCC for the **UAV navigation state uplink** | Aligns all documents with the implemented system |
| **A3** | One-sentence return-link scope note in the system model | Closes the architecture objection formally |
| **A4** | Approve extension waves **W-A** (loss λ ablations + task-aware weighted-reconstruction training) and **W-B** (CSI estimation-error ablation) as **post-freeze amendments**, M5-expanded | Directly diagnoses WHY task-oriented underperforms (λ_u = λ_v = 0 is a candidate cause); needed for paper-level contribution |

**Guarantee:** all frozen Step-9C values and all M1–M4 artifacts remain byte-identical and remain
the primary comparisons; extension waves run under new experiment IDs (`WAVE_A`, `WAVE_B`) with
the same seeds/evaluation machinery, clearly labeled as amendments.

## 3._questions we need answered by supervisor (blocks: only A4)

1. Do you approve A1–A4 as listed? (A1–A3 are document edits; A4 is new compute.)
2. For the paper track, do you prefer the diagnostic-study framing
   ("When does task-oriented JSCC fail — and what fixes it") or a method-paper framing after
   the ablations land?
3. Timeline check: is thesis-first (then journal submission) the right order in your view?
