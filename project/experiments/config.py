"""Explicit experiment configuration (audit section 7, item 9).

Every experiment run is fully described by an :class:`ExperimentConfig`;
nothing is hard-coded in model/training code. All frozen values come from
``project.config``:

* training seeds {42, 43, 44}, test seed 10042
* training SNR policy U(0, 20) dB (explicit, inspectable)
* test SNR grids: M1 {0, 2, ..., 20} dB, M2 {0, 5, 10, 15, 20} dB
* bandwidth: neural k in {1, 2, 3}; matched-budget M2 k in
  {12, 18, 24, 30, 54} with B = (k - 6)/6
* digital infeasible at k in {1, 2, 3} (6B + 6 <= k) - EXPECTED for M3
  cells, an error everywhere else

Run modes: ``dry_run`` (configuration summary only, NO training), ``smoke``
(small deterministic validation, non-scientific), ``full`` (explicitly
BLOCKED until the user approves full-scale M1-M5 execution).

Note: ``digital_feasible`` lives HERE (not in matrix.py) because matrix.py
imports this module; the budget arithmetic itself is the frozen
``config.digital_packet`` - no duplicated mathematics.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from project import config
from project.baselines.digital import DigitalBaseline
from project.evaluation.evaluate import Condition as EvalCondition
from project.evaluation.evaluate import evaluate_condition
from project.models.reconstruction_jscc import ReconstructionDeepJSCC
from project.training.common import build_optimizer, save_checkpoint
from project.training.train_reconstruction import train_reconstruction
from project.training.train_task_oriented import train_task_oriented


# ---------------------------------------------------------------------------
# frozen matrix definitions (structural; execution is a later step)
# ---------------------------------------------------------------------------

M1_K_VALUES = (1, 2, 3)
M1_SNR_DB = tuple(float(s) for s in range(0, 21, 2))
M2_K_VALUES = (12, 18, 24, 30, 54)
M2_SNR_DB = (0.0, 5.0, 10.0, 15.0, 20.0)
M2_B_BY_K = dict(config.DIGITAL_B_BY_K)   # {12:1, 18:2, 24:3, 30:4, 54:8}
M4_DIGITAL_B = 8                          # secondary reference: B = 8
M4_K = 54                                 # n_dig = 54 channel uses (rho_dig = 9)
CHANNELS = ("awgn", "rayleigh")
TRAINING_SEEDS = tuple(config.TRAINING_SEEDS)      # (42, 43, 44)


def b_for_m2_k(k: int) -> int:
    """B = (k - 6) / 6 for the matched-budget points (audit 8.6)."""
    return M2_B_BY_K[k]


def digital_feasible(k: int, bits_per_component: Optional[int]) -> Tuple[bool, str]:
    """Frozen budget check: payload 6B + 6 must fit inside k channel uses.

    Single implementation for the config layer (matrix.py imports this);
    delegates the arithmetic to the frozen ``config.digital_packet``.
    """
    if bits_per_component is None:
        return False, "digital requires bits_per_component (B)"
    try:
        pkt = config.digital_packet(k, bits_per_component)
    except config.InfeasibleDigitalConfig as e:
        return False, str(e)
    return True, (f"6B={pkt['n_src_bits']}, info={pkt['n_info_bits']}, "
                  f"coded={pkt['n_coded_bits']}, payload="
                  f"{pkt['n_payload_symbols']}, pad={pkt['n_padding_symbols']}, "
                  f"k={k}")


# ---------------------------------------------------------------------------
# experiment configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """One experiment cell, fully self-describing."""

    experiment: str                 # "M1" | "M2" | "M3" | "M4" | "M5"
    method: str                     # "reconstruction" | "task" | "digital" | "oracle"
    channel: str                    # "awgn" | "rayleigh"
    k: int
    training_seed: int
    test_seed: int = config.TEST_SEED
    # training configuration (smoke scale by default; full scale is a later,
    # explicitly approved step - run_mode='full' is rejected by validate())
    train_steps: int = 400
    batch_episodes: int = config.BATCH_EPISODES
    batch_states: int = config.BATCH_STATES
    lr: Optional[float] = None      # None -> frozen config.LEARNING_RATE
    # training SNR policy: explicit and inspectable, never hard-coded in
    # the trainers' callers; the trainers implement it (U(range) per batch)
    train_snr_policy: Dict = field(default_factory=lambda: {
        "distribution": "uniform",
        "range_db": list(config.SNR_TRAIN_RANGE_DB),
        "granularity": "per_batch"})
    # evaluation configuration
    eval_snr_grid_db: Tuple[float, ...] = M1_SNR_DB
    t_max: int = config.T_MAX
    eval_snr_db: float = 10.0       # fixed SNR for the task trainer's eval set
    n_train_episodes: int = config.N_TRAIN_EPISODES
    n_test_episodes: int = config.N_TEST_EPISODES_SMOKE
    eval_episodes: int = 256        # task trainer's fixed evaluation episodes
    # method configuration
    bits_per_component: Optional[int] = None   # required for the digital method
    checkpoint_path: Optional[str] = None
    output_path: Optional[str] = None
    run_mode: str = "smoke"         # "dry_run" | "smoke" | "full"

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    @property
    def expected_infeasible(self) -> bool:
        """M3 cells are digital-infeasible BY DESIGN (audit 8.5)."""
        return self.experiment == "M3" and self.method == "digital"

    def validate(self) -> List[str]:
        """Return a list of configuration errors (empty = valid)."""
        errors: List[str] = []
        if self.experiment not in ("M1", "M2", "M3", "M4", "M5"):
            errors.append(f"unknown experiment {self.experiment!r}")
        if self.method not in ("reconstruction", "task", "digital", "oracle"):
            errors.append(f"unknown method {self.method!r}")
        if self.channel not in CHANNELS:
            errors.append(f"unknown channel {self.channel!r}")
        if self.run_mode not in ("dry_run", "smoke", "full"):
            errors.append(f"unknown run_mode {self.run_mode!r}")
        if self.training_seed not in TRAINING_SEEDS:
            errors.append(
                f"training seed {self.training_seed} not in {TRAINING_SEEDS}")
        if self.test_seed != config.TEST_SEED:
            errors.append(f"test seed must be {config.TEST_SEED}")
        if self.k <= 0:
            errors.append("k must be a positive integer")
        if self.method == "digital":
            if self.bits_per_component is None:
                errors.append("digital method requires bits_per_component")
            else:
                feasible, reason = digital_feasible(self.k, self.bits_per_component)
                if self.expected_infeasible:
                    if feasible:
                        errors.append(
                            "M3 cell unexpectedly feasible - digital must remain "
                            f"infeasible at k in {{1, 2, 3}} ({reason})")
                elif not feasible:
                    errors.append(f"digital infeasible: {reason}")
        if self.run_mode == "full":
            errors.append(
                "run_mode='full' is BLOCKED in this step: full-scale M1-M5 "
                "execution requires explicit user approval")
        return errors

    def validate_or_raise(self) -> "ExperimentConfig":
        errs = self.validate()
        if errs:
            raise ValueError("invalid experiment config: " + "; ".join(errs))
        return self

    # ------------------------------------------------------------------
    # presets (frozen matrix rows, validated at construction)
    # ------------------------------------------------------------------

    @classmethod
    def m1(cls, method: str, k: int, channel: str, seed: int,
           run_mode: str = "smoke") -> "ExperimentConfig":
        return cls(experiment="M1", method=method, channel=channel, k=k,
                   training_seed=seed, eval_snr_grid_db=M1_SNR_DB,
                   run_mode=run_mode).validate_or_raise()

    @classmethod
    def m2(cls, method: str, k: int, channel: str, seed: int,
           run_mode: str = "smoke") -> "ExperimentConfig":
        b = b_for_m2_k(k) if method == "digital" else None
        return cls(experiment="M2", method=method, channel=channel, k=k,
                   training_seed=seed, bits_per_component=b,
                   eval_snr_grid_db=M2_SNR_DB,
                   run_mode=run_mode).validate_or_raise()

    @classmethod
    def m3(cls, k: int, bits: int = 1, channel: str = "awgn", seed: int = 42,
           run_mode: str = "dry_run") -> "ExperimentConfig":
        """Digital infeasibility cell (k in {1, 2, 3}); expected infeasible."""
        return cls(experiment="M3", method="digital", channel=channel, k=k,
                   training_seed=seed, bits_per_component=bits,
                   run_mode=run_mode).validate_or_raise()

    @classmethod
    def m4_secondary(cls, channel: str, seed: int,
                     run_mode: str = "smoke") -> "ExperimentConfig":
        """SECONDARY digital reference: B = 8 -> n_dig = 54 (rho_dig = 9).

        Never a primary cell; the matrix validator and this preset both
        enforce the labeling.
        """
        return cls(experiment="M4", method="digital", channel=channel, k=M4_K,
                   training_seed=seed, bits_per_component=M4_DIGITAL_B,
                   eval_snr_grid_db=M2_SNR_DB,
                   run_mode=run_mode).validate_or_raise()

    # ------------------------------------------------------------------
    # dry run (NO training, NO side effects)
    # ------------------------------------------------------------------

    def digital_packet_summary(self) -> Optional[Dict]:
        """Packet arithmetic for digital cells (None for neural/oracle)."""
        if self.method != "digital" or self.bits_per_component is None:
            return None
        feasible, note = digital_feasible(self.k, self.bits_per_component)
        if not feasible:
            return {"feasible": False, "note": note}
        return dict(config.digital_packet(self.k, self.bits_per_component),
                    feasible=True)

    def dry_run_summary(self) -> Dict:
        """Structured summary for the configuration-only dry run."""
        feasibility = None
        if self.method == "digital":
            feasible, reason = digital_feasible(self.k, self.bits_per_component)
            feasibility = {"feasible": feasible, "expected_infeasible":
                           self.expected_infeasible, "note": reason}
        return {
            "experiment": self.experiment,
            "method": self.method,
            "channel": self.channel,
            "k": self.k,
            # expected channel-use budget
            "channel_uses": self.k,
            "rho": self.k / 6.0,
            "digital_packet": self.digital_packet_summary(),
            "bits_per_component": self.bits_per_component,
            "training_seed": self.training_seed,
            "test_seed": self.test_seed,
            "training_config": {
                "train_steps": self.train_steps,
                "batch_episodes": self.batch_episodes,
                "batch_states": self.batch_states,
                "lr": self.lr if self.lr is not None else config.LEARNING_RATE,
                "train_snr_policy": dict(self.train_snr_policy),
                "n_train_episodes": self.n_train_episodes},
            "evaluation_config": {
                "eval_snr_grid_db": list(self.eval_snr_grid_db),
                "t_max": self.t_max,
                "eval_snr_db": self.eval_snr_db,
                "n_test_episodes": self.n_test_episodes,
                "eval_episodes": self.eval_episodes},
            "model_config": {
                "hidden_dims": list(config.HIDDEN_DIMS),
                "activation": "PReLU",
                "state_dim": config.STATE_DIM},
            "digital_feasibility": feasibility,
            "checkpoint_path": self.checkpoint_path,
            "output_path": self.output_path,
            "run_mode": self.run_mode,
            "training_will_occur": self.run_mode == "smoke",
            "errors": self.validate(),
        }

    # ------------------------------------------------------------------
    # execution (smoke only in this step)
    # ------------------------------------------------------------------

    def to_eval_condition(self, snr_db: float) -> EvalCondition:
        return EvalCondition(k=self.k, channel=self.channel, snr_db=snr_db,
                             t_max=self.t_max,
                             bits_per_component=self.bits_per_component,
                             episodes=self.n_test_episodes,
                             seed=self.test_seed)

    def run(self, dry_run: Optional[bool] = None) -> Dict:
        """Dispatch by run_mode. ``full`` is rejected by validate()."""
        summary = self.dry_run_summary()
        mode = self.run_mode if dry_run is None else (
            "dry_run" if dry_run else "smoke")
        if mode == "dry_run":
            dry = replace(self, run_mode="dry_run")
            return {"config": dry.dry_run_summary(), "status": "dry_run"}
        self.validate_or_raise()          # rejects run_mode='full'

        ckpt_path = self.checkpoint_path or (
            f"results/checkpoints/{self.experiment}_{self.method}_"
            f"{self.channel}_k{self.k}_seed{self.training_seed}.pth")

        model: Optional[ReconstructionDeepJSCC] = None
        train_block: Optional[Dict] = None
        if self.method == "reconstruction":
            model, rep = train_reconstruction(
                k=self.k, channel=self.channel, seed=self.training_seed,
                steps=self.train_steps, batch_states=self.batch_states,
                lr=self.lr)
            train_block = {"loss_initial": rep["loss_initial"],
                           "loss_final": rep["loss_final"],
                           "loss_reduction_pct": rep["loss_reduction_pct"],
                           "nan_batches": rep["nan_batches"],
                           "history": rep["history"]}
        elif self.method == "task":
            model, rep = train_task_oriented(
                k=self.k, channel=self.channel, seed=self.training_seed,
                steps=self.train_steps, batch_episodes=self.batch_episodes,
                t_max=self.t_max, lr=self.lr,
                snr_range=tuple(self.train_snr_policy["range_db"]),
                eval_episodes=self.eval_episodes,
                eval_snr_db=self.eval_snr_db)
            train_block = {"train_total_initial": rep["train_total_initial"],
                           "train_total_final": rep["train_total_final"],
                           "lambdas": rep["lambdas"],
                           "eval_before": rep["eval_before"],
                           "eval_after": rep["eval_after"],
                           "nan_batches": rep["nan_batches"],
                           "history": rep["history"]}

        # evaluation over the configured grid via the Step-7 paired harness
        digital_model: Optional[DigitalBaseline] = None
        if self.method == "digital":
            feasible, _ = digital_feasible(self.k, self.bits_per_component)
            if feasible:
                digital_model = DigitalBaseline(
                    k=self.k, bits_per_component=self.bits_per_component)
        eval_rows: Dict[str, Dict] = {}
        for snr_db in self.eval_snr_grid_db:
            cond = self.to_eval_condition(snr_db)
            cell = evaluate_condition(
                cond, methods=[self.method],
                recon_model=model if self.method == "reconstruction" else None,
                task_model=model if self.method == "task" else None,
                digital=digital_model)
            row = cell[self.method]
            eval_rows[f"{snr_db:g}"] = row.get("aggregate", row.get("status"))

        # checkpoint: neural methods only (digital/oracle have no state)
        ckpt: Optional[str] = None
        if model is not None:
            ckpt = str(save_checkpoint(
                ckpt_path, model, optimizer=build_optimizer(model, self.lr),
                step=self.train_steps, seed=self.training_seed,
                metadata={
                    "experiment": self.experiment,
                    "method": self.method,
                    "channel": self.channel,
                    "k": self.k,
                    "bits_per_component": self.bits_per_component,
                    "training_seed": self.training_seed,
                    "test_seed": self.test_seed,
                    "train_steps": self.train_steps,
                    "batch_episodes": self.batch_episodes,
                    "batch_states": self.batch_states,
                    "lr": self.lr if self.lr is not None else config.LEARNING_RATE,
                    "train_snr_policy": dict(self.train_snr_policy),
                    "eval_snr_grid_db": list(self.eval_snr_grid_db),
                    "t_max": self.t_max,
                    "n_test_episodes": self.n_test_episodes,
                    "model_config": {"hidden_dims": list(config.HIDDEN_DIMS),
                                     "activation": "PReLU"},
                    "run_mode": self.run_mode},
                history=train_block["history"] if train_block else None))

        return {"config": summary, "status": "smoke_complete",
                "train": train_block, "eval": eval_rows,
                "checkpoint": ckpt,
                "non_scientific": True,
                "note": ("Step-8 smoke validation of infrastructure only; "
                         "not an experimental result")}


# convenience re-export (runner.py / tests use dataclasses.replace on configs)
__all__ = ["ExperimentConfig", "M1_K_VALUES", "M1_SNR_DB", "M2_K_VALUES",
           "M2_SNR_DB", "M2_B_BY_K", "M4_DIGITAL_B", "M4_K", "CHANNELS",
           "TRAINING_SEEDS", "b_for_m2_k", "digital_feasible", "replace"]
