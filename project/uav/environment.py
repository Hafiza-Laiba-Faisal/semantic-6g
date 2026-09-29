"""Deterministic 2-D navigation environment (audit 1.1 rows 2-4, A4/A5).

Minimal, deterministic, vectorized, obstacle-free. Success and failure
semantics follow the frozen decisions:

* success  : d_t <= goal_radius at ANY step t in {0, 1, ..., T} (A4)
* failure  : horizon elapsed without success, or workspace exit during
             evaluation (A5). On exit the episode terminates for that UAV:
             its state is frozen at the first out-of-workspace position and
             all later records are held constant (deterministic, distinct
             from a silent in-place clamp).

During training the environment is bypassed: rollouts run directly through
``project.uav.dynamics`` with fixed horizons (audit A5).

The environment holds no learned components and no notion of channels;
communication systems plug in upstream by producing the state estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from project import config
from project.uav.controller import pd_controller
from project.uav.dynamics import step as dynamics_step


@dataclass
class EpisodeResult:
    """Full trajectory record of one vectorized episode batch."""

    positions: List[torch.Tensor]        # [B, 2] per step, t = 0..T (record=True)
    velocities: List[torch.Tensor]       # [B, 2] per step
    controls: List[torch.Tensor]         # [B, 2] per step, u applied at t -> t+1
    distances: List[torch.Tensor]        # [B]    per step, t = 0..T
    goal: torch.Tensor                   # [B, 2]
    success: torch.Tensor                # [B] bool: d_t <= r_g for some t
    time_to_goal: torch.Tensor           # [B] int: first t of success/exit, T+1 if none
    final_distance: torch.Tensor         # [B]
    min_distance: torch.Tensor           # [B]
    exited_workspace: torch.Tensor       # [B] bool


def episode_distances(p: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
    """Euclidean distance to the goal, shape ``[batch]``."""
    return torch.linalg.vector_norm(p - goal, dim=-1)


class NavigationEnv:
    """Vectorized double-integrator navigation environment."""

    def __init__(
        self,
        dt: float = config.DT,
        t_max: int = config.T_MAX,
        goal_radius: float = config.GOAL_RADIUS,
        v_max: float = config.V_MAX,
        u_max: float = config.U_MAX,
        workspace_min: float = config.WORKSPACE_MIN,
        workspace_max: float = config.WORKSPACE_MAX,
    ) -> None:
        self.dt = dt
        self.t_max = t_max
        self.goal_radius = goal_radius
        self.v_max = v_max
        self.u_max = u_max
        self.workspace_min = workspace_min
        self.workspace_max = workspace_max

    # ------------------------------------------------------------------
    # episode creation
    # ------------------------------------------------------------------

    def sample_starts(
        self,
        n: int,
        generator: Optional[torch.Generator] = None,
        enforce_d_min: bool = True,
        enforce_d_max: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample initial positions, velocities and goals.

        Positions are uniform over the workspace with the configured
        minimum start-goal distance (and ``d_max`` when frozen, A6).
        Initial velocity is the frozen ``v_0 = (0, 0)``.

        Returns ``(p0, v0, goal)``, each shape ``[n, 2]``.
        """
        lo, hi = self.workspace_min, self.workspace_max
        p0 = torch.empty(n, 2).uniform_(lo, hi, generator=generator)
        goal = torch.empty(n, 2).uniform_(lo, hi, generator=generator)

        if enforce_d_min:
            d = episode_distances(p0, goal)
            while bool((d < config.D_MIN).any()):
                m = int((d < config.D_MIN).sum())
                p0[d < config.D_MIN] = torch.empty(m, 2).uniform_(
                    lo, hi, generator=generator)
                goal[d < config.D_MIN] = torch.empty(m, 2).uniform_(
                    lo, hi, generator=generator)
                d = episode_distances(p0, goal)

        if enforce_d_max and config.D_MAX is not None:
            d = episode_distances(p0, goal)
            while bool((d > config.D_MAX).any()):
                m = int((d > config.D_MAX).sum())
                p0[d > config.D_MAX] = torch.empty(m, 2).uniform_(
                    lo, hi, generator=generator)
                goal[d > config.D_MAX] = torch.empty(m, 2).uniform_(
                    lo, hi, generator=generator)
                d = episode_distances(p0, goal)

        v0 = torch.zeros(n, 2) + torch.tensor(config.V_0)
        return p0, v0, goal

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------

    def run_noiseless(
        self,
        p0: torch.Tensor,
        v0: torch.Tensor,
        goal: torch.Tensor,
        record: bool = False,
    ) -> EpisodeResult:
        """Closed-loop noiseless episode; the controller gets ground truth.

        This is the sanity-gate oracle path. Communication methods are
        exercised by scripts that feed their own decoded estimates into the
        same ``pd_controller``.
        """
        n = p0.shape[0]
        device = p0.device
        p = p0.clone()
        v = v0.clone()
        g = goal.clone()

        active = torch.ones(n, dtype=torch.bool, device=device)
        exited = torch.zeros(n, dtype=torch.bool, device=device)

        distances: List[torch.Tensor] = [episode_distances(p, g)]
        positions: List[torch.Tensor] = [p.clone()] if record else []
        velocities: List[torch.Tensor] = [v.clone()] if record else []
        controls: List[torch.Tensor] = []

        d0 = distances[0]
        success = d0 <= self.goal_radius
        time_to_goal = torch.where(
            success,
            torch.zeros(n, dtype=torch.long, device=device),
            torch.full((n,), self.t_max + 1, dtype=torch.long, device=device),
        )

        for t in range(self.t_max):
            # ground-truth full-information state for the oracle path
            s = torch.cat([p, v, g], dim=-1)
            u = pd_controller(s, u_max=self.u_max)
            if record:
                controls.append(u.clone())
            p_new, v_new = dynamics_step(p, v, u, dt=self.dt, v_max=self.v_max)
            d_new = episode_distances(p_new, g)

            # first success time (success may occur while still active);
            # gated on ~success so lingering inside the radius never
            # overwrites the FIRST entry time (A4 semantics)
            newly_success = active & ~success & (d_new <= self.goal_radius)
            time_to_goal = torch.where(
                newly_success,
                torch.full_like(time_to_goal, t + 1),
                time_to_goal,
            )
            success = success | newly_success

            # boundary exit freezes the episode for that UAV (A5)
            outside = (
                (p_new[:, 0] < self.workspace_min)
                | (p_new[:, 0] > self.workspace_max)
                | (p_new[:, 1] < self.workspace_min)
                | (p_new[:, 1] > self.workspace_max)
            )
            newly_exited = active & outside
            time_to_goal = torch.where(
                newly_exited & ~success,
                torch.full_like(time_to_goal, t + 1),
                time_to_goal,
            )
            exited = exited | newly_exited

            # update only still-active UAVs; frozen UAVs hold state & distance
            mask = active.unsqueeze(-1)
            p = torch.where(mask, p_new, p)
            v = torch.where(mask, v_new, v)
            distances.append(
                torch.where(active, d_new, distances[-1]))

            if record:
                positions.append(p.clone())
                velocities.append(v.clone())

            active = active & ~outside
            if not bool(active.any()):
                break

        final_d = distances[-1]
        min_d = torch.stack(distances).min(dim=0).values
        return EpisodeResult(
            positions=positions,
            velocities=velocities,
            controls=controls,
            distances=distances,
            goal=g,
            success=success,
            time_to_goal=time_to_goal,
            final_distance=final_d,
            min_distance=min_d,
            exited_workspace=exited,
        )
