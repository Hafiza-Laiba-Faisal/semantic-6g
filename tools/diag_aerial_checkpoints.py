"""Read-only diagnostic: are the A-12 checkpoints broken networks?

Evaluates the A-12 trained checkpoints on the FROZEN awgn and rayleigh
channels - no aerial gain anywhere.  If they fail there too, the divergent
training destroyed the network, and the aerial curves are not measuring a
channel effect.

Writes nothing; uses the same evaluation path as the runners.

    python tools/diag_aerial_checkpoints.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from project import config  # noqa: E402
from project.evaluation.evaluate import (Condition, aggregate,  # noqa: E402
                                         generate_realizations, run_method)
import tools.run_aerial_training as at  # noqa: E402

EPISODES = 1000
SNRS = (0.0, 10.0, 20.0)


def main():
    rows = []
    for k in at.KS:
        cell = [c for c in at.cells() if c.k == k][0]
        model = cell.load_ckpt()
        print(f"\n== A-12 trained k={k} ({cell.tag}) on the frozen channels ==")
        for channel in ("awgn", "rayleigh"):
            cells = []
            for snr in SNRS:
                cond = Condition(k=k, channel=channel, snr_db=snr,
                                 t_max=config.T_MAX, episodes=EPISODES,
                                 seed=config.TEST_SEED)
                real = generate_realizations(cond, EPISODES)
                m = run_method(cond, real, estimate_fn=lambda s, n, h, _s=snr,
                               _c=channel: model.forward(
                                   s, _s, channel=_c, noise=n, h=h))
                cells.append(f"{snr:+.0f}dB {aggregate(m)['success_rate']:.3f}")
            print(f"   {channel:8s} " + "  ".join(cells))
        rows.append(k)
    print("\nIf these columns are ~0.02 and flat, the checkpoint is dead on"
          " AWGN too: the training collapsed, not the evaluation.")
    return rows


if __name__ == "__main__":
    main()
