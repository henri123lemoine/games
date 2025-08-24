from pathlib import Path

from loguru import logger

from twentyone_rl.agents.mccfr import MCCFR
from twentyone_rl.agents.mccfr.utils import save_policy


def main() -> None:
    """Train a regular MCCFR agent and save the policy."""
    logger.info("Training MCCFR Agent")
    logger.info("=" * 25)

    # Create agent
    agent = MCCFR(seed=42)
    logger.info("MCCFR agent created")

    # Training parameters
    total_iterations = 200000
    log_interval = 20000

    logger.info(f"Training for {total_iterations:,} iterations...")
    logger.info("")

    # Training loop with progress updates
    for i in range(0, total_iterations, log_interval):
        batch_size = min(log_interval, total_iterations - i)
        current_iter = i + batch_size

        logger.info(f"Training iterations {i+1:,} to {current_iter:,}...")
        agent.train(iterations=batch_size)

        progress = current_iter / total_iterations * 100
        logger.info(f"Progress: {progress:.1f}% ({current_iter:,}/{total_iterations:,})")
        logger.info("")

    # Save final policy
    policy = agent.average_policy()
    policy_path = Path("data/policy_mccfr.json")
    policy_path.parent.mkdir(exist_ok=True)
    save_policy(policy, policy_path)


if __name__ == "__main__":
    main()
