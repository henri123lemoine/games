import sys
from pathlib import Path

# Add the twentyone package to the path
sys.path.insert(0, str(Path(__file__).parent / "../../twentyone-py/python"))

# Add the RL package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from twentyone_rl.agents.mccfr import MCCFR, save_policy


def main() -> None:
    """Train a regular MCCFR agent and save the policy."""
    logger.info("Training MCCFR Agent")
    logger.info("=" * 25)

    # Create agent
    agent = MCCFR(seed=42)
    logger.info("MCCFR agent created")

    # Training parameters
    total_iterations = 10000  # Good baseline training
    log_interval = 2000

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

    logger.info("Training completed!")
    logger.info(f"Policy saved to {policy_path}")
    logger.info("")
    logger.info("You can now evaluate against this baseline with:")
    logger.info("  uv run examples/evaluate_agents.py")
    logger.info("")
    logger.info("Or play against it with:")
    logger.info(f"  uv run examples/play_vs_agent.py {policy_path}")


if __name__ == "__main__":
    main()
