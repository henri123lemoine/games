import sys
from pathlib import Path

# Add the twentyone package to the path
sys.path.insert(0, str(Path(__file__).parent / "../../twentyone-py/python"))

# Add the RL package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from twentyone_rl.agents.deep_mccfr import DeepMCCFR, save_policy


def main() -> None:
    """Train a Deep MCCFR agent and save the model."""
    logger.info("Training Deep MCCFR Agent")
    logger.info("=" * 30)

    # Create agent
    device = "cpu"  # Change to "cuda" if GPU available
    agent = DeepMCCFR(seed=42, learning_rate=1e-4, device=device)
    logger.info(f"Using device: {device}")

    # Training parameters
    total_iterations = 50000  # Substantial training for real performance
    save_interval = 10000  # Save every 10k iterations
    log_interval = 1000  # Log every 1k iterations

    logger.info(f"Training for {total_iterations:,} iterations...")
    logger.info(f"Saving model every {save_interval:,} iterations")
    logger.info("")

    # Training loop with progress updates
    for i in range(0, total_iterations, save_interval):
        batch_size = min(save_interval, total_iterations - i)

        logger.info(f"Training batch {i//save_interval + 1}/{total_iterations//save_interval}...")
        logger.info(f"Iterations {i+1:,} to {i+batch_size:,}")

        # Train in smaller chunks for logging
        for j in range(0, batch_size, log_interval):
            chunk_size = min(log_interval, batch_size - j)
            current_iter = i + j

            stats = agent.train(iterations=chunk_size)

            progress = (current_iter + chunk_size) / total_iterations * 100
            logger.info(
                f"Progress: {progress:.1f}% ({current_iter + chunk_size:,}/{total_iterations:,})"
            )

            if stats.get("losses"):
                latest_losses = stats["losses"][-1] if stats["losses"] else {}
                if latest_losses:
                    logger.info(f"Latest losses: {latest_losses}")

            logger.info(f"Buffer size: {stats.get('buffer_size', 0):,}")
            logger.info("")

        # Save intermediate model
        model_path = Path(f"data/deep_mccfr_model_{i+batch_size}.pth")
        model_path.parent.mkdir(exist_ok=True)
        agent.save_model(model_path)
        logger.info(f"Saved intermediate model to {model_path}")

        # Save policy metadata
        policy = agent.average_policy()
        policy_path = Path(f"data/policy_deep_mccfr_{i+batch_size}.json")
        save_policy(policy, policy_path)
        logger.info(f"Saved policy metadata to {policy_path}")
        logger.info("")

    # Save final model
    final_model_path = Path("data/deep_mccfr_model_final.pth")
    agent.save_model(final_model_path)

    final_policy = agent.average_policy()
    final_policy_path = Path("data/policy_deep_mccfr_final.json")
    save_policy(final_policy, final_policy_path)

    logger.info("Training completed!")
    logger.info(f"Final model saved to {final_model_path}")
    logger.info(f"Final policy saved to {final_policy_path}")

    if final_model_path.exists():
        model_size_mb = final_model_path.stat().st_size / (1024 * 1024)
        logger.info(f"Model file size: {model_size_mb:.2f} MB")

    logger.info("")
    logger.info("You can now evaluate the agent with:")
    logger.info(f"  uv run examples/evaluate_agents.py")
    logger.info("")
    logger.info("Or play against it with:")
    logger.info(f"  uv run examples/play_vs_agent.py {final_policy_path}")


if __name__ == "__main__":
    main()
