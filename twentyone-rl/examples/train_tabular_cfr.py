from pathlib import Path

from loguru import logger

from twentyone_rl.agents.tabular_cfr import TabularCFR, save_policy


def main() -> None:
    """Train a Tabular CFR agent and save the model."""
    logger.info("Training Tabular CFR Agent")
    logger.info("=" * 30)

    # Create agent
    agent = TabularCFR(seed=42)
    logger.info("Initialized Tabular CFR agent")

    # Training parameters - tabular CFR converges much faster than neural approaches
    total_iterations = 20000  # Should be sufficient for convergence
    save_interval = 5000  # Save every 5k iterations
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

            logger.info(f"Information sets discovered: {stats['infosets_discovered']:,}")
            logger.info(f"New information sets: {stats['new_infosets']:,}")

            # Get detailed stats
            detailed_stats = agent.get_stats()
            logger.info(
                f"Average regret per infoset: {detailed_stats['average_regret_per_infoset']:.4f}"
            )
            logger.info(f"Convergence metric: {detailed_stats['convergence_metric']:.6f}")
            logger.info("")

        # Save intermediate model
        model_path = Path(f"data/tabular_cfr_model_{i+batch_size}.json")
        model_path.parent.mkdir(exist_ok=True)
        agent.save_model(model_path)
        logger.info(f"Saved intermediate model to {model_path}")

        # Save policy metadata
        policy = agent.average_policy()
        policy_path = Path(f"data/policy_tabular_cfr_{i+batch_size}.json")
        save_policy(policy, policy_path)
        logger.info(f"Saved policy metadata to {policy_path}")
        logger.info("")

    # Save final model
    final_model_path = Path("data/tabular_cfr_model_final.json")
    agent.save_model(final_model_path)

    final_policy = agent.average_policy()
    final_policy_path = Path("data/policy_tabular_cfr_final.json")
    save_policy(final_policy, final_policy_path)

    logger.info("Training completed!")
    logger.info(f"Final model saved to {final_model_path}")
    logger.info(f"Final policy saved to {final_policy_path}")

    # Report final statistics
    final_stats = agent.get_stats()
    logger.info("")
    logger.info("Final Statistics:")
    logger.info(f"  Total information sets learned: {final_stats['total_information_sets']:,}")
    logger.info(f"  Average regret per infoset: {final_stats['average_regret_per_infoset']:.6f}")
    logger.info(f"  Convergence metric: {final_stats['convergence_metric']:.8f}")

    if final_model_path.exists():
        model_size_kb = final_model_path.stat().st_size / 1024
        logger.info(f"  Model file size: {model_size_kb:.2f} KB")

    logger.info("")
    logger.info("You can now evaluate the agent with:")
    logger.info(f"  uv run examples/evaluate_agents.py")
    logger.info("")
    logger.info("Or play against it with:")
    logger.info(f"  uv run examples/play_vs_agent.py {final_policy_path}")


if __name__ == "__main__":
    main()
