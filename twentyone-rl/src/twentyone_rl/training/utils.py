"""Shared utilities for training scripts."""

from pathlib import Path
from typing import Any

from loguru import logger


def log_training_start(agent_name: str, total_iterations: int, **kwargs: Any) -> None:
    """Log the start of training with consistent formatting."""
    logger.info(f"Training {agent_name}")
    logger.info("=" * (len(agent_name) + 9))
    logger.info(f"Training for {total_iterations:,} iterations...")

    for key, value in kwargs.items():
        if key == "device":
            logger.info(f"Using device: {value}")
        elif key == "save_interval":
            logger.info(f"Saving model every {value:,} iterations")
        elif key == "log_interval":
            logger.info(f"Logging every {value:,} iterations")

    logger.info("")


def log_training_progress(
    current_iter: int, total_iterations: int, batch_num: int = 0, total_batches: int = 0
) -> None:
    """Log training progress with consistent formatting."""
    progress = current_iter / total_iterations * 100
    logger.info(f"Progress: {progress:.1f}% ({current_iter:,}/{total_iterations:,})")

    if batch_num and total_batches:
        logger.info(f"Training batch {batch_num}/{total_batches}...")
        logger.info(f"Iterations {current_iter - chunk_size + 1:,} to {current_iter:,}")


def save_intermediate_model(
    agent: Any, iteration: int, data_dir: Path = Path("data")
) -> tuple[Path, Path]:
    """Save intermediate model and policy with consistent naming."""
    # Determine agent type for file naming
    agent_type = getattr(agent, "agent_type", type(agent).__name__.lower())
    if hasattr(agent, "average_policy"):
        policy_info = agent.average_policy()
        agent_type = policy_info.get("agent_type", agent_type)

    # Create file paths
    model_path = (
        data_dir / f"{agent_type}_model_{iteration}.pth"
        if hasattr(agent, "save_model")
        else data_dir / f"{agent_type}_model_{iteration}.json"
    )
    policy_path = data_dir / f"policy_{agent_type}_{iteration}.json"

    # Ensure directory exists
    data_dir.mkdir(exist_ok=True)

    # Save model
    agent.save_model(model_path)
    logger.info(f"Saved intermediate model to {model_path}")

    # Save policy metadata
    from ..agents.mccfr.utils import save_policy

    policy = agent.average_policy()
    save_policy(policy, policy_path)
    logger.info(f"Saved policy metadata to {policy_path}")

    return model_path, policy_path


def save_final_model(agent: Any, data_dir: Path = Path("data")) -> tuple[Path, Path]:
    """Save final model and policy with consistent naming."""
    # Determine agent type for file naming
    agent_type = getattr(agent, "agent_type", type(agent).__name__.lower())
    if hasattr(agent, "average_policy"):
        policy_info = agent.average_policy()
        agent_type = policy_info.get("agent_type", agent_type)

    # Create file paths
    model_path = (
        data_dir / f"{agent_type}_model_final.pth"
        if hasattr(agent, "save_model")
        else data_dir / f"{agent_type}_model_final.json"
    )
    policy_path = data_dir / f"policy_{agent_type}_final.json"

    # Ensure directory exists
    data_dir.mkdir(exist_ok=True)

    # Save model
    agent.save_model(model_path)

    # Save policy metadata
    from ..agents.mccfr.utils import save_policy

    policy = agent.average_policy()
    save_policy(policy, policy_path)

    logger.info("Training completed!")
    logger.info(f"Final model saved to {model_path}")
    logger.info(f"Final policy saved to {policy_path}")

    return model_path, policy_path


def log_final_stats(agent: Any, model_path: Path) -> None:
    """Log final training statistics."""
    if hasattr(agent, "get_stats"):
        # For tabular agents with detailed stats
        final_stats = agent.get_stats()
        logger.info("")
        logger.info("Final Statistics:")
        for key, value in final_stats.items():
            if isinstance(value, float):
                logger.info(f"  {key.replace('_', ' ').title()}: {value:.6f}")
            else:
                logger.info(f"  {key.replace('_', ' ').title()}: {value:,}")

    # Log file size
    if model_path.exists():
        size_bytes = model_path.stat().st_size
        if size_bytes > 1024 * 1024:  # > 1MB
            size_str = f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            size_str = f"{size_bytes / 1024:.2f} KB"
        logger.info(f"  Model file size: {size_str}")

    logger.info("")
    logger.info("You can now evaluate the agent with:")
    logger.info("  uv run examples/evaluate_agents.py")
    logger.info("")


def log_training_stats(stats: dict[str, Any]) -> None:
    """Log training statistics with consistent formatting."""
    if stats.get("losses"):
        latest_losses = stats["losses"][-1] if stats["losses"] else {}
        if latest_losses:
            logger.info(f"Latest losses: {latest_losses}")

    if "infosets_discovered" in stats:
        logger.info(f"Information sets discovered: {stats['infosets_discovered']:,}")

    if "new_infosets" in stats:
        logger.info(f"New information sets: {stats['new_infosets']:,}")

    if "buffer_size" in stats:
        logger.info(f"Buffer size: {stats['buffer_size']:,}")

    # Log convergence metrics for tabular agents
    if hasattr(stats, "get") and stats.get("convergence_metric"):
        logger.info(f"Convergence metric: {stats['convergence_metric']:.6f}")

    logger.info("")
