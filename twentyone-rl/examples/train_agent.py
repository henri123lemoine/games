from pathlib import Path

from twentyone_rl.agents.mccfr import MCCFR
from twentyone_rl.agents.mccfr.utils import save_policy


def main() -> None:
    """Train an MCCFR agent and save the policy."""
    print("Training MCCFR Agent")
    print("=" * 20)

    # Create agent
    agent = MCCFR(seed=42)

    # Train with progress updates
    iterations = 100000
    update_interval = 20000

    print(f"Training for {iterations:,} iterations...")
    print("")

    for i in range(0, iterations, update_interval):
        batch_size = min(update_interval, iterations - i)
        print(f"Training batch {i//update_interval + 1}/{iterations//update_interval}...")
        agent.train(iterations=batch_size)
        progress = (i + batch_size) / iterations * 100
        print(f"Progress: {progress:.1f}% ({i + batch_size:,}/{iterations:,} iterations completed)")
        print("")

    # Save the trained policy
    policy = agent.average_policy()
    output_path = Path("data/policy_mccfr.json")
    output_path.parent.mkdir(exist_ok=True)
    save_policy(policy, output_path)

    print("Training completed!")
    print(f"Saved policy with {len(policy):,} information sets to {output_path}")

    if output_path.exists():
        print(f"Policy file size: {output_path.stat().st_size / 1024:.1f} KB")

    print("")
    print("You can now play against the agent with:")
    print(f"  uv run examples/play_vs_agent.py {output_path}")


if __name__ == "__main__":
    main()
