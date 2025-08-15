import sys
from pathlib import Path

# Add the twentyone package to the path
sys.path.insert(0, str(Path(__file__).parent / "../../twentyone-py/python"))

# Add the RL package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from twentyone_rl.evaluation.tournament import (
    DeepMCCFRAgent,
    HeuristicAgent,
    PolicyAgent,
    Tournament,
)


def create_baseline_agents() -> list:
    """Create baseline agents for comparison."""
    agents = []

    # Heuristic agents with different thresholds
    for threshold in [15, 16, 17, 18, 19]:
        agents.append(HeuristicAgent(threshold, f"Heuristic_{threshold}"))

    # Load traditional MCCFR if available
    mccfr_policy_path = Path("data/policy_mccfr.json")
    if mccfr_policy_path.exists():
        agents.append(PolicyAgent(mccfr_policy_path, "MCCFR"))
        logger.info("Loaded traditional MCCFR agent")
    else:
        logger.warning(f"MCCFR policy not found at {mccfr_policy_path}")

    return agents


def evaluate_deep_mccfr() -> None:
    """Evaluate Deep MCCFR against baselines."""
    logger.info("Deep MCCFR Agent Evaluation")
    logger.info("=" * 40)

    # Create tournament
    tournament = Tournament(seed=42)

    # Create baseline agents
    baseline_agents = create_baseline_agents()
    logger.info(f"Created {len(baseline_agents)} baseline agents")

    # Load Deep MCCFR agent
    deep_model_path = Path("data/deep_mccfr_model_final.pth")
    if not deep_model_path.exists():
        logger.warning(f"Final model not found, looking for latest checkpoint...")
        # Look for the latest checkpoint
        checkpoints = list(Path("data").glob("deep_mccfr_model_*.pth"))
        if checkpoints:
            deep_model_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            logger.info(f"Using latest checkpoint: {deep_model_path}")
        else:
            logger.error("No Deep MCCFR model found! Train first with:")
            logger.error("  uv run examples/train_deep_mccfr.py")
            return

    deep_agent = DeepMCCFRAgent(deep_model_path, "DeepMCCFR")
    logger.info(f"Loaded Deep MCCFR from {deep_model_path}")

    # Run head-to-head matches against each baseline
    num_games = 100  # Reduced for testing
    results = {}

    logger.info(f"Running {num_games} games per matchup...")
    logger.info("")

    for baseline in baseline_agents:
        logger.info(f"Evaluating DeepMCCFR vs {baseline.name()}")
        match_result = tournament.run_match(deep_agent, baseline, num_games)
        results[baseline.name()] = match_result

        # Print quick summary
        deep_winrate = match_result["agent1_winrate"]
        baseline_winrate = match_result["agent2_winrate"]
        logger.info(f"DeepMCCFR: {deep_winrate:.3f}, {baseline.name()}: {baseline_winrate:.3f}")

        if deep_winrate > baseline_winrate:
            logger.info("✓ DeepMCCFR wins!")
        elif baseline_winrate > deep_winrate:
            logger.info("✗ DeepMCCFR loses")
        else:
            logger.info("= Tie")
        logger.info("")

    # Summary statistics
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 20)

    wins = 0
    total_matches = len(results)
    overall_winrate = 0

    for opponent, result in results.items():
        deep_winrate = result["agent1_winrate"]
        overall_winrate += deep_winrate

        if deep_winrate > 0.5:
            wins += 1

        ci_low, ci_high = result["agent1_ci"]
        logger.info(f"vs {opponent}: {deep_winrate:.3f} ({ci_low:.3f}-{ci_high:.3f})")

    overall_winrate /= total_matches

    logger.info("")
    logger.info(f"DeepMCCFR defeated {wins}/{total_matches} opponents")
    logger.info(f"Overall average winrate: {overall_winrate:.3f}")

    if wins >= total_matches * 0.8:
        logger.info("🎉 DeepMCCFR is performing excellently!")
    elif wins >= total_matches * 0.6:
        logger.info("👍 DeepMCCFR is performing well")
    elif wins >= total_matches * 0.4:
        logger.info("📈 DeepMCCFR shows promise, may need more training")
    else:
        logger.info("📉 DeepMCCFR needs improvement")

    # Save detailed results
    results_path = Path("data/evaluation_results.json")
    tournament.save_results(
        {
            "evaluation_type": "deep_mccfr_vs_baselines",
            "deep_agent_path": str(deep_model_path),
            "num_games_per_match": num_games,
            "matches": results,
            "summary": {
                "wins": wins,
                "total_matches": total_matches,
                "overall_winrate": overall_winrate,
            },
        },
        results_path,
    )


def run_full_tournament() -> None:
    """Run a full round-robin tournament."""
    logger.info("Full Agent Tournament")
    logger.info("=" * 25)

    tournament = Tournament(seed=42)

    # Collect all agents
    agents = []

    # Add baseline agents (fewer for tournament)
    agents.extend(
        [
            HeuristicAgent(16, "Heuristic_16"),
            HeuristicAgent(17, "Heuristic_17"),
            HeuristicAgent(18, "Heuristic_18"),
        ]
    )

    # Add MCCFR if available
    mccfr_policy_path = Path("data/policy_mccfr.json")
    if mccfr_policy_path.exists():
        agents.append(PolicyAgent(mccfr_policy_path, "MCCFR"))

    # Add Deep MCCFR
    deep_model_path = Path("data/deep_mccfr_model_final.pth")
    if deep_model_path.exists():
        agents.append(DeepMCCFRAgent(deep_model_path, "DeepMCCFR"))
    else:
        # Look for latest checkpoint
        checkpoints = list(Path("data").glob("deep_mccfr_model_*.pth"))
        if checkpoints:
            deep_model_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            agents.append(DeepMCCFRAgent(deep_model_path, "DeepMCCFR"))

    if len(agents) < 2:
        logger.error("Need at least 2 agents for tournament!")
        return

    logger.info(f"Tournament with {len(agents)} agents:")
    for agent in agents:
        logger.info(f"  - {agent.name()}")
    logger.info("")

    # Run tournament
    num_games = 1000  # Games per match
    results = tournament.run_tournament(agents, num_games)

    # Save results
    results_path = Path("data/tournament_results.json")
    tournament.save_results(results, results_path)

    logger.info("")
    logger.info("Tournament complete! Check results in data/tournament_results.json")


def main() -> None:
    """Main evaluation script."""
    if len(sys.argv) > 1 and sys.argv[1] == "tournament":
        run_full_tournament()
    else:
        evaluate_deep_mccfr()


if __name__ == "__main__":
    main()
