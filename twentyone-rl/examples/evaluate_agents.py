import sys
from pathlib import Path

# Add the twentyone package to the path
sys.path.insert(0, str(Path(__file__).parent / "../../twentyone-py/python"))

# Add the RL package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger

from twentyone_rl.evaluation.tournament import (
    AgentInterface,
    DeepMCCFRAgent,
    HeuristicAgent,
    PolicyAgent,
    TabularCFRAgent,
    Tournament,
)


def create_baseline_agents() -> list[AgentInterface]:
    """Create baseline agents for comparison."""
    agents: list[AgentInterface] = []

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
    overall_winrate = 0.0

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
    logger.info("Full Agent Tournament - Round Robin")
    logger.info("=" * 40)

    tournament = Tournament(seed=42)

    # Collect all agents
    agents: list[AgentInterface] = []

    # Add all heuristic agents
    agents.extend(
        [
            HeuristicAgent(15, "Heuristic_15"),
            HeuristicAgent(16, "Heuristic_16"),
            HeuristicAgent(17, "Heuristic_17"),
            HeuristicAgent(18, "Heuristic_18"),
            HeuristicAgent(19, "Heuristic_19"),
        ]
    )

    # Add MCCFR if available
    mccfr_policy_path = Path("data/policy_mccfr.json")
    if mccfr_policy_path.exists():
        agents.append(PolicyAgent(mccfr_policy_path, "MCCFR"))
        logger.info("✓ Added traditional MCCFR agent")
    else:
        logger.warning("✗ Traditional MCCFR agent not found")

    # Add Tabular CFR
    tabular_model_path = Path("data/tabular_cfr_model_final.json")
    if tabular_model_path.exists():
        agents.append(TabularCFRAgent(tabular_model_path, "TabularCFR"))
        logger.info("✓ Added Tabular CFR agent (final model)")
    else:
        # Look for latest checkpoint
        tabular_checkpoints = list(Path("data").glob("tabular_cfr_model_*.json"))
        if tabular_checkpoints:
            tabular_model_path = max(tabular_checkpoints, key=lambda p: p.stat().st_mtime)
            agents.append(
                TabularCFRAgent(tabular_model_path, f"TabularCFR_{tabular_model_path.stem.split('_')[-1]}")
            )
            logger.info(f"✓ Added Tabular CFR agent ({tabular_model_path.name})")
        else:
            logger.warning("✗ No Tabular CFR models found")

    # Add Deep MCCFR
    deep_model_path = Path("data/deep_mccfr_model_final.pth")
    if deep_model_path.exists():
        agents.append(DeepMCCFRAgent(deep_model_path, "DeepMCCFR"))
        logger.info("✓ Added Deep MCCFR agent (final model)")
    else:
        # Look for latest checkpoint
        checkpoints = list(Path("data").glob("deep_mccfr_model_*.pth"))
        if checkpoints:
            deep_model_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
            agents.append(
                DeepMCCFRAgent(deep_model_path, f"DeepMCCFR_{deep_model_path.stem.split('_')[-1]}")
            )
            logger.info(f"✓ Added Deep MCCFR agent ({deep_model_path.name})")
        else:
            logger.warning("✗ No Deep MCCFR models found")

    if len(agents) < 2:
        logger.error("Need at least 2 agents for tournament!")
        return

    logger.info("")
    logger.info(f"Tournament Participants ({len(agents)} agents):")
    for i, agent in enumerate(agents, 1):
        logger.info(f"  {i}. {agent.name()}")
    logger.info("")

    # Run tournament
    num_games = 500  # Games per match (reduced for faster testing)
    logger.info(f"Running round-robin tournament: {num_games} games per match")
    logger.info(f"Total matches: {len(agents) * (len(agents) - 1) // 2}")
    logger.info("")

    results = tournament.run_tournament(agents, num_games)

    # Display detailed results
    logger.info("=" * 50)
    logger.info("TOURNAMENT RESULTS")
    logger.info("=" * 50)

    # Show head-to-head matrix
    logger.info("")
    logger.info("Head-to-Head Win Rates:")
    logger.info("-" * 30)

    agent_names = [agent.name() for agent in agents]
    matches = results.get("matches", {})

    # Create and display win rate matrix
    for i, agent1 in enumerate(agent_names):
        for j, agent2 in enumerate(agent_names):
            if i < j:  # Only show upper triangle
                match_key = f"{agent1}_vs_{agent2}"
                if match_key in matches:
                    match_result = matches[match_key]
                    winrate1 = match_result["agent1_winrate"]
                    winrate2 = match_result["agent2_winrate"]
                    logger.info(f"{agent1} vs {agent2}: {winrate1:.3f} - {winrate2:.3f}")

    # Display final leaderboard
    logger.info("")
    logger.info("FINAL LEADERBOARD:")
    logger.info("=" * 20)

    leaderboard = results.get("leaderboard", [])
    for i, (agent_name, stats) in enumerate(leaderboard, 1):
        overall_winrate = stats["winrate"]
        wins = stats["total_wins"]
        total_games = stats["total_games"]
        opponents_beaten = stats["opponents_beaten"]

        logger.info(f"{i}. {agent_name}")
        logger.info(f"   Win Rate: {overall_winrate:.3f} ({wins}/{total_games})")
        logger.info(f"   Opponents Beaten: {opponents_beaten}/{len(agents)-1}")
        logger.info("")

    # Save results
    results_path = Path("data/tournament_results.json")
    tournament.save_results(results, results_path)

    logger.info(f"Detailed results saved to: {results_path}")
    logger.info("Tournament complete!")


def main() -> None:
    """Main evaluation script."""
    if len(sys.argv) > 1 and sys.argv[1] == "deep_only":
        evaluate_deep_mccfr()
    else:
        run_full_tournament()


if __name__ == "__main__":
    main()
