"""Tournament evaluation for comparing agents."""

import ast
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import twentyone
from loguru import logger


class AgentInterface:
    """Interface for tournament agents."""

    def choose_action(
        self, obs: twentyone.Observation, player: int, round_num: int
    ) -> twentyone.Action:
        """Choose action given observation."""
        raise NotImplementedError

    def name(self) -> str:
        """Return agent name."""
        raise NotImplementedError


class PolicyAgent(AgentInterface):
    """Agent that uses a saved policy."""

    def __init__(self, policy_path: Path, agent_name: str):
        self.policy_path = policy_path
        self.agent_name = agent_name
        self.policy = self._load_policy(policy_path)

    def _load_policy(self, path: Path) -> dict[str, Any]:
        """Load policy from JSON file."""
        with open(path) as f:
            raw = json.load(f)

        # Handle different policy formats
        if isinstance(raw, dict) and "simple_strategy" in raw:
            return raw  # Simple strategy format

        if isinstance(raw, dict) and "agent_type" in raw:
            return raw  # Deep MCCFR format

        # Traditional MCCFR format - keys are stringified tuples
        try:
            pol = {ast.literal_eval(k): v for k, v in raw.items()}
            return pol
        except (ValueError, SyntaxError):
            return raw

    def _infoset_from_obs(self, obs: twentyone.Observation, player: int) -> tuple:
        """Create information set from observation."""
        bucket = int(obs.deck_count) & 0xF
        return (
            player,
            obs.self_total,
            obs.opp_face_up,
            obs.self_stood,
            obs.opp_stood,
            bucket,
        )

    def choose_action(
        self, obs: twentyone.Observation, player: int, round_num: int
    ) -> twentyone.Action:
        """Choose action based on policy."""
        # Handle simple strategy format
        if "simple_strategy" in self.policy:
            threshold = self.policy["simple_strategy"].get("draw_threshold", 17)
            return twentyone.Action.Draw if obs.self_total < threshold else twentyone.Action.Stand

        # Handle deep MCCFR format
        if "agent_type" in self.policy and self.policy["agent_type"] == "deep_mccfr":
            # For now, use a heuristic since we'd need the actual model
            # In practice, this would load and use the neural network
            return twentyone.Action.Draw if obs.self_total < 17 else twentyone.Action.Stand

        # Handle traditional MCCFR policy format
        info = self._infoset_from_obs(obs, player)
        strat = self.policy.get(info)
        if strat is None:
            # Fallback heuristic
            return twentyone.Action.Draw if obs.self_total < 17 else twentyone.Action.Stand

        return twentyone.Action.Draw if strat[0] >= strat[1] else twentyone.Action.Stand

    def name(self) -> str:
        return self.agent_name


class DeepMCCFRAgent(AgentInterface):
    """Agent wrapper for DeepMCCFR."""

    def __init__(self, model_path: Path | None = None, agent_name: str = "DeepMCCFR"):
        from twentyone_rl.agents.deep_mccfr import DeepMCCFR

        try:
            self.agent = DeepMCCFR()
            self.agent_name = agent_name

            if model_path and model_path.exists():
                logger.info(f"Loading Deep MCCFR model from {model_path}")
                self.agent.load_model(model_path)
                logger.info(f"Successfully loaded model with input_dim={self.agent.input_dim}")
            elif model_path:
                logger.warning(f"Model path {model_path} does not exist, using untrained agent")
        except Exception as e:
            logger.error(f"Failed to initialize DeepMCCFR agent: {e}")
            logger.error("This often indicates model compatibility issues or missing dependencies")
            raise RuntimeError(f"DeepMCCFR initialization failed: {e}") from e

    def choose_action(
        self, obs: twentyone.Observation, player: int, round_num: int
    ) -> twentyone.Action:
        try:
            return self.agent.choose_action(obs, player, round_num)
        except Exception as e:
            logger.warning(f"DeepMCCFR agent error: {e}, falling back to heuristic")
            # Fallback to simple heuristic if neural network fails
            return twentyone.Action.Draw if obs.self_total < 17 else twentyone.Action.Stand

    def name(self) -> str:
        return self.agent_name


class TabularCFRAgent(AgentInterface):
    """Agent wrapper for Tabular CFR."""

    def __init__(self, model_path: Path | None = None, agent_name: str = "TabularCFR"):
        from twentyone_rl.agents.tabular_cfr import TabularCFR

        try:
            self.agent = TabularCFR()
            self.agent_name = agent_name

            if model_path and model_path.exists():
                logger.info(f"Loading Tabular CFR model from {model_path}")
                self.agent.load_model(model_path)
                logger.info(
                    f"Successfully loaded model with {len(self.agent.regret_sum)} information sets"
                )
            elif model_path:
                logger.warning(f"Model path {model_path} does not exist, using untrained agent")
        except Exception as e:
            logger.error(f"Failed to initialize Tabular CFR agent: {e}")
            raise RuntimeError(f"Tabular CFR initialization failed: {e}") from e

    def choose_action(
        self, obs: twentyone.Observation, player: int, round_num: int
    ) -> twentyone.Action:
        try:
            return self.agent.select_action(obs, player, round_num)
        except Exception as e:
            logger.warning(f"Tabular CFR agent error: {e}, falling back to heuristic")
            # Fallback to simple heuristic if agent fails
            return twentyone.Action.Draw if obs.self_total < 17 else twentyone.Action.Stand

    def name(self) -> str:
        return self.agent_name


class HeuristicAgent(AgentInterface):
    """Simple heuristic agent for baseline comparison."""

    def __init__(self, threshold: int = 17, agent_name: str = "Heuristic"):
        self.threshold = threshold
        self.agent_name = agent_name

    def choose_action(
        self, obs: twentyone.Observation, player: int, round_num: int
    ) -> twentyone.Action:
        return twentyone.Action.Draw if obs.self_total < self.threshold else twentyone.Action.Stand

    def name(self) -> str:
        return f"{self.agent_name}({self.threshold})"


class Tournament:
    """Tournament system for evaluating agents."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.random = random.Random(seed)

    def play_game(
        self, agent1: AgentInterface, agent2: AgentInterface, game_seed: int | None = None
    ) -> tuple[int, dict[str, Any]]:
        """Play a single game between two agents."""
        if game_seed is None:
            game_seed = self.random.randint(0, 2**31)

        env = twentyone.Env(seed=game_seed)
        game_stats = {"rounds": 0, "total_actions": 0, "winner": None, "final_hearts": [0, 0]}

        agents = [agent1, agent2]

        while True:
            env.start_new_round()
            game_stats["rounds"] += 1
            round_num = env.round()

            while True:
                current_player = env.current_player()
                obs = env.observation(current_player)

                action = agents[current_player].choose_action(obs, current_player, round_num)
                game_stats["total_actions"] += 1

                result = env.step(action)

                if result.round_over or result.game_over:
                    if result.game_over:
                        # Determine winner based on hearts
                        p0_hearts = env.hearts(0)
                        p1_hearts = env.hearts(1)
                        if p0_hearts > p1_hearts:
                            winner = 0
                        elif p1_hearts > p0_hearts:
                            winner = 1
                        else:
                            winner = None  # Tie (shouldn't happen in this game)

                        game_stats["winner"] = winner
                        game_stats["final_hearts"] = [p0_hearts, p1_hearts]
                        return winner, game_stats
                    break

    def run_match(
        self, agent1: AgentInterface, agent2: AgentInterface, num_games: int = 1000
    ) -> dict[str, Any]:
        """Run a match between two agents."""

        results = {
            "agent1_name": agent1.name(),
            "agent2_name": agent2.name(),
            "num_games": num_games,
            "agent1_wins": 0,
            "agent2_wins": 0,
            "ties": 0,
            "games": [],
        }

        for i in range(num_games):
            winner, game_stats = self.play_game(agent1, agent2)

            if winner == 0:
                results["agent1_wins"] += 1
            elif winner == 1:
                results["agent2_wins"] += 1
            else:
                results["ties"] += 1

            results["games"].append({"game_id": i, "winner": winner, "stats": game_stats})

        # Calculate statistics
        results["agent1_winrate"] = results["agent1_wins"] / num_games
        results["agent2_winrate"] = results["agent2_wins"] / num_games
        results["tie_rate"] = results["ties"] / num_games

        # Calculate confidence intervals for win rates
        results["agent1_ci"] = self._confidence_interval(results["agent1_wins"], num_games)
        results["agent2_ci"] = self._confidence_interval(results["agent2_wins"], num_games)

        return results

    def _confidence_interval(
        self, wins: int, total: int, confidence: float = 0.95
    ) -> tuple[float, float]:
        """Calculate confidence interval for win rate."""
        if total == 0:
            return (0.0, 0.0)

        p = wins / total
        z = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        margin = z * np.sqrt(p * (1 - p) / total)

        return (max(0, p - margin), min(1, p + margin))

    def run_tournament(self, agents: list[AgentInterface], num_games: int = 1000) -> dict[str, Any]:
        """Run round-robin tournament between multiple agents."""

        results = {
            "agents": [agent.name() for agent in agents],
            "num_games_per_match": num_games,
            "matches": {},
            "leaderboard": [],
        }

        # Run all pairwise matches
        for i in range(len(agents)):
            for j in range(i + 1, len(agents)):
                agent1, agent2 = agents[i], agents[j]
                match_result = self.run_match(agent1, agent2, num_games)

                match_key = f"{agent1.name()}_vs_{agent2.name()}"
                results["matches"][match_key] = match_result

        # Calculate overall statistics
        agent_stats = {}
        for agent in agents:
            agent_stats[agent.name()] = {
                "total_wins": 0,
                "total_games": 0,
                "winrate": 0.0,
                "opponents_beaten": 0,
            }

        for match_result in results["matches"].values():
            agent1_name = match_result["agent1_name"]
            agent2_name = match_result["agent2_name"]

            agent_stats[agent1_name]["total_wins"] += match_result["agent1_wins"]
            agent_stats[agent1_name]["total_games"] += num_games

            agent_stats[agent2_name]["total_wins"] += match_result["agent2_wins"]
            agent_stats[agent2_name]["total_games"] += num_games

            # Track head-to-head wins
            if match_result["agent1_winrate"] > match_result["agent2_winrate"]:
                agent_stats[agent1_name]["opponents_beaten"] += 1
            elif match_result["agent2_winrate"] > match_result["agent1_winrate"]:
                agent_stats[agent2_name]["opponents_beaten"] += 1

        # Calculate final win rates and create leaderboard
        for name, stats in agent_stats.items():
            if stats["total_games"] > 0:
                stats["winrate"] = stats["total_wins"] / stats["total_games"]

        results["leaderboard"] = sorted(
            agent_stats.items(),
            key=lambda x: (x[1]["winrate"], x[1]["opponents_beaten"]),
            reverse=True,
        )

        logger.info("Tournament completed!")
        logger.info("Leaderboard:")
        for i, (name, stats) in enumerate(results["leaderboard"]):
            logger.info(
                f"{i+1}. {name}: {stats['winrate']:.3f} "
                f"({stats['total_wins']}/{stats['total_games']}) "
                f"[Beat {stats['opponents_beaten']} opponents]"
            )

        return results

    def save_results(self, results: dict[str, Any], path: Path) -> None:
        """Save tournament results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {path}")


def load_agent_from_config(config: dict[str, Any]) -> AgentInterface:
    """Load agent from configuration."""
    agent_type = config["type"]

    if agent_type == "policy":
        return PolicyAgent(Path(config["policy_path"]), config.get("name", "PolicyAgent"))
    elif agent_type == "deep_mccfr":
        model_path = Path(config["model_path"]) if "model_path" in config else None
        return DeepMCCFRAgent(model_path, config.get("name", "DeepMCCFR"))
    elif agent_type == "heuristic":
        return HeuristicAgent(config.get("threshold", 17), config.get("name", "Heuristic"))
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
