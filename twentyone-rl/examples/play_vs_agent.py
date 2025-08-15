import sys
from pathlib import Path
from typing import Any

# Add the twentyone package to the path
sys.path.insert(0, str(Path(__file__).parent / "../../twentyone-py/python"))

# Add the RL package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pathlib import Path

import twentyone

from twentyone_rl.agents.deep_mccfr import DeepMCCFR
from twentyone_rl.display import (
    check_game_over,
    show_action_choice,
    show_game_start,
    show_hearts_status,
    show_round_result,
    show_round_start,
    show_turn_info,
)


def load_policy(path: Path) -> dict[str, Any]:
    """Load a policy from a JSON file."""
    import ast
    import json

    with open(path) as f:
        raw = json.load(f)

    # Handle different policy formats
    if isinstance(raw, dict) and "simple_strategy" in raw:
        return raw  # Simple strategy format

    if isinstance(raw, dict) and "agent_type" in raw:
        return raw  # Deep MCCFR or other agent type format

    # Traditional MCCFR format - keys are stringified tuples; parse back with ast.literal_eval
    try:
        pol = {ast.literal_eval(k): v for k, v in raw.items()}
        return pol
    except (ValueError, SyntaxError):
        # If parsing fails, return raw dict
        return raw


def infoset_from_obs(obs: twentyone.Observation, player: int, deck_mask_est: int = 0) -> tuple:
    """Create an information set from observation."""
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
    policy: dict[str, Any], obs: twentyone.Observation, player: int, round_num: int
) -> twentyone.Action:
    """Choose action based on policy."""
    # Handle simple strategy format
    if "simple_strategy" in policy:
        threshold = policy["simple_strategy"].get("draw_threshold", 17)
        return twentyone.Action.Draw if obs.self_total < threshold else twentyone.Action.Stand

    # Handle Deep MCCFR format
    if "agent_type" in policy and policy["agent_type"] == "deep_mccfr":
        # Load and use the actual Deep MCCFR agent
        if hasattr(choose_action, "_deep_agent"):
            agent = choose_action._deep_agent
        else:
            agent = DeepMCCFR()
            model_path = Path(policy.get("model_path", "data/deep_mccfr_model_final.pth"))
            if model_path.exists():
                agent.load_model(model_path)
            choose_action._deep_agent = agent

        return agent.choose_action(obs, player, round_num)

    # Handle traditional MCCFR policy format
    info = infoset_from_obs(obs, player)
    strat = policy.get(info)
    if strat is None:
        # fallback heuristic
        return twentyone.Action.Draw if obs.self_total < 17 else twentyone.Action.Stand
    return twentyone.Action.Draw if strat[0] >= strat[1] else twentyone.Action.Stand


def main() -> None:
    """Run interactive gameplay."""
    if len(sys.argv) < 2:
        print("Usage: play_vs_agent.py policy_file.json [seed]")
        return

    policy_path = Path(sys.argv[1])
    if not policy_path.exists():
        print(f"Policy file not found: {policy_path}")
        return

    policy = load_policy(policy_path)

    # Use provided seed or default
    seed = int(sys.argv[2], 0) if len(sys.argv) > 2 else 42

    player_names = ("You", "Agent")
    p_user = 0  # user plays as player 0

    # Create environment and show game start
    env = twentyone.Env(seed=seed)
    show_game_start(env, seed, player_names)

    while True:
        # Start new round
        env.start_new_round()
        round_num = env.round()
        show_round_start(round_num)

        # Play until round ends
        while True:
            player = env.current_player()
            obs = env.observation(player)

            if player == p_user:
                # Human player turn
                p0_up, p1_up = show_turn_info(env, player, obs, player_names)

                while True:
                    choice = input("Type d=draw, s=stand: ").strip().lower()
                    if choice.startswith("d"):
                        action = twentyone.Action.Draw
                        break
                    elif choice.startswith("s"):
                        action = twentyone.Action.Stand
                        break
                    else:
                        print("Please enter 'd' for draw or 's' for stand.")

                show_action_choice(player_names[player], action)
            else:
                # Agent turn
                p0_up, p1_up = show_turn_info(env, player, obs, player_names)
                action = choose_action(policy, obs, player, round_num)
                show_action_choice(player_names[player], action)

            # Take step
            result = env.step(action)

            if result.round_over:
                # Show round results
                show_round_result(env, result, p0_up, p1_up, player_names)
                show_hearts_status(env, player_names)

                if check_game_over(env):
                    return
                break


if __name__ == "__main__":
    main()
