import sys
from pathlib import Path
from typing import Dict, Any

# Add the twentyone package to the path
sys.path.insert(0, str(Path(__file__).parent / "../../twentyone-py/python"))

# Add the RL package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import twentyone
from twentyone_rl.display import (
    show_game_start,
    show_round_start,
    show_turn_info,
    show_action_choice,
    show_round_result,
    show_hearts_status,
    check_game_over,
    get_public_cards,
)


def load_policy(path: Path) -> Dict[str, Any]:
    """Load a policy from a JSON file."""
    import json
    import ast

    with open(path) as f:
        raw = json.load(f)

    # Handle different policy formats
    if isinstance(raw, dict) and "simple_strategy" in raw:
        return raw  # Simple strategy format

    # keys are stringified tuples; parse back with ast.literal_eval
    pol = {ast.literal_eval(k): v for k, v in raw.items()}
    return pol


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


def choose_action(policy: Dict[str, Any], obs: twentyone.Observation, player: int) -> twentyone.Action:
    """Choose action based on policy."""
    # Handle simple strategy format
    if "simple_strategy" in policy:
        threshold = policy["simple_strategy"].get("draw_threshold", 17)
        return twentyone.Action.Draw if obs.self_total < threshold else twentyone.Action.Stand

    # Handle MCCFR policy format
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
                    if choice.startswith('d'):
                        action = twentyone.Action.Draw
                        break
                    elif choice.startswith('s'):
                        action = twentyone.Action.Stand
                        break
                    else:
                        print("Please enter 'd' for draw or 's' for stand.")

                show_action_choice(player_names[player], action)
            else:
                # Agent turn
                p0_up, p1_up = show_turn_info(env, player, obs, player_names)
                action = choose_action(policy, obs, player)
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
