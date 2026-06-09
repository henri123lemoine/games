import twentyone

from twentyone_rl.display import (
    check_game_over,
    show_action_choice,
    show_game_start,
    show_hearts_status,
    show_round_result,
    show_round_start,
    show_turn_info,
)


def simple_strategy(obs: twentyone.Observation) -> twentyone.Action:
    """Simple strategy: draw if total < 17, otherwise stand."""
    return twentyone.Action.Draw if obs.self_total < 17 else twentyone.Action.Stand


def main() -> None:
    """Run a simple game demonstration."""
    seed = 42
    player_names = ("Bot0", "Bot1")

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

            # Show turn info and get public cards
            p0_up, p1_up = show_turn_info(env, player, obs, player_names)

            # Get action from simple strategy
            action = simple_strategy(obs)
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
