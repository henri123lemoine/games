"""Display utilities for clean game output."""

from typing import Optional, List
import twentyone


def show_game_start(env: twentyone.Env, seed: int, player_names: tuple[str, str]) -> None:
    """Display game initialization information."""
    print(f"Seed: {seed}")
    print("New Game.")
    print(f"Hearts: {player_names[0]}={env.hearts(0)} {player_names[1]}={env.hearts(1)}")


def show_round_start(round_num: int) -> None:
    """Display round start information."""
    print("")
    print(f"Round {round_num}!")
    print("")


def get_public_cards(env: twentyone.Env) -> tuple[List[int], List[int]]:
    """Get public card information as lists."""
    p0_up_raw = env.public_up_cards(0)
    p1_up_raw = env.public_up_cards(1)
    p0_up = list(p0_up_raw) if p0_up_raw else []
    p1_up = list(p1_up_raw) if p1_up_raw else []
    return p0_up, p1_up


def show_turn_info(
    env: twentyone.Env, 
    current_player: int, 
    obs: twentyone.Observation, 
    player_names: tuple[str, str]
) -> tuple[List[int], List[int]]:
    """Display turn information and return public card info."""
    p0_up, p1_up = get_public_cards(env)

    print(f"{player_names[current_player]}'s turn.")
    if current_player == 0:
        print(f"{player_names[0]}: hidden=[{obs.self_face_down}], shown={p0_up}")
        print(f"{player_names[1]}: shown={p1_up}")
    else:
        print(f"{player_names[1]}: hidden=[{obs.self_face_down}], shown={p1_up}")
        print(f"{player_names[0]}: shown={p0_up}")

    print(f"Cards remaining in the deck: {obs.deck_count}")
    return p0_up, p1_up


def show_action_choice(player_name: str, action: twentyone.Action) -> None:
    """Display player's action choice."""
    action_str = "draw" if action == twentyone.Action.Draw else "stand"
    print(f"{player_name} chooses: {action_str}")


def show_round_result(
    env: twentyone.Env,
    result: twentyone.StepResult,
    p0_up: List[int],
    p1_up: List[int],
    player_names: tuple[str, str]
) -> None:
    """Display round end results including final hands and outcome."""
    reveal = env.last_reveal()
    reveal_list = list(reveal) if reveal else [None, None]
    
    p0_final = p0_up + ([reveal_list[0]] if reveal_list[0] is not None else [])
    p1_final = p1_up + ([reveal_list[1]] if reveal_list[1] is not None else [])
    p0_total = sum(p0_final)
    p1_total = sum(p1_final)

    print("")
    print("Round over. Final cards:")
    print(f"{player_names[0]}: {p0_final}, total={p0_total}")
    print(f"{player_names[1]}: {p1_final}, total={p1_total}")

    if result.outcome:
        if result.outcome.winner is not None:
            print(f"Outcome: {player_names[result.outcome.winner]} wins.")
        else:
            print("Outcome: Tie.")


def show_hearts_status(env: twentyone.Env, player_names: tuple[str, str]) -> None:
    """Display current hearts status."""
    print("")
    print("Hearts:")
    print(f"{player_names[0]}: {env.hearts(0)}")
    print(f"{player_names[1]}: {env.hearts(1)}")


def check_game_over(env: twentyone.Env) -> bool:
    """Check if the game is over and display message."""
    if env.hearts(0) == 0 or env.hearts(1) == 0:
        print("")
        print("Game over.")
        return True
    return False