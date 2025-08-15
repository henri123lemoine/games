"""Type stubs for twentyone package."""

from typing import Optional, List, Dict, Any

class Action:
    """Action available to the current player: draw a card or stand."""
    Draw: Action
    Stand: Action

class Observation:
    """Partial observation available to an agent controlling one player."""
    self_total: int
    opp_face_up: int
    self_face_up: int
    self_face_down: int
    self_stood: bool
    opp_stood: bool
    deck_count: int
    round: int
    self_hearts: int
    opp_hearts: int
    
    def to_dict(self) -> Dict[str, Any]: ...

class RoundOutcome:
    """Result of a completed round."""
    winner: Optional[int]
    damage: int

class StepResult:
    """Result of a step in the game."""
    round_over: bool
    game_over: bool
    outcome: Optional[RoundOutcome]

class Env:
    """Twenty-One game environment for two players."""
    
    def __init__(self, seed: int) -> None: ...
    
    @classmethod
    def with_preset_decks(cls, preset_decks: List[List[int]]) -> "Env": ...
    
    def start_new_round(self) -> None: ...
    def observation(self, player: int) -> Observation: ...
    def step(self, action: Action) -> StepResult: ...
    def round(self) -> int: ...
    def hearts(self, player: int) -> int: ...
    def current_player(self) -> int: ...
    def public_up_cards(self, player: int) -> Optional[List[int]]: ...
    def last_reveal(self) -> Optional[List[int]]: ...