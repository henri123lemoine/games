# Four-player chess

This crate implements the standard Chess.com free-for-all (FFA) game on the
14×14 cross board. It is four independent players, not two teams: Red, Blue,
Yellow, and Green maximize their own score and move clockwise in that order.

## Rules contract

The normative external reference is Chess.com's Help Center article
[“4 Player Chess (4PC)”](https://support.chess.com/en/articles/8614233-4-player-chess-4pc),
dated **October 10, 2025**. That date is pinned because Chess.com has changed
4PC rules over time and its older “Chess Terms” page still describes a
different opponent-stalemate award. The modern start squares are cross-checked
against the standard-board illustration on that official
[Chess Terms page](https://www.chess.com/terms/4-player-chess); where its older
scoring prose conflicts, the newer dated Help Center contract controls.

Piece movement follows ordinary chess, oriented from each army's home edge,
with these FFA rules:

- The board has 160 playable squares: a 14×14 square with each 3×3 corner
  removed. The modern Chess.com setup places the four kings on h1, a8, g14,
  and n7 (Blue and Green use the site's central king/queen orientation), with
  eight pawns per army.
- Pawns move toward the opposite arm, may double from their starting rank, and
  promote automatically to a queen on their own eighth rank. A promoted queen
  remains marked as a one-point queen.
- Castling follows ordinary chess safety rules. En passant remains available
  until the double-pushing army's next turn, so every intervening opponent gets
  its own next move as an opportunity.
- A player must answer check on their turn. Checkmate is determined only when
  the checked player's turn is reached. Capturing a live king also eliminates
  its army.
- Checkmate is worth 20 points to the player who made the preceding move.
  Self-stalemate is worth 20 points to the stalemated player.
- Capturing a live pawn or promoted queen scores 1; knight 3; bishop 5; rook 5;
  original queen 9; king 20. An eliminated army is grey/dead: its pieces stay
  as inert blockers, may be captured, and score no points.
- A simultaneous check of two kings scores 1 when delivered by a queen and 5
  otherwise. A simultaneous check of three scores 5 with a queen and 20
  otherwise.
- The board game ends when only one active army remains. Threefold repetition,
  fifty complete four-player rounds without a pawn move/capture (200 plies),
  and insufficient material instead award 10 points to every active army and
  end the game. Final placement is by total points.

Clock, disconnect, and early-abort policy are server/session concerns rather
than search actions. The terminal, arena, self-play, and site matches do not
run clocks or make bots resign. Consequently the Help Center's timeout-only
random “spare king” process is unreachable in normal play; every reachable
board position and score transition follows the pinned rules above.

## Engine and learning contracts

`FourPlayerChess` implements the shared `Game` and `GameUi` traits. Its value
is an absolute four-seat, zero-sum vector determined by final point placement;
there is no partner symmetry. `FourPlayerChessEncoder` uses 71 absolute-seat
planes over the 14×14 tensor and a 112-plane move encoding per origin square.
The policy width is 21,952 and the value head has four seat logits.

Evaluation rotates the hero through all four seats against a field of three
opponents. Strict win share is reported against the fair baseline of 25%; score
share is reported separately by the game-specific evaluator.
