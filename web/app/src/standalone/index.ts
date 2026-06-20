// Registry of standalone arcade games — real-time canvas games that live
// entirely on the client and do not pass through the wasm engine. The shell
// merges these onto the home grid and launches them in the match screen.

import { SlitherGame } from './slither/game';
import type { StandaloneInfo } from './types';

const SLITHER_MINI = `<div class="mini mini-slither">
  <span class="mini-worm"></span>
  <span class="mini-food" style="left:20%;top:30%"></span>
  <span class="mini-food" style="left:74%;top:24%"></span>
  <span class="mini-food" style="left:60%;top:70%"></span>
  <span class="mini-food" style="left:32%;top:78%"></span>
</div>`;

const GAMES: StandaloneInfo[] = [
  {
    id: 'slither',
    name: 'Slither',
    mini: SLITHER_MINI,
    create: () => new SlitherGame(),
  },
];

export function standaloneGames(): StandaloneInfo[] {
  return GAMES;
}

export function standaloneInfo(id: string): StandaloneInfo | undefined {
  return GAMES.find((g) => g.id === id);
}
