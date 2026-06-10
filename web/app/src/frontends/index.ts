// Frontend registry: game id → its frontend package. A game without an entry
// gets the generic fallback, so new Rust games are playable immediately.

import { createG2048Frontend } from './2048';
import { createChessFrontend } from './chess';
import { createConnect4Frontend } from './connect4';
import { GenericFrontend } from './generic';
import { createGoFrontend } from './go';
import { createLiarsDiceFrontend } from './liars-dice';
import { createOthelloFrontend } from './othello';
import { createSnakeFrontend } from './snake';
import { createTwentyOneFrontend } from './twentyone';
import type { FrontendFactory, GameFrontend } from './types';

const registry: Record<string, FrontendFactory> = {
  '2048': createG2048Frontend,
  chess: createChessFrontend,
  connect4: createConnect4Frontend,
  go: createGoFrontend,
  'liars-dice': createLiarsDiceFrontend,
  othello: createOthelloFrontend,
  snake: createSnakeFrontend,
  twentyone: createTwentyOneFrontend,
};

export function registerFrontend(gameId: string, factory: FrontendFactory): void {
  registry[gameId] = factory;
}

export function frontendFor(gameId: string): GameFrontend {
  const factory = registry[gameId];
  return factory ? factory() : new GenericFrontend();
}
