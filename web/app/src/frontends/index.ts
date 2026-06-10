// Frontend registry: game id → its frontend package. A game without an entry
// gets the generic fallback, so new Rust games are playable immediately.

import { GenericFrontend } from './generic';
import type { FrontendFactory, GameFrontend } from './types';

const registry: Record<string, FrontendFactory> = {};

export function registerFrontend(gameId: string, factory: FrontendFactory): void {
  registry[gameId] = factory;
}

export function frontendFor(gameId: string): GameFrontend {
  const factory = registry[gameId];
  return factory ? factory() : new GenericFrontend();
}
