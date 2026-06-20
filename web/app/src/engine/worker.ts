// The engine worker: owns the wasm instance and at most one live match.
// Bot search can take seconds; running here keeps the page responsive.

import init, {
  AzChessBot,
  AzGoBot,
  AzSnakeBot,
  WebMatch,
  create_match,
  elo,
  fit_elo_table,
  list_games,
  load_artifact,
  play_field,
  play_pairs,
} from 'web-engine';
import wasmUrl from 'web-engine/web_engine_bg.wasm?url';
import type { EngineRequest, EngineResponse, ViewState } from './protocol';

let match: WebMatch | null = null;
// One client-driven search bot at a time; chess and go share the push/
// advance/best surface, so the rest of the az* ops are bot-agnostic.
let azBot: AzChessBot | AzGoBot | null = null;
// Snake has its own surface: it reconstructs its search root from the view
// JSON each move (chance nodes make move-mirroring unsound), so it does not
// share the push/advance protocol.
let snakeBot: AzSnakeBot | null = null;
const ready = init({ module_or_path: wasmUrl });

function state(): ViewState {
  if (!match) throw new Error('no live match');
  return {
    view: match.view(),
    viewData: parseMaybe(match.view_data()),
    labels: JSON.parse(match.legal_labels()) as string[],
    toAct: match.to_act(),
    isOver: match.is_over(),
    result: match.is_over() ? match.result_text() : null,
    humanSeat: match.human_seat(),
    numSeats: match.num_seats(),
  };
}

function parseMaybe(s: string | undefined): unknown {
  if (s === undefined || s === '') return null;
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

function handle(req: EngineRequest): unknown {
  switch (req.op) {
    case 'manifest':
      return JSON.parse(list_games());
    case 'create':
      match?.free();
      match = create_match(req.game, JSON.stringify(req.opts));
      return state();
    case 'step': {
      if (!match) throw new Error('no live match');
      const ev = match.step();
      return ev ? JSON.parse(ev) : null;
    }
    case 'state':
      return state();
    case 'apply': {
      if (!match) throw new Error('no live match');
      return JSON.parse(match.apply_human(req.input));
    }
    case 'artifact':
      load_artifact(req.key, new Uint8Array(req.bytes));
      return null;
    case 'pairs':
      return JSON.parse(
        play_pairs(req.game, JSON.stringify(req.opts), req.a, req.b, req.seed, req.lo, req.hi),
      );
    case 'field':
      return JSON.parse(
        play_field(req.game, JSON.stringify(req.opts), req.a, req.b, req.seed, req.lo, req.hi),
      );
    case 'elo':
      return JSON.parse(elo(req.w, req.d, req.l));
    case 'fitElo':
      return JSON.parse(fit_elo_table(JSON.stringify(req.records)));
    case 'azNew':
      azBot?.free();
      azBot = new AzChessBot(req.sims, req.leaves, req.seed);
      if (req.weights) azBot.load_weights(new Uint8Array(req.weights));
      return null;
    case 'goNew':
      azBot?.free();
      azBot = new AzGoBot(req.sims, req.leaves, req.seed, req.size);
      if (req.weights) azBot.load_weights(new Uint8Array(req.weights));
      return null;
    case 'azPush': {
      if (!azBot) throw new Error('no az bot');
      azBot.push(req.uci);
      return null;
    }
    case 'azAdvance': {
      if (!azBot) throw new Error('no az bot');
      const n = azBot.advance(req.priors, req.values);
      return {
        n,
        features: n > 0 ? azBot.batch_features() : new Float32Array(0),
        support: n > 0 ? azBot.batch_support() : new Uint16Array(0),
        offsets: n > 0 ? azBot.batch_offsets() : new Uint32Array(0),
      };
    }
    case 'azPlayCpu': {
      if (!azBot) throw new Error('no az bot');
      return { uci: azBot.play_cpu(), stats: JSON.parse(azBot.stats()) };
    }
    case 'azBest': {
      if (!azBot) throw new Error('no az bot');
      return { uci: azBot.best(), stats: JSON.parse(azBot.stats()) };
    }
    case 'azFinalResult': {
      if (!azBot) throw new Error('no az bot');
      return azBot.final_result();
    }
    case 'snakeNew':
      snakeBot?.free();
      snakeBot = new AzSnakeBot(req.sims, req.leaves, req.seed);
      if (req.weights) snakeBot.load_weights(new Uint8Array(req.weights));
      return null;
    case 'snakePlayCpu': {
      if (!snakeBot) throw new Error('no snake bot');
      snakeBot.set_state(req.view);
      return { uci: snakeBot.play_cpu(), stats: JSON.parse(snakeBot.stats()) };
    }
    case 'snakeSetState': {
      if (!snakeBot) throw new Error('no snake bot');
      snakeBot.set_state(req.view);
      return null;
    }
    case 'snakeAdvance': {
      if (!snakeBot) throw new Error('no snake bot');
      const n = snakeBot.advance(req.priors, req.values);
      return {
        n,
        features: n > 0 ? snakeBot.batch_features() : new Float32Array(0),
        support: n > 0 ? snakeBot.batch_support() : new Uint16Array(0),
        offsets: n > 0 ? snakeBot.batch_offsets() : new Uint32Array(0),
      };
    }
    case 'snakeBest': {
      if (!snakeBot) throw new Error('no snake bot');
      return { uci: snakeBot.best(), stats: JSON.parse(snakeBot.stats()) };
    }
  }
}

self.onmessage = async (e: MessageEvent<EngineRequest>) => {
  await ready;
  const req = e.data;
  let resp: EngineResponse;
  try {
    resp = { id: req.id, ok: true, data: handle(req) };
  } catch (err) {
    resp = { id: req.id, ok: false, error: err instanceof Error ? err.message : String(err) };
  }
  (self as unknown as Worker).postMessage(resp);
};
