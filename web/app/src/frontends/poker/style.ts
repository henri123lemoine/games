// All styling for the No-Limit Hold'em table, injected once by mount().
// Classes are prefixed `pk-`. A self-contained casino scene — green felt,
// brass rail, playing cards — hard-coded so it reads the same on the shell's
// light and dark pages.

export const STYLE_ID = 'poker-frontend-style';

export const STYLE = `
.pk-root {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
  max-width: 940px;
  margin: 0 auto;
  user-select: none;
  --card-w: clamp(30px, 4.2vw, 44px);
}

/* ---------- the table ---------- */

.pk-table {
  position: relative;
  width: 100%;
  height: clamp(380px, 58vh, 560px);
}

.pk-felt {
  position: absolute;
  inset: 6% 2%;
  border-radius: 46% / 50%;
  border: 11px solid transparent;
  background:
    radial-gradient(ellipse 58% 42% at 50% 32%, rgba(255, 252, 230, 0.08), transparent 70%)
      padding-box,
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)' opacity='0.05'/%3E%3C/svg%3E")
      padding-box,
    radial-gradient(ellipse at 50% 40%, #2f6043 0%, #224a31 55%, #122a1c 100%) padding-box,
    linear-gradient(155deg, #6f4c2c 0%, #3f2b18 42%, #543820 72%, #2c1c0d 100%) border-box;
  box-shadow:
    inset 0 0 90px rgba(0, 0, 0, 0.55),
    inset 0 3px 8px rgba(0, 0, 0, 0.5),
    0 0 0 1px rgba(0, 0, 0, 0.6),
    0 1px 0 rgba(255, 255, 255, 0.06),
    0 18px 44px rgba(0, 0, 0, 0.55);
}

.pk-felt::after {
  content: '';
  position: absolute;
  inset: 6%;
  border-radius: inherit;
  border: 1px solid rgba(212, 169, 92, 0.18);
}

/* ---------- center: board + pot ---------- */

.pk-center {
  position: absolute;
  left: 50%;
  top: 42%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  width: 64%;
  pointer-events: none;
}

.pk-board {
  display: flex;
  gap: 6px;
  min-height: calc(var(--card-w) * 1.4);
  align-items: center;
}

.pk-pot {
  font: 600 13px/1 ui-monospace, 'SF Mono', Menlo, monospace;
  color: #f4e3b8;
  letter-spacing: 0.04em;
  background: rgba(8, 22, 14, 0.62);
  border: 1px solid rgba(212, 169, 92, 0.4);
  border-radius: 999px;
  padding: 5px 14px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.45);
  white-space: nowrap;
}
.pk-pot b { color: #fff; }
.pk-street {
  font: 600 10px/1 system-ui, sans-serif;
  text-transform: uppercase;
  letter-spacing: 0.18em;
  color: rgba(244, 227, 184, 0.65);
}

/* ---------- cards ---------- */

.pk-card {
  width: var(--card-w);
  height: calc(var(--card-w) * 1.4);
  border-radius: 5px;
  background: linear-gradient(160deg, #fff 0%, #f1f1ec 100%);
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.45), inset 0 0 0 1px rgba(0, 0, 0, 0.08);
  position: relative;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  padding: 3px 4px;
  font: 700 calc(var(--card-w) * 0.42) / 1 'Georgia', serif;
  color: #1b1b1b;
}
.pk-card.red { color: #c01f2e; }
.pk-card .pk-rank { line-height: 0.9; }
.pk-card .pk-suit { font-size: calc(var(--card-w) * 0.5); align-self: flex-end; line-height: 0.8; }
.pk-card.deal-in { animation: pk-deal 0.32s ease-out backwards; }
@keyframes pk-deal {
  from { opacity: 0; transform: translateY(-18px) rotate(-6deg) scale(0.9); }
  to { opacity: 1; transform: none; }
}

.pk-card.back {
  background:
    repeating-linear-gradient(45deg, #243f7a 0 6px, #1d3260 6px 12px),
    #1d3260;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.45), inset 0 0 0 2px rgba(255, 255, 255, 0.14);
}
.pk-card.muck { opacity: 0.32; filter: grayscale(0.6); }
.pk-card.win-card { box-shadow: 0 0 0 2px #f4d35e, 0 3px 12px rgba(244, 211, 94, 0.6); }

/* ---------- seats ---------- */

.pk-seats { position: absolute; inset: 0; }

.pk-seat {
  position: absolute;
  transform: translate(-50%, -50%);
  width: clamp(118px, 16vw, 150px);
}
.pk-pod {
  position: relative;
  background: linear-gradient(180deg, rgba(20, 28, 22, 0.92), rgba(10, 16, 12, 0.92));
  border: 1px solid rgba(212, 169, 92, 0.28);
  border-radius: 12px;
  padding: 7px 8px 6px;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.5);
  text-align: center;
  transition: border-color 0.2s, box-shadow 0.2s;
}
.pk-seat.turn .pk-pod {
  border-color: #f4d35e;
  box-shadow: 0 0 0 1px #f4d35e, 0 0 18px rgba(244, 211, 94, 0.4), 0 6px 18px rgba(0, 0, 0, 0.5);
}
.pk-seat.folded .pk-pod { opacity: 0.46; }
.pk-seat.winner .pk-pod {
  border-color: #66e08a;
  box-shadow: 0 0 0 1px #66e08a, 0 0 22px rgba(102, 224, 138, 0.5);
}

.pk-holes { display: flex; gap: 3px; justify-content: center; margin-bottom: 4px; min-height: calc(var(--card-w) * 1.4); }
.pk-seat .pk-card { --card-w: clamp(26px, 3.4vw, 36px); }

.pk-name {
  font: 600 12px/1.2 system-ui, sans-serif;
  color: #f1e6cb;
  display: flex; align-items: center; justify-content: center; gap: 4px;
}
.pk-stack {
  font: 600 11px/1.3 ui-monospace, Menlo, monospace;
  color: #bfe9cf;
}
.pk-stack .pk-bust { color: #e88; }
.pk-badge {
  display: inline-block;
  font: 700 8px/1 system-ui;
  background: #d4a95c; color: #201400;
  border-radius: 3px; padding: 2px 3px; margin-left: 2px;
  vertical-align: middle;
}
.pk-tag {
  position: absolute; top: -8px; right: -6px;
  font: 700 8px/1 system-ui; letter-spacing: 0.06em;
  padding: 2px 5px; border-radius: 999px;
}
.pk-tag.allin { background: #c0392b; color: #fff; }
.pk-tag.folded { background: #555; color: #ddd; }

/* a seat's current bet chips, pushed toward the pot */
.pk-bet {
  position: absolute;
  left: 50%; transform: translateX(-50%);
  font: 700 10px/1 ui-monospace, Menlo, monospace;
  color: #1a1208;
  background: #f4d35e;
  border: 1px solid #b8901f;
  border-radius: 999px;
  padding: 2px 7px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.4);
  white-space: nowrap;
}
.pk-seat.below .pk-bet { top: -16px; }
.pk-seat:not(.below) .pk-bet { bottom: -16px; }

.pk-dealer {
  position: absolute;
  width: 18px; height: 18px; border-radius: 50%;
  background: radial-gradient(circle at 35% 30%, #fff, #d8d2c4 70%, #b7b1a3);
  color: #222; font: 800 9px/18px system-ui; text-align: center;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.5);
  border: 1px solid rgba(0,0,0,0.25);
}

/* float a chip delta over a seat */
.pk-float {
  position: absolute; left: 50%; top: 50%;
  transform: translate(-50%, -50%);
  font: 800 14px/1 ui-monospace, Menlo, monospace;
  pointer-events: none;
  animation: pk-rise 1.1s ease-out forwards;
}
.pk-float.win { color: #66e08a; text-shadow: 0 1px 4px rgba(0,0,0,0.7); }
.pk-float.lose { color: #ff7a7a; text-shadow: 0 1px 4px rgba(0,0,0,0.7); }
@keyframes pk-rise {
  0% { opacity: 0; transform: translate(-50%, -30%); }
  20% { opacity: 1; }
  100% { opacity: 0; transform: translate(-50%, -130%); }
}

/* ---------- banner ---------- */

.pk-banner {
  position: absolute; left: 50%; top: 14%;
  transform: translate(-50%, -50%) scale(0.9);
  background: rgba(8, 16, 11, 0.92);
  border: 1px solid rgba(212, 169, 92, 0.5);
  border-radius: 10px;
  padding: 8px 18px;
  font: 700 14px/1.2 system-ui, sans-serif;
  color: #f6eccf; text-align: center;
  opacity: 0; pointer-events: none;
  transition: opacity 0.22s, transform 0.22s;
  z-index: 6; max-width: 70%;
}
.pk-banner.show { opacity: 1; transform: translate(-50%, -50%) scale(1); }
.pk-banner.good { border-color: #66e08a; color: #d6ffe3; }
.pk-banner.bad { border-color: #e0664f; color: #ffdcd2; }

/* ---------- controls ---------- */

.pk-controls {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
  justify-content: center;
  min-height: 46px;
}
.pk-btn {
  font: 600 14px/1 system-ui, sans-serif;
  padding: 11px 18px;
  border-radius: 9px;
  border: 1px solid rgba(0, 0, 0, 0.25);
  background: linear-gradient(180deg, #f4f4f0, #e1e1d8);
  color: #1b1b1b;
  cursor: pointer;
  transition: transform 0.08s, filter 0.15s;
}
.pk-btn:hover:not(:disabled) { filter: brightness(1.05); }
.pk-btn:active:not(:disabled) { transform: translateY(1px); }
.pk-btn:disabled { opacity: 0.5; cursor: default; }
.pk-btn.fold { background: linear-gradient(180deg, #e9b0a6, #d98e80); color: #3a120a; }
.pk-btn.call { background: linear-gradient(180deg, #a9d9b6, #7cc18f); color: #0c2c16; }
.pk-btn.raise { background: linear-gradient(180deg, #f6dd8a, #e8c45a); color: #3a2a06; }

.pk-raiser {
  display: flex; align-items: center; gap: 8px;
  background: rgba(8, 16, 11, 0.06);
  border: 1px solid rgba(0,0,0,0.12);
  border-radius: 10px; padding: 6px 10px;
}
.pk-raiser input[type=range] { width: clamp(90px, 18vw, 180px); accent-color: #c79a3a; }
.pk-raiser .pk-amt {
  font: 700 13px/1 ui-monospace, Menlo, monospace;
  min-width: 46px; text-align: right; color: inherit;
}
.pk-quick { display: flex; gap: 4px; }
.pk-quick button {
  font: 600 11px/1 system-ui; padding: 5px 7px; border-radius: 6px;
  border: 1px solid rgba(0,0,0,0.18); background: #efe9da; cursor: pointer; color: #1b1b1b;
}
.pk-quick button:hover { background: #f7f1e2; }

.pk-fallback {
  white-space: pre-wrap;
  font: 13px/1.5 ui-monospace, Menlo, monospace;
  color: var(--fg, #222);
  padding: 12px;
}
`;
