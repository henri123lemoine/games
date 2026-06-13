// All styling for the Liar's Dice table, injected once by mount(). Classes
// are prefixed `ld-`. The table is a self-contained night scene — felt,
// brass, and linen are hard-coded so it reads the same on the shell's light
// and dark pages; only the text fallback follows the shell tokens.

export const STYLE_ID = 'liars-dice-frontend-style';

export const STYLE = `
.ld-root {
  display: flex;
  flex-direction: column;
  gap: 12px;
  width: 100%;
  max-width: 920px;
  margin: 0 auto;
  user-select: none;
}

/* ---------- the table ---------- */

.ld-table {
  position: relative;
  width: 100%;
  height: clamp(360px, 56vh, 540px);
}

/* Layered felt: a lamp-light sheen, woven-grain noise, and the green pile
 * sit on the padding box; the mahogany rail is painted on the border box. */
.ld-felt {
  position: absolute;
  inset: 5% 2%;
  border-radius: 50% / 46%;
  border: 10px solid transparent;
  background:
    radial-gradient(ellipse 60% 44% at 50% 30%, rgba(255, 252, 230, 0.07), transparent 70%)
      padding-box,
    url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='140' height='140'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.85' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='140' height='140' filter='url(%23n)' opacity='0.05'/%3E%3C/svg%3E")
      padding-box,
    radial-gradient(ellipse at 50% 38%, #2e5e40 0%, #224830 55%, #142c1d 100%) padding-box,
    linear-gradient(155deg, #7a5530 0%, #46301b 40%, #5d3e22 70%, #33210f 100%) border-box;
  box-shadow:
    inset 0 0 80px rgba(0, 0, 0, 0.55),
    inset 0 3px 8px rgba(0, 0, 0, 0.5),
    0 0 0 1px rgba(0, 0, 0, 0.6),
    0 1px 0 rgba(255, 255, 255, 0.06),
    0 16px 40px rgba(0, 0, 0, 0.55);
}

.ld-felt::after {
  content: '';
  position: absolute;
  inset: 7%;
  border-radius: inherit;
  border: 1px solid rgba(212, 169, 92, 0.16);
}

/* ---------- center: bid, ladder, round ---------- */

.ld-center {
  position: absolute;
  left: 50%;
  top: 50%;
  transform: translate(-50%, -50%);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  text-align: center;
  z-index: 2;
  max-width: 46%;
  pointer-events: none;
}

.ld-round {
  font-family: var(--mono);
  font-size: 10.5px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: rgba(234, 230, 216, 0.55);
}

.ld-bid-box {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 2px;
  min-height: 60px;
}

.ld-bid-main {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 34px;
  font-weight: 700;
  color: #eae6d8;
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
}

.ld-bid-main .ld-die {
  width: 36px;
  height: 36px;
  rotate: -3deg;
}

.ld-x {
  color: #9da28e;
  font-size: 20px;
  font-weight: 600;
}

.ld-tally-n {
  color: #d4a95c;
  min-width: 1.1em;
  text-align: right;
}

.ld-bid-sub {
  font-size: 12px;
  color: rgba(234, 230, 216, 0.65);
}

.ld-open-hint {
  font-size: 14px;
  font-style: italic;
  color: rgba(234, 230, 216, 0.7);
}

.ld-win-text {
  font-size: 24px;
  font-weight: 700;
  color: #d4a95c;
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
}

.ld-ladder {
  display: flex;
  flex-direction: column;
  gap: 2px;
  align-items: center;
}

.ld-rung {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: rgba(234, 230, 216, 0.45);
}

.ld-rung .ld-die {
  width: 13px;
  height: 13px;
}

.ld-rung-now {
  color: #eae6d8;
  font-weight: 600;
}

/* ---------- seats ---------- */

.ld-seats {
  position: absolute;
  inset: 0;
}

.ld-seat {
  position: absolute;
  transform: translate(-50%, -50%);
  z-index: 3;
  transition: opacity 0.4s;
}

.ld-pod {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  background: rgba(10, 13, 9, 0.6);
  border: 1px solid #2d352c;
  border-radius: 14px;
  backdrop-filter: blur(3px);
  transition: box-shadow 0.3s, border-color 0.3s;
}

.ld-turn .ld-pod {
  border-color: #d4a95c;
  animation: ld-glow 1.6s ease-in-out infinite;
}

@keyframes ld-glow {
  0%, 100% { box-shadow: 0 0 0 1px #d4a95c, 0 0 12px rgba(212, 169, 92, 0.25); }
  50% { box-shadow: 0 0 0 1px #d4a95c, 0 0 24px rgba(212, 169, 92, 0.55); }
}

.ld-out {
  opacity: 0.35;
  filter: grayscale(0.9);
}

.ld-out-mark {
  font-size: 20px;
  line-height: 30px;
  color: #9da28e;
}

.ld-out-tag {
  font-size: 10px;
  letter-spacing: 1px;
  color: #d96a5a;
  font-weight: 700;
}

.ld-name {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 600;
  color: #eae6d8;
  white-space: nowrap;
}

.ld-tag {
  font-size: 10px;
  font-weight: 400;
  color: #9da28e;
}

.ld-crown {
  color: #d4a95c;
}

.ld-hand {
  display: flex;
  gap: 4px;
  align-items: center;
  justify-content: center;
  flex-wrap: wrap;
  max-width: 144px;
  min-height: 34px;
}

.ld-bubble {
  position: absolute;
  top: -13px;
  left: 50%;
  transform: translateX(-50%);
  display: flex;
  align-items: center;
  gap: 4px;
  background: #0b0e0a;
  border: 1px solid #d4a95c;
  border-radius: 999px;
  padding: 2px 9px;
  font-size: 11px;
  font-weight: 700;
  color: #eae6d8;
  white-space: nowrap;
  z-index: 2;
}

.ld-bubble .ld-die {
  width: 14px;
  height: 14px;
}

/* ---------- dice ---------- */

/* Pips are absolutely positioned with percentage offsets, which resolve
 * against the die's own box — the pattern stays correct at every size the
 * die is used (hand, bid, ladder, buttons, fly chip). */
.ld-die {
  position: relative;
  width: 24px;
  height: 24px;
  border-radius: 20%;
  background: linear-gradient(145deg, #f9f4e2 0%, #ece4ca 55%, #d6cbab 100%);
  box-shadow:
    inset 0 1px 1px rgba(255, 255, 255, 0.75),
    inset 0 -2px 3px rgba(94, 78, 48, 0.3),
    0 2px 4px rgba(0, 0, 0, 0.45);
  flex: none;
}

.ld-die i {
  position: absolute;
  width: 22%;
  height: 22%;
  border-radius: 50%;
  background: radial-gradient(circle at 36% 30%, #51463a, #221b12 75%);
  transform: translate(-50%, -50%);
}

.ld-pip-nw { left: 26%; top: 26%; }
.ld-pip-n  { left: 50%; top: 26%; }
.ld-pip-ne { left: 74%; top: 26%; }
.ld-pip-w  { left: 26%; top: 50%; }
.ld-pip-c  { left: 50%; top: 50%; }
.ld-pip-e  { left: 74%; top: 50%; }
.ld-pip-sw { left: 26%; top: 74%; }
.ld-pip-s  { left: 50%; top: 74%; }
.ld-pip-se { left: 74%; top: 74%; }

.ld-die-num {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 70%;
  font-weight: 800;
  color: #221b12;
}

/* A thrown hand, not a stamped row: each die settles at its own angle. */
.ld-hand .ld-die:nth-child(2n) { rotate: 2.5deg; }
.ld-hand .ld-die:nth-child(3n) { rotate: -2deg; }
.ld-hand .ld-die:nth-child(4n + 1) { rotate: -1.4deg; }
.ld-hand .ld-die:nth-child(5n + 2) { rotate: 1.8deg; }

.ld-die.ld-hit {
  background: linear-gradient(145deg, #ffedb9, #e9cb74);
  box-shadow: 0 0 0 2px #d4a95c, 0 0 10px rgba(212, 169, 92, 0.6);
}

.ld-flip {
  animation: ld-flip 0.35s ease;
}

@keyframes ld-flip {
  from { transform: rotateX(90deg) scale(0.6); opacity: 0; }
  to { transform: rotateX(0) scale(1); opacity: 1; }
}

/* ---------- cups ---------- */

.ld-cup {
  position: relative;
  width: 36px;
  height: 34px;
  flex: none;
}

.ld-cup::before {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(160deg, #9a6b3d, #5d3c20 70%);
  clip-path: polygon(15% 0, 85% 0, 100% 88%, 0 88%);
  border-radius: 3px;
}

.ld-cup::after {
  content: '';
  position: absolute;
  left: -4%;
  right: -4%;
  bottom: 0;
  height: 14%;
  background: #3f2814;
  border-radius: 3px;
}

.ld-cup-count {
  position: absolute;
  top: -7px;
  right: -9px;
  z-index: 1;
  min-width: 17px;
  height: 17px;
  border-radius: 999px;
  background: #0b0e0a;
  border: 1px solid #2d352c;
  color: #eae6d8;
  font-size: 10px;
  font-weight: 700;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 0 3px;
}

.ld-roll .ld-cup {
  animation: ld-shake 0.45s ease-in-out infinite;
}

@keyframes ld-shake {
  0%, 100% { transform: rotate(0); }
  25% { transform: rotate(-7deg) translateY(-2px); }
  75% { transform: rotate(7deg) translateY(-1px); }
}

/* ---------- reveal choreography ---------- */

.ld-lose .ld-pod {
  border-color: #d96a5a;
  animation: ld-lose-flash 0.4s ease 3;
}

@keyframes ld-lose-flash {
  0%, 100% { box-shadow: 0 0 0 1px #d96a5a; }
  50% {
    box-shadow: 0 0 0 3px #d96a5a, 0 0 26px rgba(217, 106, 90, 0.7);
    background: rgba(217, 106, 90, 0.18);
  }
}

.ld-safe .ld-pod {
  border-color: #8fae6e;
  box-shadow: 0 0 0 1px #8fae6e, 0 0 18px rgba(143, 174, 110, 0.5);
}

.ld-winner .ld-pod {
  border-color: #d4a95c;
  box-shadow: 0 0 0 1px #d4a95c, 0 0 26px rgba(212, 169, 92, 0.55);
}

.ld-float {
  position: absolute;
  left: 50%;
  top: -6px;
  color: #d96a5a;
  font-weight: 800;
  font-size: 15px;
  text-shadow: 0 1px 4px rgba(0, 0, 0, 0.8);
  animation: ld-float 1s ease-out forwards;
  pointer-events: none;
  z-index: 4;
  white-space: nowrap;
}

@keyframes ld-float {
  from { opacity: 1; transform: translate(-50%, 0); }
  to { opacity: 0; transform: translate(-50%, -28px); }
}

.ld-banner {
  position: absolute;
  left: 50%;
  top: 13%;
  transform: translate(-50%, -50%) scale(0.85);
  z-index: 6;
  background: #0b0e0a;
  border: 1px solid #2d352c;
  border-radius: 999px;
  padding: 9px 22px;
  font-size: 15px;
  font-weight: 800;
  letter-spacing: 0.4px;
  white-space: nowrap;
  opacity: 0;
  transition: opacity 0.22s ease, transform 0.22s ease;
  pointer-events: none;
  box-shadow: 0 6px 24px rgba(0, 0, 0, 0.5);
  color: #eae6d8;
}

.ld-banner.ld-show {
  opacity: 1;
  transform: translate(-50%, -50%) scale(1);
}

.ld-banner-liar { color: #d96a5a; border-color: #d96a5a; }
.ld-banner-exact { color: #8fae6e; border-color: #8fae6e; }
.ld-banner-good { color: #8fae6e; border-color: #8fae6e; }

.ld-fly {
  position: absolute;
  z-index: 5;
  transform: translate(-50%, -50%);
  display: flex;
  align-items: center;
  gap: 6px;
  background: #0b0e0a;
  border: 1px solid #d4a95c;
  border-radius: 999px;
  padding: 4px 12px;
  font-weight: 800;
  font-size: 14px;
  color: #eae6d8;
  pointer-events: none;
}

.ld-fly .ld-die {
  width: 18px;
  height: 18px;
}

/* ---------- controls ---------- */

/* The player's rail: a leather strip below the felt that the bid controls
 * sit on, so they read as part of the table rather than a floating toolbar. */
.ld-controls {
  display: flex;
  gap: 10px;
  justify-content: center;
  align-items: center;
  flex-wrap: wrap;
  min-height: 64px;
  padding: 9px 14px;
  background: linear-gradient(180deg, rgba(40, 56, 38, 0.5), rgba(21, 31, 20, 0.5));
  border: 1px solid rgba(212, 169, 92, 0.14);
  border-radius: 16px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.04), inset 0 0 24px rgba(0, 0, 0, 0.25);
}

.ld-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 18px;
  background: linear-gradient(180deg, #233527 0%, #162417 100%);
  border: 1px solid #3a4a38;
  border-radius: var(--radius);
  color: #eae6d8;
  font: inherit;
  font-weight: 700;
  cursor: pointer;
  transition: border-color 0.15s, transform 0.15s, box-shadow 0.15s;
}

.ld-btn:hover:not(:disabled) {
  border-color: #d4a95c;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}

.ld-btn:disabled {
  opacity: 0.45;
  cursor: default;
}

.ld-btn .ld-die {
  width: 20px;
  height: 20px;
}

.ld-btn-liar {
  color: #d96a5a;
  border-color: rgba(217, 106, 90, 0.55);
  letter-spacing: 1px;
}

.ld-btn-liar:hover:not(:disabled) {
  border-color: #d96a5a;
  background: rgba(217, 106, 90, 0.12);
  box-shadow: 0 0 14px rgba(217, 106, 90, 0.35);
}

.ld-btn-exact {
  color: #8fae6e;
  border-color: rgba(143, 174, 110, 0.55);
  letter-spacing: 1px;
}

.ld-btn-exact:hover:not(:disabled) {
  border-color: #8fae6e;
  background: rgba(143, 174, 110, 0.12);
  box-shadow: 0 0 14px rgba(143, 174, 110, 0.3);
}

.ld-open {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  justify-content: center;
}

.ld-open-label {
  font-size: 11px;
  color: #9da28e;
  text-transform: uppercase;
  letter-spacing: 1px;
}

.ld-qty {
  display: flex;
  align-items: center;
  gap: 8px;
}

.ld-qty-n {
  font-size: 23px;
  font-weight: 700;
  min-width: 2ch;
  text-align: center;
}

.ld-step {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  border: 1px solid #2d352c;
  background: #1a211a;
  color: #eae6d8;
  font-size: 16px;
  font-weight: 700;
  cursor: pointer;
  line-height: 1;
}

.ld-step:hover:not(:disabled) {
  border-color: #d4a95c;
}

.ld-step:disabled {
  opacity: 0.4;
  cursor: default;
}

.ld-faces {
  display: flex;
  gap: 6px;
}

.ld-face-btn {
  padding: 3px;
  border-radius: 8px;
  border: 2px solid transparent;
  background: none;
  cursor: pointer;
  display: flex;
}

.ld-face-btn:hover {
  border-color: #2d352c;
}

.ld-face-btn.ld-sel {
  border-color: #d4a95c;
  box-shadow: 0 0 10px rgba(212, 169, 92, 0.35);
}

.ld-fallback {
  font-family: ui-monospace, monospace;
  white-space: pre-wrap;
  color: var(--text);
  padding: 12px;
}

@media (prefers-reduced-motion: reduce) {
  .ld-turn .ld-pod,
  .ld-roll .ld-cup,
  .ld-flip,
  .ld-lose .ld-pod {
    animation: none;
  }
}
`;
