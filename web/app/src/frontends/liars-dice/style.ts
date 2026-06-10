// All styling for the Liar's Dice table, injected once by mount(). Classes
// are prefixed `ld-`; colors come from the shell's theme tokens, with the
// felt and the dice cups as this game's own identity.

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

.ld-felt {
  position: absolute;
  inset: 5% 2%;
  border-radius: 50% / 46%;
  background: radial-gradient(ellipse at 50% 38%, #2c6a44 0%, #1e4c30 55%, #133221 100%);
  border: 9px solid #4a3120;
  box-shadow:
    inset 0 0 70px rgba(0, 0, 0, 0.55),
    0 0 0 2px #2a1c12,
    0 12px 32px rgba(0, 0, 0, 0.5);
}

.ld-felt::after {
  content: '';
  position: absolute;
  inset: 8%;
  border-radius: inherit;
  border: 2px dashed rgba(255, 255, 255, 0.07);
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
  font-size: 11px;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: rgba(230, 237, 243, 0.55);
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
  font-size: 32px;
  font-weight: 800;
  color: var(--text);
  text-shadow: 0 2px 8px rgba(0, 0, 0, 0.6);
}

.ld-bid-main .ld-die {
  width: 36px;
  height: 36px;
}

.ld-x {
  color: var(--text-dim);
  font-size: 20px;
  font-weight: 600;
}

.ld-tally-n {
  color: var(--accent);
  min-width: 1.1em;
  text-align: right;
}

.ld-bid-sub {
  font-size: 12px;
  color: rgba(230, 237, 243, 0.65);
}

.ld-open-hint {
  font-size: 14px;
  font-style: italic;
  color: rgba(230, 237, 243, 0.7);
}

.ld-win-text {
  font-size: 24px;
  font-weight: 800;
  color: #e3b341;
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
  color: rgba(230, 237, 243, 0.45);
}

.ld-rung .ld-die {
  width: 13px;
  height: 13px;
}

.ld-rung-now {
  color: var(--text);
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
  background: rgba(1, 4, 9, 0.55);
  border: 1px solid var(--border);
  border-radius: 14px;
  backdrop-filter: blur(3px);
  transition: box-shadow 0.3s, border-color 0.3s;
}

.ld-turn .ld-pod {
  border-color: var(--accent);
  animation: ld-glow 1.6s ease-in-out infinite;
}

@keyframes ld-glow {
  0%, 100% { box-shadow: 0 0 0 1px var(--accent), 0 0 12px rgba(88, 166, 255, 0.25); }
  50% { box-shadow: 0 0 0 1px var(--accent), 0 0 24px rgba(88, 166, 255, 0.55); }
}

.ld-out {
  opacity: 0.35;
  filter: grayscale(0.9);
}

.ld-out-mark {
  font-size: 20px;
  line-height: 30px;
  color: var(--text-dim);
}

.ld-out-tag {
  font-size: 10px;
  letter-spacing: 1px;
  color: var(--bad);
  font-weight: 700;
}

.ld-name {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  font-weight: 600;
  color: var(--text);
  white-space: nowrap;
}

.ld-tag {
  font-size: 10px;
  font-weight: 400;
  color: var(--text-dim);
}

.ld-crown {
  color: #e3b341;
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
  background: var(--bg-inset);
  border: 1px solid var(--accent);
  border-radius: 999px;
  padding: 2px 9px;
  font-size: 11px;
  font-weight: 700;
  color: var(--text);
  white-space: nowrap;
  z-index: 2;
}

.ld-bubble .ld-die {
  width: 14px;
  height: 14px;
}

/* ---------- dice ---------- */

.ld-die {
  width: 24px;
  height: 24px;
  border-radius: 18%;
  background: linear-gradient(145deg, #fdfdf4, #d9d9cb);
  box-shadow: inset 0 -2px 3px rgba(0, 0, 0, 0.18), 0 1px 3px rgba(0, 0, 0, 0.5);
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  grid-template-rows: repeat(3, 1fr);
  padding: 12%;
  box-sizing: border-box;
  flex: none;
}

.ld-die i {
  width: 82%;
  height: 82%;
  place-self: center;
  border-radius: 50%;
  background: #202028;
  visibility: hidden;
}

.ld-die[data-v='1'] i:nth-child(5),
.ld-die[data-v='2'] i:nth-child(3),
.ld-die[data-v='2'] i:nth-child(7),
.ld-die[data-v='3'] i:nth-child(3),
.ld-die[data-v='3'] i:nth-child(5),
.ld-die[data-v='3'] i:nth-child(7),
.ld-die[data-v='4'] i:nth-child(1),
.ld-die[data-v='4'] i:nth-child(3),
.ld-die[data-v='4'] i:nth-child(7),
.ld-die[data-v='4'] i:nth-child(9),
.ld-die[data-v='5'] i:nth-child(1),
.ld-die[data-v='5'] i:nth-child(3),
.ld-die[data-v='5'] i:nth-child(5),
.ld-die[data-v='5'] i:nth-child(7),
.ld-die[data-v='5'] i:nth-child(9),
.ld-die[data-v='6'] i:nth-child(1),
.ld-die[data-v='6'] i:nth-child(3),
.ld-die[data-v='6'] i:nth-child(4),
.ld-die[data-v='6'] i:nth-child(6),
.ld-die[data-v='6'] i:nth-child(7),
.ld-die[data-v='6'] i:nth-child(9) {
  visibility: visible;
}

.ld-die.ld-hit {
  background: linear-gradient(145deg, #ffeaae, #e9ca6c);
  box-shadow: 0 0 0 2px var(--accent), 0 0 10px rgba(88, 166, 255, 0.6);
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
  background: var(--bg-inset);
  border: 1px solid var(--border);
  color: var(--text);
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
  border-color: var(--bad);
  animation: ld-lose-flash 0.4s ease 3;
}

@keyframes ld-lose-flash {
  0%, 100% { box-shadow: 0 0 0 1px var(--bad); }
  50% {
    box-shadow: 0 0 0 3px var(--bad), 0 0 26px rgba(248, 81, 73, 0.7);
    background: rgba(248, 81, 73, 0.18);
  }
}

.ld-safe .ld-pod {
  border-color: var(--good);
  box-shadow: 0 0 0 1px var(--good), 0 0 18px rgba(63, 185, 80, 0.5);
}

.ld-winner .ld-pod {
  border-color: #e3b341;
  box-shadow: 0 0 0 1px #e3b341, 0 0 26px rgba(227, 179, 65, 0.55);
}

.ld-float {
  position: absolute;
  left: 50%;
  top: -6px;
  color: var(--bad);
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
  background: var(--bg-inset);
  border: 1px solid var(--border);
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
  color: var(--text);
}

.ld-banner.ld-show {
  opacity: 1;
  transform: translate(-50%, -50%) scale(1);
}

.ld-banner-liar { color: var(--bad); border-color: var(--bad); }
.ld-banner-exact { color: var(--accent-2); border-color: var(--accent-2); }
.ld-banner-good { color: var(--good); border-color: var(--good); }

.ld-fly {
  position: absolute;
  z-index: 5;
  transform: translate(-50%, -50%);
  display: flex;
  align-items: center;
  gap: 6px;
  background: var(--bg-inset);
  border: 1px solid var(--accent);
  border-radius: 999px;
  padding: 4px 12px;
  font-weight: 800;
  font-size: 14px;
  color: var(--text);
  pointer-events: none;
}

.ld-fly .ld-die {
  width: 18px;
  height: 18px;
}

/* ---------- controls ---------- */

.ld-controls {
  display: flex;
  gap: 10px;
  justify-content: center;
  align-items: center;
  flex-wrap: wrap;
  min-height: 64px;
}

.ld-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 10px 18px;
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text);
  font: inherit;
  font-weight: 700;
  cursor: pointer;
  transition: border-color 0.15s, transform 0.15s, box-shadow 0.15s;
}

.ld-btn:hover:not(:disabled) {
  border-color: var(--accent);
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
  color: var(--bad);
  border-color: rgba(248, 81, 73, 0.55);
  letter-spacing: 1px;
}

.ld-btn-liar:hover:not(:disabled) {
  border-color: var(--bad);
  background: rgba(248, 81, 73, 0.12);
  box-shadow: 0 0 14px rgba(248, 81, 73, 0.35);
}

.ld-btn-exact {
  color: var(--accent-2);
  border-color: rgba(188, 140, 255, 0.55);
  letter-spacing: 1px;
}

.ld-btn-exact:hover:not(:disabled) {
  border-color: var(--accent-2);
  background: rgba(188, 140, 255, 0.12);
  box-shadow: 0 0 14px rgba(188, 140, 255, 0.3);
}

.ld-open {
  display: flex;
  align-items: center;
  gap: 16px;
  flex-wrap: wrap;
  justify-content: center;
  padding: 10px 16px;
  background: var(--bg-inset);
  border: 1px solid var(--border);
  border-radius: var(--radius);
}

.ld-open-label {
  font-size: 11px;
  color: var(--text-dim);
  text-transform: uppercase;
  letter-spacing: 1px;
}

.ld-qty {
  display: flex;
  align-items: center;
  gap: 8px;
}

.ld-qty-n {
  font-size: 22px;
  font-weight: 800;
  min-width: 2ch;
  text-align: center;
}

.ld-step {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  border: 1px solid var(--border);
  background: var(--bg-raised);
  color: var(--text);
  font-size: 16px;
  font-weight: 700;
  cursor: pointer;
  line-height: 1;
}

.ld-step:hover:not(:disabled) {
  border-color: var(--accent);
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
  border-color: var(--border);
}

.ld-face-btn.ld-sel {
  border-color: var(--accent);
  box-shadow: 0 0 10px rgba(88, 166, 255, 0.35);
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
