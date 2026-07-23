#!/usr/bin/env python3
"""Browser smoke for the strategic Doom arena, collision, and tactical FFA."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:5173/doom-ai/index.html")
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()

    output = Path(__file__).resolve().parents[1] / ".validation" / "doom-strategic.png"
    menu_output = output.with_name("doom-strategic-menu.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    browser_logs: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        page = browser.new_page(viewport={"width": 1120, "height": 720})
        page.on("console", lambda msg: browser_logs.append(f"[{msg.type}] {msg.text}"))
        page.on("pageerror", lambda error: browser_logs.append(f"[pageerror] {error}"))

        samples = []
        controls = None
        try:
            url = f"{args.url}?botboth&autostart&speed=8&bots=3&difficulty=relentless"
            page.goto(url, wait_until="networkidle")
            page.wait_for_function(
                "window.__doomStrategic && window.__doomStrategic.running()",
                timeout=30_000,
            )
            render = page.evaluate(
                """() => {
                  const api = window.__doomStrategic;
                  const canvas = document.getElementById('canvas');
                  const rect = canvas.getBoundingClientRect();
                  return {
                    engine: [api.screen.width, api.screen.height],
                    backing: [canvas.width, canvas.height],
                    cssAspect: rect.width / rect.height,
                    imageRendering: getComputedStyle(canvas).imageRendering,
                  };
                }"""
            )
            print("render:", json.dumps(render, separators=(",", ":")))

            deadline = time.monotonic() + 35
            while time.monotonic() < deadline:
                sample = page.evaluate(
                    """() => {
                  const api = window.__doomStrategic;
                  const playerCount = api.playerCount();
                  const states = Array.from({length: playerCount}, (_, seat) => Array.from(api.readState(seat)));
                  const outline = [[-1120,768],[1120,768],[1280,608],[1280,-608],[1120,-768],[-1120,-768],[-1280,-608],[-1280,608]];
                  const inside = ([x,y]) => {
                    let result = false;
                    for (let i = 0, j = outline.length - 1; i < outline.length; j = i++) {
                      const [xi,yi] = outline[i], [xj,yj] = outline[j];
                      if ((yi > y) !== (yj > y) && x < (xj-xi)*(y-yi)/(yj-yi)+xi) result = !result;
                    }
                    return result;
                  };
                  const canvas = document.getElementById('canvas');
                  const pixels = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
                  let lit = 0, hash = 0;
                  for (let i = 0; i < pixels.length; i += 388) {
                    const value = pixels[i] + pixels[i + 1] + pixels[i + 2];
                    if (value > 24) lit += 1;
                    hash = (hash + value * (i + 1)) % 2147483647;
                  }
                  const quarterLit = [0, 0, 0, 0];
                  const quarterSamples = [0, 0, 0, 0];
                  const viewHeight = Math.floor(canvas.height * 0.84);
                  for (let y = 0; y < viewHeight; y += 4) {
                    for (let x = 0; x < canvas.width; x += 4) {
                      const q = Math.min(3, Math.floor(x * 4 / canvas.width));
                      const i = (y * canvas.width + x) * 4;
                      quarterSamples[q] += 1;
                      if (pixels[i] + pixels[i + 1] + pixels[i + 2] > 24) quarterLit[q] += 1;
                    }
                  }
                  return {
                    tics: api.metrics.tics,
                    playerCount,
                    frags: states.map((s) => s[15]), deaths: states.map((s) => s[16]),
                    hp: states.map((s) => s[7]), visible: states.map((s) => s[17]),
                    positions: states.map((s) => [s[1], s[2], s[3]]),
                    inArena: states.map((s) => inside([s[1], s[2]])),
                    wallBreaches: Number(canvas.dataset.wallBreaches || 0),
                    wallBreachDetails: JSON.parse(canvas.dataset.wallBreachDetails || "[]"),
                    items: [states[0][27], states[0][31], states[0][35]],
                    litFraction: lit / Math.ceil(pixels.length / 388), hash,
                    quarterLit: quarterLit.map((n, q) => n / quarterSamples[q]),
                  };
                }"""
                )
                samples.append(sample)
                print(json.dumps(sample, separators=(",", ":")))
                if sum(sample["frags"]) >= 4 and sum(frag > 0 for frag in sample["frags"]) >= 2:
                    break
                page.wait_for_timeout(500)

            control_page = browser.new_page(viewport={"width": 1120, "height": 720})
            control_page.goto(args.url, wait_until="networkidle")
            control_page.wait_for_function(
                "document.querySelector('#start') && !document.querySelector('#start').disabled",
                timeout=30_000,
            )
            control_page.screenshot(path=str(menu_output))
            control_page.click("#start")
            control_page.wait_for_function(
                "window.__doomStrategic && window.__doomStrategic.running()",
                timeout=5_000,
            )
            before = control_page.evaluate("Array.from(window.__doomStrategic.readState(0))")
            control_page.keyboard.down("KeyW")
            control_page.wait_for_timeout(700)
            control_page.keyboard.up("KeyW")
            moved = control_page.evaluate("Array.from(window.__doomStrategic.readState(0))")
            control_page.keyboard.down("ArrowRight")
            control_page.wait_for_timeout(350)
            control_page.keyboard.up("ArrowRight")
            turned = control_page.evaluate("Array.from(window.__doomStrategic.readState(0))")
            controls = {
                "travel": ((moved[1] - before[1]) ** 2 + (moved[2] - before[2]) ** 2) ** 0.5,
                "angle_change": abs(turned[4] - moved[4]),
            }
            print("controls:", json.dumps(controls, separators=(",", ":")))
            control_page.close()
        finally:
            page.screenshot(path=str(output))
            if browser_logs:
                print("browser logs:")
                print("\n".join(browser_logs[-20:]))
            status = page.locator("#status")
            print("status:", status.text_content(timeout=1_000) if status.count() else "missing")
            browser.close()

    errors = [line for line in browser_logs if line.startswith("[pageerror]")]
    if errors:
        raise AssertionError(f"browser errors: {errors}")
    if not samples:
        raise AssertionError("no gameplay samples")
    if samples[-1]["playerCount"] != 4:
        raise AssertionError(f"requested four-player FFA did not start: {samples[-1]}")
    if render["backing"] != render["engine"]:
        raise AssertionError(f"canvas/engine framebuffer mismatch: {render}")
    expected_aspect = render["engine"][0] / render["engine"][1]
    if abs(render["cssAspect"] - expected_aspect) > 0.01:
        raise AssertionError(f"canvas aspect ratio is distorted: {render}")
    if render["imageRendering"] not in {"pixelated", "crisp-edges"}:
        raise AssertionError(f"canvas is blur-filtered: {render}")
    if controls is None or controls["travel"] < 5 or controls["angle_change"] < 2:
        raise AssertionError(f"human controls did not move and turn the player: {controls}")

    first, last = samples[0], samples[-1]
    if last["tics"] - first["tics"] < 350:
        raise AssertionError("engine did not advance enough tics")
    if max(sample["litFraction"] for sample in samples) < 0.3:
        raise AssertionError("canvas remained mostly black")
    complete_views = sum(min(sample["quarterLit"]) > 0.25 for sample in samples)
    if complete_views < max(2, len(samples) // 2):
        raise AssertionError("rendering contains persistent black/HOM screen regions")
    if len({sample["hash"] for sample in samples}) < 4:
        raise AssertionError("rendered frame did not animate")
    if not all(all(sample["inArena"]) for sample in samples):
        raise AssertionError("a player escaped or spawned beyond the arena boundary")
    if any(sample["wallBreaches"] for sample in samples):
        failures = next(sample for sample in samples if sample["wallBreaches"])
        raise AssertionError(f"a player crossed a solid wall: {failures['wallBreachDetails']}")
    if not any(any(v > 0.5 for v in sample["visible"]) for sample in samples):
        raise AssertionError("bots never found line of sight")
    if sum(last["frags"]) < 1 and sum(last["deaths"]) < 1:
        raise AssertionError("no completed combat")

    travel = 0.0
    for seat in range(last["playerCount"]):
        xs = [sample["positions"][seat][0] for sample in samples]
        ys = [sample["positions"][seat][1] for sample in samples]
        travel = max(travel, max(xs) - min(xs), max(ys) - min(ys))
    if travel < 300:
        raise AssertionError("opponents did not navigate the arena")
    if max(position[2] for sample in samples for position in sample["positions"]) < 32:
        raise AssertionError("no opponent could climb the megaarmor tiers")

    print(f"PASS strategic Doom: tics={last['tics']} frags={last['frags']} screenshot={output}")


if __name__ == "__main__":
    main()
