"""Browser smoke for the four-player chess board and inference surfaces."""

import json
from pathlib import Path

from playwright.sync_api import sync_playwright


BASE = "http://127.0.0.1:5173"
OUT = Path(__file__).resolve().parent.parent / ".validation"
OUT.mkdir(exist_ok=True)


def browser_page(browser, width: int, height: int):
    context = browser.new_context(viewport={"width": width, "height": height})
    page = context.new_page()
    errors: list[str] = []
    failed_responses: list[str] = []
    page.on("pageerror", lambda error: errors.append(f"page: {error}"))
    page.on(
        "console",
        lambda message: errors.append(f"console: {message.text}")
        if message.type == "error"
        else None,
    )
    page.on(
        "response",
        lambda response: failed_responses.append(f"{response.status} {response.url}")
        if response.status >= 400
        else None,
    )
    return context, page, errors, failed_responses


with sync_playwright() as playwright:
    browser = playwright.chromium.launch(
        channel="chrome",
        headless=True,
        args=[
            "--enable-unsafe-webgpu",
            "--enable-features=Vulkan",
            "--use-angle=metal",
            "--ignore-gpu-blocklist",
        ],
    )
    results: dict[str, object] = {}

    desktop, page, errors, failed = browser_page(browser, 1440, 1050)
    page.goto(f"{BASE}/#/g/four-player-chess")
    page.wait_for_load_state("networkidle")
    page.locator(".fpc-root").wait_for(timeout=30_000)
    page.locator(".fpc-origin").first.wait_for(timeout=90_000)
    assert page.locator(".fpc-square").count() == 160
    assert page.locator(".fpc-piece").count() == 64
    assert page.locator(".fpc-rail").count() == 4
    assert page.locator(".fpc-rail .seat-select").count() == 4
    assert page.locator(".fpc-you").count() == 1

    page.locator(".fpc-origin").first.click()
    target = page.locator(".fpc-target, .fpc-capture").first
    target.wait_for(timeout=5_000)
    target.click()
    page.locator(".fpc-last-to").wait_for(timeout=15_000)
    page.screenshot(path=OUT / "four-player-chess-desktop.png", full_page=True)
    results["desktop"] = {
        "squares": page.locator(".fpc-square").count(),
        "pieces_after_move": page.locator(".fpc-piece").count(),
        "rails": page.locator(".fpc-rail").count(),
        "cpu_fallback": page.locator(".cpu-note:not([hidden])").count() == 1,
        "message": page.locator(".fpc-message").inner_text(),
        "errors": errors,
        "failed_responses": failed,
    }
    assert not failed, failed
    assert not [error for error in errors if "Failed to load resource" not in error], errors
    desktop.close()

    mobile, page, errors, failed = browser_page(browser, 390, 844)
    page.goto(f"{BASE}/#/g/four-player-chess")
    page.wait_for_load_state("networkidle")
    page.locator(".fpc-root").wait_for(timeout=30_000)
    page.screenshot(path=OUT / "four-player-chess-mobile.png", full_page=True)
    board_box = page.locator(".fpc-stage").bounding_box()
    assert board_box is not None and board_box["width"] <= 390
    results["mobile"] = {"board": board_box, "errors": errors, "failed_responses": failed}
    assert not failed, failed
    assert not [error for error in errors if "Failed to load resource" not in error], errors
    mobile.close()

    parity, page, errors, failed = browser_page(browser, 1100, 850)
    page.goto(f"{BASE}/four-player-chess-azero-test.html")
    page.wait_for_load_state("networkidle")
    page.wait_for_function(
        "document.querySelector('#log').textContent.includes('PASS') || "
        "document.querySelector('#log').textContent.includes('ERROR')",
        timeout=90_000,
    )
    parity_text = page.locator("#log").inner_text()
    page.screenshot(path=OUT / "four-player-chess-inference.png", full_page=True)
    results["inference"] = {"text": parity_text, "errors": errors, "failed_responses": failed}
    assert not failed, failed
    if "PASS" not in parity_text:
        assert "WebGPU" in parity_text or "GPU" in parity_text, parity_text
    parity.close()

    browser.close()

    cpu_browser = playwright.chromium.launch(
        channel="chrome",
        headless=True,
        args=["--disable-gpu", "--disable-software-rasterizer"],
    )
    fallback, page, errors, failed = browser_page(cpu_browser, 1200, 900)
    page.goto(f"{BASE}/#/g/four-player-chess")
    page.wait_for_load_state("networkidle")
    note = page.locator(".cpu-note:not([hidden])")
    note.wait_for(timeout=90_000)
    fallback_text = note.inner_text()
    assert "CPU FALLBACK ACTIVE" in fallback_text
    page.locator(".fpc-origin").first.wait_for(timeout=90_000)
    page.screenshot(path=OUT / "four-player-chess-cpu-fallback.png", full_page=True)
    results["cpu_fallback"] = {
        "text": fallback_text,
        "errors": errors,
        "failed_responses": failed,
    }
    assert not failed, failed
    assert not [error for error in errors if "Failed to load resource" not in error], errors
    fallback.close()
    cpu_browser.close()
    print(json.dumps(results, indent=2))
