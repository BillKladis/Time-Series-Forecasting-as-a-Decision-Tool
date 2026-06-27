"""
Capture screenshots of the Stock Sense Streamlit app using Playwright.
"""

import os
import subprocess
import sys
import time

import requests

SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
APP_PORT = 8502
APP_URL = f"http://localhost:{APP_PORT}"


def wait_for_streamlit(timeout: int = 60) -> bool:
    """Poll until Streamlit is ready or timeout expires."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(APP_URL, timeout=3)
            if resp.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def capture_screenshots():
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)

    # Start Streamlit
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            app_path,
            "--server.port",
            str(APP_PORT),
            "--server.headless",
            "true",
            "--server.enableCORS",
            "false",
            "--server.enableXsrfProtection",
            "false",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    print(f"Started Streamlit (PID {proc.pid})...")

    try:
        print("Waiting for Streamlit to be ready...")
        ready = wait_for_streamlit(timeout=60)
        if not ready:
            print("ERROR: Streamlit did not start in time.")
            proc.terminate()
            sys.exit(1)
        print("Streamlit is ready!")

        # Extra wait for app to fully initialize
        time.sleep(5)

        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            # Use the system-installed headless shell binary
            chromium_executable = "/opt/pw-browsers/chromium_headless_shell-1194/chrome-linux/headless_shell"
            if not os.path.exists(chromium_executable):
                # Fallback: let playwright find it automatically
                chromium_executable = None
            browser = p.chromium.launch(
                headless=True,
                **({"executable_path": chromium_executable} if chromium_executable else {}),
            )
            context = browser.new_context(
                viewport={"width": 1400, "height": 900},
            )
            page = context.new_page()

            # Navigate to the app
            print(f"Opening {APP_URL}...")
            page.goto(APP_URL, wait_until="networkidle", timeout=60000)
            print("Page loaded. Waiting for charts...")

            # Wait for plotly charts to render
            try:
                page.wait_for_selector(".js-plotly-plot", timeout=60000)
            except Exception:
                print("Warning: Could not find plotly chart selector, sleeping 10s...")
                time.sleep(10)

            # Extra wait for everything to settle
            time.sleep(5)

            # Screenshot 1: Forecast tab (default)
            screenshot_1 = os.path.join(SCREENSHOTS_DIR, "01_forecast_chart.png")
            page.screenshot(path=screenshot_1, full_page=True)
            print(f"Saved: {screenshot_1}")

            # Click Benchmark tab
            print("Clicking Benchmark tab...")
            try:
                page.click("text=Benchmark")
            except Exception:
                try:
                    page.click("button:has-text('Benchmark')")
                except Exception:
                    # Try finding tab by partial text
                    page.locator("[role='tab']").filter(has_text="Benchmark").click()

            time.sleep(6)

            screenshot_2 = os.path.join(SCREENSHOTS_DIR, "02_benchmark_table.png")
            page.screenshot(path=screenshot_2, full_page=True)
            print(f"Saved: {screenshot_2}")

            # Click Decision tab
            print("Clicking Decision tab...")
            try:
                page.click("text=Decision")
            except Exception:
                try:
                    page.click("button:has-text('Decision')")
                except Exception:
                    page.locator("[role='tab']").filter(has_text="Decision").click()

            time.sleep(3)

            screenshot_3 = os.path.join(SCREENSHOTS_DIR, "03_decision_callout.png")
            page.screenshot(path=screenshot_3, full_page=True)
            print(f"Saved: {screenshot_3}")

            browser.close()

        # Verify file sizes
        print("\n--- Screenshot Verification ---")
        all_ok = True
        for fname in ["01_forecast_chart.png", "02_benchmark_table.png", "03_decision_callout.png"]:
            fpath = os.path.join(SCREENSHOTS_DIR, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                status = "OK" if size > 10 * 1024 else "TOO SMALL"
                print(f"  {fname}: {size:,} bytes [{status}]")
                if size <= 10 * 1024:
                    all_ok = False
            else:
                print(f"  {fname}: MISSING")
                all_ok = False

        if all_ok:
            print("\nAll screenshots captured successfully!")
        else:
            print("\nWARNING: Some screenshots may be missing or too small.")

    finally:
        print(f"\nKilling Streamlit (PID {proc.pid})...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Done.")


if __name__ == "__main__":
    capture_screenshots()
