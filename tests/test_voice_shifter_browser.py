"""Browser smoke coverage for the hardware-independent voice shifter lifecycle."""

from __future__ import annotations

import ast
import functools
import http.server
import os
import shutil
import threading
from pathlib import Path

import pytest


def _webui_constant(name: str) -> str:
    """Read a JavaScript/HTML constant without importing webui's model pipeline."""
    source = Path(__file__).parents[1].joinpath("webui.py").read_text(encoding="utf-8")
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise TypeError(f"{name} must be a string")
            return value
    raise AssertionError(f"{name} was not found in webui.py")


def _chromium_executable() -> str | None:
    configured = os.environ.get("CHROMIUM_PATH")
    if configured and Path(configured).is_file():
        return configured
    return shutil.which("chromium") or shutil.which("chromium-browser")


@pytest.fixture
def voice_shifter_page(tmp_path):
    playwright = pytest.importorskip("playwright.sync_api")
    chromium = _chromium_executable()
    if chromium is None:
        pytest.skip("Chromium is required for the voice shifter browser smoke test")

    audio_widget = _webui_constant("AUDIO_WIDGET_HTML")
    launch_script = _webui_constant("js_func")
    test_harness = r"""
        <script>
        window.__audioTest = {
            getUserMediaCalls: 0,
            getUserMediaConstraints: null,
            trackStopped: false,
            contexts: [],
            workletSources: [],
            registeredWorklets: [],
            nodes: [],
            connections: [],
            disconnects: 0
        };

        class FakeTrack {
            stop() {
                window.__audioTest.trackStopped = true;
            }
        }

        class FakeStream {
            constructor() {
                this.track = new FakeTrack();
            }

            getTracks() {
                return [this.track];
            }
        }

        class FakeSource {
            constructor(context) {
                this.context = context;
            }

            connect(target) {
                window.__audioTest.connections.push(
                    target === this.context.destination
                        ? "source->destination"
                        : "source->worklet"
                );
                return target;
            }

            disconnect() {
                window.__audioTest.disconnects += 1;
            }
        }

        class FakeAudioContext {
            constructor() {
                this.state = "suspended";
                this.destination = {};
                this.audioWorklet = {
                    addModule: async (moduleUrl) => {
                        const source = await fetch(moduleUrl).then((response) => response.text());
                        window.__audioTest.workletSources.push(source);
                        if (!source.includes('registerProcessor("cyber-pitch-shift"')) {
                            throw new Error("the +12 pitch worklet was not registered");
                        }
                        if (!source.includes("this.ratio = 2.0")) {
                            throw new Error("the worklet is not configured for +12 semitones");
                        }
                        window.__audioTest.registeredWorklets.push("cyber-pitch-shift");
                    }
                };
                window.__audioTest.contexts.push(this);
            }

            createMediaStreamSource() {
                return new FakeSource(this);
            }

            resume() {
                this.state = "running";
                return Promise.resolve();
            }

            close() {
                this.state = "closed";
                return Promise.resolve();
            }
        }

        class FakeAudioWorkletNode {
            constructor(context, name) {
                this.context = context;
                this.name = name;
                window.__audioTest.nodes.push(name);
            }

            connect(target) {
                window.__audioTest.connections.push(
                    target === this.context.destination
                        ? "worklet->destination"
                        : "worklet->unknown"
                );
                return target;
            }

            disconnect() {
                window.__audioTest.disconnects += 1;
            }
        }

        Object.defineProperty(navigator.mediaDevices, "getUserMedia", {
            configurable: true,
            value: async (constraints) => {
                window.__audioTest.getUserMediaCalls += 1;
                window.__audioTest.getUserMediaConstraints = constraints;
                return new FakeStream();
            }
        });
        window.AudioContext = FakeAudioContext;
        window.AudioWorkletNode = FakeAudioWorkletNode;
        </script>
    """

    page_source = f"""<!doctype html>
    <html>
      <head><meta charset="utf-8"><title>Voice shifter smoke test</title></head>
      <body>
        {audio_widget}
        {test_harness}
        <script>({launch_script})();</script>
      </body>
    </html>
    """
    (tmp_path / "index.html").write_text(page_source, encoding="utf-8")

    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    with playwright.sync_playwright() as playwright_context:
        browser = playwright_context.chromium.launch(
            executable_path=chromium,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page()
        page.goto(
            f"http://127.0.0.1:{server.server_port}/index.html?__theme=dark",
            wait_until="domcontentloaded",
        )
        page.wait_for_function(
            "() => document.querySelector('[data-pitch-shift-widget]').dataset.bound === 'true'"
        )
        try:
            yield page
        finally:
            browser.close()
            server.shutdown()
            thread.join(timeout=5)


def test_voice_shifter_routes_virtual_microphone_and_cleans_up(voice_shifter_page):
    page = voice_shifter_page
    start_button = page.locator("[data-voice-start]")
    status = page.locator("[data-voice-status]")

    start_button.click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-status]').textContent.includes('Live')"
    )

    running = page.evaluate(
        """() => ({
            status: document.querySelector('[data-voice-status]').textContent,
            button: document.querySelector('[data-voice-start]').textContent,
            getUserMediaCalls: window.__audioTest.getUserMediaCalls,
            getUserMediaConstraints: window.__audioTest.getUserMediaConstraints,
            registeredWorklets: window.__audioTest.registeredWorklets,
            workletSource: window.__audioTest.workletSources[0],
            nodes: window.__audioTest.nodes,
            connections: window.__audioTest.connections,
            contextState: window.__audioTest.contexts[0].state
        })"""
    )

    assert running["getUserMediaCalls"] == 1
    assert running["getUserMediaConstraints"] == {
        "audio": {
            "echoCancellation": False,
            "noiseSuppression": False,
            "autoGainControl": False,
        }
    }
    assert running["registeredWorklets"] == ["cyber-pitch-shift"]
    assert 'registerProcessor("cyber-pitch-shift"' in running["workletSource"]
    assert "this.ratio = 2.0" in running["workletSource"]
    assert running["nodes"] == ["cyber-pitch-shift"]
    assert running["connections"] == ["source->worklet", "worklet->destination"]
    assert running["contextState"] == "running"
    assert "+12 semitones" in running["status"]
    assert running["button"] == "Stop voice shifter"

    start_button.click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-status]').textContent === 'Microphone idle'"
    )

    stopped = page.evaluate(
        """() => ({
            status: document.querySelector('[data-voice-status]').textContent,
            button: document.querySelector('[data-voice-start]').textContent,
            trackStopped: window.__audioTest.trackStopped,
            contextState: window.__audioTest.contexts[0].state,
            disconnects: window.__audioTest.disconnects
        })"""
    )

    assert stopped == {
        "status": "Microphone idle",
        "button": "Start feminine pitch",
        "trackStopped": True,
        "contextState": "closed",
        "disconnects": 2,
    }
    assert status.text_content() == "Microphone idle"