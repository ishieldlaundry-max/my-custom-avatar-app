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
            getUserMediaHistory: [],
            enumerateDevicesCalls: 0,
            permissionGranted: false,
            rejectEnumerateDevices: false,
            enumeratedDevices: [
                { kind: "audioinput", deviceId: "builtin-mic", label: "Built-in Microphone" },
                { kind: "audioinput", deviceId: "usb-mic", label: "USB Headset Microphone" },
                { kind: "videoinput", deviceId: "camera-1", label: "Webcam" }
            ],
            rejectGetUserMedia: false,
            getUserMediaErrorName: "NotAllowedError",
            getUserMediaErrorMessage: "microphone permission was denied",
            rejectWorklet: false,
            workletErrorName: "SecurityError",
            removeAudioWorkletNode: false,
            removeAudioWorkletProperty: false,
            trackStopped: false,
            trackStopCalls: 0,
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
                window.__audioTest.trackStopCalls += 1;
                window.__audioTest.persistCleanup();
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
                window.__audioTest.getUserMediaHistory.push(constraints);
                if (window.__audioTest.rejectGetUserMedia) {
                    const error = new Error(window.__audioTest.getUserMediaErrorMessage);
                    error.name = window.__audioTest.getUserMediaErrorName;
                    throw error;
                }
                window.__audioTest.permissionGranted = true;
                return new FakeStream();
            }
        });
        Object.defineProperty(navigator.mediaDevices, "enumerateDevices", {
            configurable: true,
            value: async () => {
                window.__audioTest.enumerateDevicesCalls += 1;
                if (window.__audioTest.rejectEnumerateDevices) {
                    throw new Error("device enumeration failed");
                }
                return window.__audioTest.enumeratedDevices.map((device) => ({
                    ...device,
                    label: window.__audioTest.permissionGranted ? device.label : ""
                }));
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


def test_voice_shifter_lists_microphones_after_permission_and_uses_selected_device(
    voice_shifter_page,
):
    page = voice_shifter_page
    device_select = page.locator("[data-voice-device]")
    list_button = page.locator("[data-voice-list]")
    device_status = page.locator("[data-voice-device-status]")
    start_button = page.locator("[data-voice-start]")

    assert device_select.is_disabled()
    assert "names appear after permission" in device_status.text_content()
    assert page.evaluate("() => window.__audioTest.enumerateDevicesCalls") == 0

    list_button.click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-device-status]').textContent.includes('Found 2 microphone inputs')"
    )

    listed = page.evaluate(
        """() => ({
            options: [...document.querySelector('[data-voice-device]').options]
                .map(option => ({label: option.textContent, value: option.value})),
            getUserMediaCalls: window.__audioTest.getUserMediaCalls,
            getUserMediaHistory: window.__audioTest.getUserMediaHistory,
            enumerateDevicesCalls: window.__audioTest.enumerateDevicesCalls,
            permissionGranted: window.__audioTest.permissionGranted,
            trackStopCalls: window.__audioTest.trackStopCalls
        })"""
    )
    assert listed["options"] == [
        {"label": "System default microphone", "value": ""},
        {"label": "Built-in Microphone", "value": "builtin-mic"},
        {"label": "USB Headset Microphone", "value": "usb-mic"},
    ]
    assert listed["getUserMediaCalls"] == 1
    assert listed["getUserMediaHistory"] == [{"audio": True}]
    assert listed["enumerateDevicesCalls"] == 1
    assert listed["permissionGranted"] is True
    assert listed["trackStopCalls"] == 1
    assert device_select.is_enabled()
    assert list_button.text_content() == "Refresh microphone list"

    device_select.select_option("usb-mic")
    start_button.click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-status]').textContent.includes('Live')"
    )

    running = page.evaluate(
        """() => ({
            status: document.querySelector('[data-voice-status]').textContent,
            constraints: window.__audioTest.getUserMediaConstraints,
            contextState: window.__audioTest.contexts[0].state,
            deviceDisabled: document.querySelector('[data-voice-device]').disabled,
            listDisabled: document.querySelector('[data-voice-list]').disabled,
            trackStopCalls: window.__audioTest.trackStopCalls
        })"""
    )
    assert running["constraints"]["audio"] == {
        "echoCancellation": False,
        "noiseSuppression": False,
        "autoGainControl": False,
        "deviceId": {"exact": "usb-mic"},
    }
    assert "USB Headset Microphone" in running["status"]
    assert running["contextState"] == "running"
    assert running["deviceDisabled"] is True
    assert running["listDisabled"] is True
    assert running["trackStopCalls"] == 1

    start_button.click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-status]').textContent === 'Microphone idle'"
    )
    stopped = page.evaluate(
        """() => ({
            trackStopCalls: window.__audioTest.trackStopCalls,
            contextState: window.__audioTest.contexts[0].state,
            disconnects: window.__audioTest.disconnects,
            deviceDisabled: document.querySelector('[data-voice-device]').disabled,
            listDisabled: document.querySelector('[data-voice-list]').disabled
        })"""
    )
    assert stopped == {
        "trackStopCalls": 2,
        "contextState": "closed",
        "disconnects": 2,
        "deviceDisabled": False,
        "listDisabled": False,
    }


def test_voice_shifter_microphone_list_permission_denial_is_explained(voice_shifter_page):
    page = voice_shifter_page
    page.evaluate("() => { window.__audioTest.rejectGetUserMedia = true; }")
    page.locator("[data-voice-list]").click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-device-status]').textContent.includes('Microphone access was blocked')"
    )

    result = page.evaluate(
        """() => ({
            status: document.querySelector('[data-voice-device-status]').textContent,
            selectDisabled: document.querySelector('[data-voice-device]').disabled,
            listDisabled: document.querySelector('[data-voice-list]').disabled,
            enumerateDevicesCalls: window.__audioTest.enumerateDevicesCalls,
            trackStopCalls: window.__audioTest.trackStopCalls
        })"""
    )
    assert "Chrome or Edge" in result["status"]
    assert "Settings > Privacy & security > Microphone" in result["status"]
    assert result["selectDisabled"] is True
    assert result["listDisabled"] is False
    assert result["enumerateDevicesCalls"] == 0
    assert result["trackStopCalls"] == 0


def test_voice_shifter_recovers_after_no_inputs_are_listed(voice_shifter_page):
    page = voice_shifter_page
    list_button = page.locator("[data-voice-list]")
    device_select = page.locator("[data-voice-device]")

    page.evaluate("() => { window.__audioTest.enumeratedDevices = []; }")
    list_button.click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-device-status]').textContent.includes('Windows listed no individual microphone inputs')"
    )

    status = page.locator("[data-voice-device-status]").text_content()
    assert "Settings > System > Sound > Input" in status
    assert "click Refresh microphone list" in status
    assert device_select.is_disabled()
    assert list_button.text_content() == "Refresh microphone list"

    page.evaluate(
        """() => {
            window.__audioTest.enumeratedDevices = [
                { kind: "audioinput", deviceId: "builtin-mic", label: "Built-in Microphone" }
            ];
        }"""
    )
    list_button.click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-device-status]').textContent.includes('Found 1 microphone input')"
    )
    device_select.select_option("builtin-mic")
    page.locator("[data-voice-start]").click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-status]').textContent.includes('Live')"
    )
    assert "Built-in Microphone" in page.locator("[data-voice-status]").text_content()


def test_voice_shifter_stops_permission_probe_if_device_enumeration_fails(
    voice_shifter_page,
):
    page = voice_shifter_page
    page.evaluate("() => { window.__audioTest.rejectEnumerateDevices = true; }")
    page.locator("[data-voice-list]").click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-device-status]').textContent.includes('device enumeration failed')"
    )

    result = page.evaluate(
        """() => ({
            status: document.querySelector('[data-voice-device-status]').textContent,
            selectDisabled: document.querySelector('[data-voice-device]').disabled,
            listDisabled: document.querySelector('[data-voice-list]').disabled,
            enumerateDevicesCalls: window.__audioTest.enumerateDevicesCalls,
            trackStopCalls: window.__audioTest.trackStopCalls,
            contextCount: window.__audioTest.contexts.length
        })"""
    )
    assert "Microphone permission was granted" in result["status"]
    assert result["selectDisabled"] is True
    assert result["listDisabled"] is False
    assert result["enumerateDevicesCalls"] == 1
    assert result["trackStopCalls"] == 1
    assert result["contextCount"] == 0


def test_voice_shifter_reports_worklet_startup_error_and_cleans_up(voice_shifter_page):
    page = voice_shifter_page
    start_button = page.locator("[data-voice-start]")
    status = page.locator("[data-voice-status]")

    page.evaluate("() => { window.__audioTest.rejectWorklet = true; }")
    start_button.click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-status]').textContent.includes('Audio processing could not start')"
    )

    failed = page.evaluate(
        """() => ({
            status: document.querySelector('[data-voice-status]').textContent,
            button: document.querySelector('[data-voice-start]').textContent,
            trackStopped: window.__audioTest.trackStopped,
            contextState: window.__audioTest.contexts[0].state,
            registeredWorklets: window.__audioTest.registeredWorklets,
            disconnects: window.__audioTest.disconnects
        })"""
    )

    assert "Audio processing could not start" in failed["status"]
    assert "audio worklet initialization failed" in failed["status"]
    assert failed["button"] == "Start feminine pitch"
    assert failed["trackStopped"] is True
    assert failed["contextState"] == "closed"
    assert failed["registeredWorklets"] == []
    assert failed["disconnects"] == 0
    assert status.text_content() == failed["status"]


def test_voice_shifter_explains_denied_permission_and_retry(voice_shifter_page):
    page = voice_shifter_page
    page.evaluate("() => { window.__audioTest.rejectGetUserMedia = true; }")
    page.locator("[data-voice-start]").click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-status]').textContent.includes('Microphone access was blocked')"
    )

    result = page.evaluate(
        """() => ({
            status: document.querySelector('[data-voice-status]').textContent,
            button: document.querySelector('[data-voice-start]').textContent,
            contextCount: window.__audioTest.contexts.length
        })"""
    )
    assert "Microphone access was blocked" in result["status"]
    assert "Chrome or Edge" in result["status"]
    assert "site's controls beside the address bar" in result["status"]
    assert "Settings > Privacy & security > Microphone" in result["status"]
    assert "Let desktop apps access your microphone" in result["status"]
    assert "Allow access & list microphones" in result["status"]
    assert result["button"] == "Start feminine pitch"
    assert result["contextCount"] == 0

    page.evaluate("() => { window.__audioTest.rejectGetUserMedia = false; }")
    page.locator("[data-voice-list]").click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-device-status]').textContent.includes('Found 2 microphone inputs')"
    )
    page.locator("[data-voice-device]").select_option("builtin-mic")
    page.locator("[data-voice-start]").click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-status]').textContent.includes('Live')"
    )
    assert "Built-in Microphone" in page.locator("[data-voice-status]").text_content()


def test_voice_shifter_explains_missing_microphone_and_device_recovery(voice_shifter_page):
    page = voice_shifter_page
    page.evaluate(
        """() => {
            window.__audioTest.rejectGetUserMedia = true;
            window.__audioTest.getUserMediaErrorName = "NotFoundError";
            window.__audioTest.getUserMediaErrorMessage = "no input device";
        }"""
    )
    page.locator("[data-voice-start]").click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-status]').textContent.includes('No microphone input was found')"
    )

    status = page.locator("[data-voice-status]").text_content()
    assert "Connect or enable a microphone" in status
    assert "Settings > System > Sound > Input" in status
    assert "Refresh microphone list" in status
    assert "try Start again" in status

    page.evaluate("() => { window.__audioTest.rejectGetUserMedia = false; }")
    page.locator("[data-voice-list]").click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-device-status]').textContent.includes('Found 2 microphone inputs')"
    )
    page.locator("[data-voice-device]").select_option("builtin-mic")
    page.locator("[data-voice-start]").click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-status]').textContent.includes('Live')"
    )
    assert "Built-in Microphone" in page.locator("[data-voice-status]").text_content()


def test_voice_shifter_explains_busy_microphone_recovery(voice_shifter_page):
    page = voice_shifter_page
    page.evaluate(
        """() => {
            window.__audioTest.rejectGetUserMedia = true;
            window.__audioTest.getUserMediaErrorName = "NotReadableError";
            window.__audioTest.getUserMediaErrorMessage = "microphone is busy";
        }"""
    )
    page.locator("[data-voice-start]").click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-status]').textContent.includes('The microphone is unavailable')"
    )

    status = page.locator("[data-voice-status]").text_content()
    assert "Close apps such as Teams or Zoom" in status
    assert "Settings > Privacy & security > Microphone" in status
    assert "Let desktop apps access your microphone" in status
    assert "refresh the microphone list" in status
    assert "try Start again" in status

    page.evaluate("() => { window.__audioTest.rejectGetUserMedia = false; }")
    page.locator("[data-voice-start]").click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-status]').textContent.includes('Live')"
    )


@pytest.mark.parametrize("missing_part", ["audioWorklet", "AudioWorkletNode"])
def test_voice_shifter_detects_missing_audio_worklet_before_microphone_prompt(
    voice_shifter_page, missing_part
):
    page = voice_shifter_page
    page.evaluate(
        "(part) => window.__audioTest.setAudioWorkletSupport(false, part)",
        missing_part,
    )
    page.locator("[data-voice-start]").click()

    result = page.evaluate(
        """() => ({
            status: document.querySelector('[data-voice-status]').textContent,
            button: document.querySelector('[data-voice-start]').textContent,
            microphoneCalls: window.__audioTest.getUserMediaCalls,
            contexts: window.__audioTest.contexts.length,
            trackStopped: window.__audioTest.trackStopped
        })"""
    )
    assert "does not support the AudioWorklet" in result["status"]
    assert "current Chrome or Edge" in result["status"]
    assert "try Start again" in result["status"]
    assert result["button"] == "Start feminine pitch"
    assert result["microphoneCalls"] == 0
    assert result["contexts"] == 0
    assert result["trackStopped"] is False


def test_voice_shifter_cleans_up_when_page_navigates_away(voice_shifter_page):
    page = voice_shifter_page
    start_button = page.locator("[data-voice-start]")

    start_button.click()
    page.wait_for_function(
        "() => document.querySelector('[data-voice-status]').textContent.includes('Live')"
    )

    destination = page.evaluate("() => location.origin + '/after.html'")
    page.goto(destination, wait_until="domcontentloaded")
    cleanup = page.evaluate(
        "() => JSON.parse(localStorage.getItem('voice-shifter-cleanup'))"
    )

    assert cleanup == {
        "trackStopped": True,
        "contextState": "closed",
    }
