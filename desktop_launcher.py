"""Windows desktop shell. Uses the existing project venv, not bundled ML packages."""
import ctypes
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request


def project_root():
    return Path(sys.executable if getattr(sys, "frozen", False) else __file__).resolve().parent


def app_python(root):
    python = root / "venv" / "Scripts" / "python.exe"
    if not python.is_file() or not (root / "webui.py").is_file():
        raise RuntimeError("Place AvatarDesktop.exe in FasterLivePortrait-local, beside webui.py "
                           "and the existing venv folder. Do not move or replace that environment.")
    return python


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def choose_portrait(window, webview):
    """Use the dialog API supported by the pinned pywebview 5.x build."""
    return window.create_file_dialog(
        webview.OPEN_DIALOG, allow_multiple=False,
        file_types=("Portrait images (*.png;*.jpg;*.jpeg;*.webp)",),
    )


def guarded_action(action, error_path, notify):
    """Windowed executables have no console: surface callback errors explicitly."""
    def invoke():
        try:
            return action()
        except Exception as exc:
            try:
                with error_path.open("a", encoding="utf-8") as stream:
                    traceback.print_exc(file=stream)
            except OSError:
                pass
            notify(f"{exc}\n\nDetails: {error_path}")
    return invoke


def main():
    if os.name != "nt":
        raise RuntimeError("The desktop launcher is for Windows.")
    import webview
    from webview.menu import Menu, MenuAction

    root = project_root()
    python = app_python(root)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = logs / f"desktop-{stamp}.log"
    streams = []
    children = []
    stopped = threading.Event()
    live = None

    def start_process(arguments, label):
        stream = (logs / f"{label}-{stamp}.log").open("a", encoding="utf-8")
        streams.append(stream)
        process = subprocess.Popen(
            [str(python), "-u", *arguments], cwd=root, stdout=stream,
            stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW,
        )
        children.append(process)
        return process

    def stop_process(process):
        if process and process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                           capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)

    def cleanup():
        stopped.set()
        for process in children:
            stop_process(process)
        for stream in streams:
            stream.close()

    port = free_port()
    server = start_process(["webui.py", "--mode", "onnx", "--require-nvidia",
                            "--host_ip", "127.0.0.1", "--port", str(port)], "desktop")
    url = f"http://127.0.0.1:{port}"
    window = webview.create_window(
        "Avatar Studio — NVIDIA", html="<html><body style='background:#101b29;color:white;"
        "font:18px sans-serif;padding:40px'><h1>Starting Avatar Studio</h1>"
        "<p>Checking NVIDIA inference and loading models. This can take a few minutes.</p>"
        "<p>The app stops rather than silently using CPU inference.</p></body></html>",
        width=1180, height=780, min_size=(640, 480),
    )

    def notice(message):
        ctypes.windll.user32.MessageBoxW(0, str(message), "Avatar Studio", 0x10)

    def start_live():
        nonlocal live
        if live and live.poll() is None:
            notice("Live tracking is already running. Use Stop live tracking first.")
            return
        files = choose_portrait(window, webview)
        if not files:
            return
        if not window.create_confirmation_dialog(
            "Start live tracking",
            "This closes the workspace server to free GPU memory. The portrait will open "
            "in a separate live tracking window. Close other webcam apps first. Continue?"
        ):
            return
        stop_process(server)
        live = start_process(["run.py", "--cfg", "configs/onnx_infer.yaml", "--realtime",
                              "--require-nvidia", "--dri_video", "0", "--paste_back",
                              "--src_image", files[0]], "live")
        window.load_html("<html><body><h2>Live tracking started</h2><p>Look for the "
                         "portrait preview window. If none appears, open logs from the App menu. "
                         "Restart Avatar Studio to return to the workspace.</p></body></html>")
        def watch_live():
            code = live.wait()
            if code and not stopped.is_set():
                notice(f"Live tracking stopped with an error. See {logs / ('live-' + stamp + '.log')}")
        threading.Thread(target=watch_live, daemon=True).start()

    def wait_for_server():
        deadline = time.monotonic() + 240
        while not stopped.is_set() and time.monotonic() < deadline:
            if server.poll() is not None:
                notice(f"Startup failed. No CPU fallback was used. Open this log:\n{log_path}")
                return
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        window.load_url(url)
                        return
            except OSError:
                pass
            stopped.wait(0.5)
        if not stopped.is_set():
            notice(f"Startup timed out. See {log_path}")
            stop_process(server)

    window.events.closed += cleanup
    def guarded(action):
        return guarded_action(action, logs / f"launcher-{stamp}.log", notice)

    menu = [Menu("App", [
        MenuAction("Start live portrait from photo", guarded(start_live)),
        MenuAction("Stop live tracking", guarded(lambda: stop_process(live))),
        MenuAction("Open logs", guarded(lambda: os.startfile(str(logs)))),
        MenuAction("Exit", window.destroy),
    ])]
    try:
        webview.start(wait_for_server, gui="edgechromium", menu=menu)
    finally:
        cleanup()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if os.name == "nt":
            ctypes.windll.user32.MessageBoxW(0, str(exc), "Avatar Studio startup error", 0x10)
        else:
            raise
