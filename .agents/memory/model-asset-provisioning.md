---
name: Model asset provisioning
description: Runtime requirements for FasterLivePortrait model and optional voice assets
---

The repository does not contain the runtime checkpoint files because `checkpoints/` is ignored. A working ONNX web UI therefore requires downloading the FasterLivePortrait checkpoint tree before startup.

**Why:** A clean import can pass code and dependency checks but still fail during model construction when the ONNX files are absent. The optional Kokoro text-driving assets are a separate model tree and are not required for video, image, pickle, or audio driving.

**How to apply:** Provision the FasterLivePortrait checkpoints before starting the web UI. Treat the Kokoro voice directory as optional and keep startup functional when it is absent; text driving should only be enabled after that model is installed.