---
name: Local NVIDIA workflow
description: Deployment boundary for the FasterLivePortrait interface
---

The intended runtime is the user's local Windows/NVIDIA workstation. Replit is an editing and preview environment for the Gradio front end, not the production inference host.

**Why:** The repository is NVIDIA-optimized and the heavy model execution, webcam capture, and browser microphone flow are intended to remain local.

**How to apply:** Preserve the existing pipeline callback signatures and local launch arguments when redesigning `webui.py`. Validate visual startup in Replit where possible, but give desktop instructions for final GPU and browser-device testing.