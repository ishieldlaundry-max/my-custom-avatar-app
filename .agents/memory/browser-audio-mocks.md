---
name: Browser audio mocks
description: A Chromium behavior that matters when testing the voice shifter without hardware.
---

When mocking microphone capture in Chromium, override `navigator.mediaDevices.getUserMedia` on the existing `mediaDevices` object instead of replacing `navigator.mediaDevices`.

**Why:** The browser can expose `mediaDevices` as a read-only navigator property, so assignment silently leaves the real capture implementation in place and causes a hardware-dependent “device not found” failure.

**How to apply:** Install the mock method with `Object.defineProperty` and keep the rest of the Web Audio surface mocked only as far as the lifecycle assertions require.