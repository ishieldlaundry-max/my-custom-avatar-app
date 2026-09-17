---
name: Realtime motion fidelity
description: Constraints that keep webcam portrait motion responsive without weakening target identity.
---

Realtime rendering must use a latest-frame capture queue rather than preserving every webcam frame. Virtual-camera output should repeat the newest completed frame at a fixed 30 FPS instead of allowing a backlog to accumulate.

**Why:** Processing queued webcam frames increases input-to-output delay even when inference throughput is stable. Repeating the latest completed frame keeps downstream camera consumers paced while inference remains non-blocking.

**How to apply:** Keep capture and virtual-camera queues bounded to one frame. Drop stale captures, never queue rendered frames for completeness, and reuse target-side GPU tensors. Strict identity maps driver expression deltas into target landmark scale while freezing target pose, translation, scale, and appearance.