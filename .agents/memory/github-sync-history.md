---
name: GitHub sync history
description: How to reconcile a GitHub API commit with Replit's ahead local branch without losing the workspace implementation.
---

When a GitHub API commit is created from the remote branch while Replit has additional local commits, the branches can show both incoming and outgoing commits and the Git panel may report an unexpected merge conflict. Preserve the Replit workspace tree, create a safety branch, and merge the remote branch with the `ours` strategy before using the Git panel's Push action.

**Why:** The authenticated GitHub connector and the shell/Replit Git remote can use different credentials and histories. A remote API commit may be valid but not be an ancestor of the local Replit branch, so pulling can try to merge overlapping source changes.

**How to apply:** Do not pull blindly. First ensure the worktree is understood, create a backup branch, reconcile the remote commit while keeping the local implementation, then push from the authenticated Git panel. Leave user-uploaded untracked files uncommitted unless explicitly requested.