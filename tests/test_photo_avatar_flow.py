from pathlib import Path


def test_generated_photo_avatar_is_loaded_from_same_origin_parent_message():
    source = Path("assets/live_avatar.html").read_text(encoding="utf-8")

    assert 'event.source !== window.parent' in source
    assert 'event.origin !== window.location.origin' in source
    assert 'message?.type !== "load-photo-avatar"' in source
    assert 'modelUrl.pathname.startsWith("/gradio_api/file=")' in source
    assert "await loadAvatar(modelUrl.href" in source


def test_photo_avatar_creation_replaces_manual_rigging_step():
    source = Path("webui.py").read_text(encoding="utf-8")

    assert "generate_rigged_local_photo_avatar(" in source
    assert "CREATE & LOAD MY LIVE AVATAR" in source
    assert "photo_avatar_file.change(" in source
    assert "photo_avatar_rig_button" not in source
    assert 'type: "load-photo-avatar"' in source
