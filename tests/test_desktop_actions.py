from types import SimpleNamespace

from desktop_launcher import choose_portrait, guarded_action


def test_picker_uses_pywebview_5_constant():
    calls = []
    def picker(dialog, **kwargs):
        calls.append((dialog, kwargs))
        return ("C:\\photos\\portrait.png",)
    result = choose_portrait(
        SimpleNamespace(create_file_dialog=picker), SimpleNamespace(OPEN_DIALOG=10)
    )
    assert result == ("C:\\photos\\portrait.png",)
    assert calls[0][0] == 10
    assert calls[0][1]["allow_multiple"] is False


def test_picker_cancel_is_not_an_error():
    window = SimpleNamespace(create_file_dialog=lambda *a, **kw: None)
    assert choose_portrait(window, SimpleNamespace(OPEN_DIALOG=10)) is None


def test_menu_error_is_logged_and_displayed(tmp_path):
    def broken():
        raise RuntimeError("dialog failed")
    messages = []
    log = tmp_path / "launcher.log"
    guarded_action(broken, log, messages.append)()
    assert "dialog failed" in log.read_text()
    assert "dialog failed" in messages[0]


def test_success_returns_result_without_error(tmp_path):
    messages = []
    log = tmp_path / "launcher.log"
    assert guarded_action(lambda: 42, log, messages.append)() == 42
    assert not messages
    assert not log.exists()
