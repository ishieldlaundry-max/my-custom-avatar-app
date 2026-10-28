"""Build safe Markdown links to specific files served by Gradio."""

import re
from pathlib import Path
from urllib.parse import quote


_MARKDOWN_TEXT_SPECIALS = re.compile(r"([\\`*_{}\[\]()#+\-.!|>])")


def _escape_markdown_text(value: str) -> str:
    return _MARKDOWN_TEXT_SPECIALS.sub(r"\\\1", value)


def _inline_code(value: str) -> str:
    """Wrap text in a Markdown code span without colliding with backticks."""
    longest_run = max(
        (len(match.group(0)) for match in re.finditer(r"`+", value)),
        default=0,
    )
    delimiter = "`" * (longest_run + 1)
    padding = " " if value.startswith("`") or value.endswith("`") else ""
    return f"{delimiter}{padding}{value}{padding}{delimiter}"


def _allowed_gradio_file_link(
    path: str | Path,
    *,
    allowed_paths: list[str],
    allowed_root: str | Path,
) -> str | None:
    """Allow and link one existing MimicMotion log beneath its output folder."""
    resolved_path = Path(path).resolve()
    resolved_root = Path(allowed_root).resolve()
    if (
        not resolved_path.is_file()
        or resolved_path.suffix.lower() != ".log"
        or not resolved_path.name.startswith("mimicmotion-")
    ):
        return None
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return None

    allowed_path = str(resolved_path)
    if allowed_path not in allowed_paths:
        # Permit this exact log, not the whole output directory.
        allowed_paths.append(allowed_path)
    href = f"/gradio_api/file={quote(allowed_path, safe='/')}"
    return f"[{_inline_code(allowed_path)}]({href})"


def format_cancelled_render_status(
    message: str,
    log_path: str | Path | None,
    *,
    allowed_paths: list[str],
    allowed_root: str | Path,
) -> str:
    """Link the retained log in the existing cancellation message when safe."""
    status_message = _escape_markdown_text(message)
    if log_path is not None:
        resolved_path = Path(log_path).resolve()
        link = _allowed_gradio_file_link(
            resolved_path,
            allowed_paths=allowed_paths,
            allowed_root=allowed_root,
        )
        if link:
            before, separator, after = message.partition(str(resolved_path))
            if separator:
                status_message = (
                    _escape_markdown_text(before)
                    + link
                    + _escape_markdown_text(after)
                )
    return f"**Render cancelled.** {status_message}"
