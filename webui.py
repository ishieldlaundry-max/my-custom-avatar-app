# coding: utf-8

"""
The entrance of the gradio
"""
import os
import pdb
import inspect
import importlib.util
import time
from pathlib import Path

import gradio as gr
import os.path as osp
from omegaconf import OmegaConf

from src.pipelines.gradio_live_portrait_pipeline import GradioLivePortraitPipeline
from src.pipelines.local_photo_to_3d import (
    LocalPhotoTo3DError,
    generate_rigged_local_photo_avatar,
)
from src.pipelines.full_body_animation_pipeline import (
    FullBodyAnimationCancelled,
    FullBodyAnimationError,
    FullBodyAnimationPipeline,
    FullBodyRenderOptions,
    extract_driver_preview_frames,
    inspect_full_body_inputs,
)


def load_description(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def existing_examples(*paths):
    """Return only bundled examples that exist in the current checkout."""
    return [[path] for path in paths if osp.isfile(path)]


import argparse

parser = argparse.ArgumentParser(description='Faster Live Portrait Pipeline')
parser.add_argument('--mode', required=False, choices=("onnx", "trt", "auto"), default="auto")
parser.add_argument('--use_mp', action='store_true', help='use mediapipe or not')
parser.add_argument(
    "--require-nvidia",
    action="store_true",
    help="Require a working NVIDIA runtime before selecting a backend or loading models.",
)
parser.add_argument(
    "--host_ip", type=str, default="127.0.0.1", help="host ip"
)
parser.add_argument("--port", type=int, default=9870, help="server port")
args, unknown = parser.parse_known_args()

def resolve_runtime_mode(requested_mode):
    """Select TRT only when both CUDA and the TensorRT package are available."""
    if requested_mode == "onnx":
        return "onnx"
    try:
        import torch
        trt_available = importlib.util.find_spec("tensorrt") is not None
        cuda_available = torch.cuda.is_available()
    except (ImportError, RuntimeError):
        trt_available = cuda_available = False
    if requested_mode == "trt" and not (cuda_available and trt_available):
        print("TensorRT was requested but is unavailable; falling back to ONNX.")
    return "trt" if cuda_available and trt_available else "onnx"


if args.require_nvidia or os.name == "nt":
    from src.utils.nvidia_runtime import enforce_nvidia_runtime

    enforce_nvidia_runtime()

runtime_mode = resolve_runtime_mode(args.mode)
if runtime_mode == "onnx":
    cfg_path = "configs/onnx_mp_infer.yaml" if args.use_mp else "configs/onnx_infer.yaml"
else:
    cfg_path = "configs/trt_mp_infer.yaml" if args.use_mp else "configs/trt_infer.yaml"
infer_cfg = OmegaConf.load(cfg_path)
gradio_pipeline = GradioLivePortraitPipeline(infer_cfg)
full_body_pipeline = FullBodyAnimationPipeline()


VIDEO_PIPELINE_ARGUMENT_NAMES = (
    "input_source_image_path",
    "input_source_video_path",
    "input_driving_video_path",
    "input_driving_image_path",
    "input_driving_pickle_path",
    "input_driving_audio_path",
    "input_driving_text",
    "flag_relative_input",
    "flag_do_crop_input",
    "flag_remap_input",
    "driving_multiplier",
    "flag_stitching",
    "flag_crop_driving_video_input",
    "flag_video_editing_head_rotation",
    "flag_is_animal",
    "animation_region",
    "scale",
    "vx_ratio",
    "vy_ratio",
    "scale_crop_driving_video",
    "vx_ratio_crop_driving_video",
    "vy_ratio_crop_driving_video",
    "driving_smooth_observation_variance",
    "tab_selection",
    "v_tab_selection",
    "cfg_scale",
    "voice_name",
    "flag_color_match",
    "stitching_blending_radius",
    "diagnostic_mode",
    "flag_virtual_camera_output",
    "strict_target_identity",
    "use_driver_background",
)


def gpu_wrapped_execute_video(source_image, source_video, webcam_enabled, webcam_video, *args, **kwargs):
    """Use webcam motion to drive the selected portrait across pipeline versions."""
    values = list((source_image, source_video, *args))
    if len(values) != len(VIDEO_PIPELINE_ARGUMENT_NAMES):
        raise TypeError(
            "The animation controls no longer match the video pipeline adapter: "
            f"expected {len(VIDEO_PIPELINE_ARGUMENT_NAMES)} values, received {len(values)}."
        )

    # The portrait remains the source identity. Webcam capture replaces only
    # the driving video, so the driver's appearance is never used as source.
    source_tab_index = VIDEO_PIPELINE_ARGUMENT_NAMES.index("tab_selection")
    driving_tab_index = VIDEO_PIPELINE_ARGUMENT_NAMES.index("v_tab_selection")
    if webcam_enabled and webcam_video:
        values[2] = webcam_video
        values[driving_tab_index] = "Video"

    if values[source_tab_index] == "Image" and not values[0] and values[1]:
        values[source_tab_index] = "Video"
    elif values[source_tab_index] == "Video" and not values[1] and values[0]:
        values[source_tab_index] = "Image"

    # Prefer an available driving input if a hidden tab value is stale.
    if values[driving_tab_index] == "Video" and not values[2] and values[3]:
        values[driving_tab_index] = "Image"
    elif values[driving_tab_index] == "Image" and not values[3] and values[2]:
        values[driving_tab_index] = "Video"

    diagnostic_mode = bool(values[VIDEO_PIPELINE_ARGUMENT_NAMES.index("diagnostic_mode")])
    execute_video = gradio_pipeline.execute_video
    supported_parameters = inspect.signature(execute_video).parameters
    call_kwargs = {
        name: value
        for name, value in zip(VIDEO_PIPELINE_ARGUMENT_NAMES, values)
        if name in supported_parameters
    }
    call_kwargs.update({
        name: value
        for name, value in kwargs.items()
        if name in supported_parameters
    })
    result = execute_video(**call_kwargs)
    if isinstance(result, (list, tuple)) and len(result) >= 8:
        result = list(result)
        # The pipeline's secondary outputs are diagnostic comparisons that
        # include the driving person. Keep them hidden in the production UI.
        result[2] = gr.update(visible=diagnostic_mode)
        result[6] = gr.update(visible=diagnostic_mode)
        return tuple(result)
    return result


def gpu_wrapped_execute_image(*args, **kwargs):
    return gradio_pipeline.execute_image(*args, **kwargs)


def execute_full_body_animation(
    source_image,
    driving_video,
    resolution=512,
    chunk_size=16,
    frames_overlap=4,
    num_inference_steps=15,
    fps=15,
    seed=42,
    confirm_full_body_framing=False,
):
    """Render the target-owned full body through the isolated MimicMotion backend."""
    if not source_image:
        raise gr.Error("Upload a full-body target image first.", duration=6)
    if not driving_video:
        raise gr.Error(
            "Record or upload a webcam pose video first. Full-body mode renders offline.",
            duration=6,
        )
    input_report = inspect_full_body_inputs(source_image, driving_video)
    if not confirm_full_body_framing:
        raise gr.Error(
            "Review the full-body input check, confirm the target shows the "
            "head, full body, and both hands, then render again. "
            + " ".join(input_report.warnings),
            duration=12,
        )
    try:
        options = FullBodyRenderOptions(
            resolution=int(resolution),
            chunk_size=int(chunk_size),
            frames_overlap=int(frames_overlap),
            num_inference_steps=int(num_inference_steps),
            fps=int(fps),
            seed=int(seed),
        )
        output = full_body_pipeline.render(
            source_image,
            driving_video,
            Path("results") / "full_body",
            options,
        )
    except FullBodyAnimationCancelled as exc:
        return gr.update(), f"**Render cancelled.** {exc}"
    except FullBodyAnimationError as exc:
        raise gr.Error(str(exc), duration=12) from exc
    except (OSError, ValueError) as exc:
        raise gr.Error(f"Full-body render could not start: {exc}", duration=10) from exc
    return str(output), f"Completed full-body render: `{output}`"


def cancel_full_body_animation():
    """Cancel only the offline full-body child process."""
    if full_body_pipeline.cancel_active_render():
        return gr.update(), "Cancellation requested. Stopping MimicMotion…"
    return gr.update(), "No full-body render is currently running."


def validate_full_body_inputs(source_image, driving_video):
    """Return input checks and lightweight driver previews before rendering."""
    report = inspect_full_body_inputs(source_image, driving_video)
    previews = extract_driver_preview_frames(driving_video)
    preview_note = (
        ""
        if previews or not driving_video
        else "\n\n- Driver frames could not be decoded. Re-encode the clip as a playable MP4."
    )
    return report.to_markdown() + preview_note, source_image, previews


def generate_local_photo_avatar(source_image, texture_resolution):
    """Create and rig a local photo avatar as one user-facing operation."""
    if not source_image:
        raise gr.Error("Upload a reference photo first.", duration=8)
    try:
        result = generate_rigged_local_photo_avatar(
            source_image,
            texture_resolution,
        )
    except LocalPhotoTo3DError as exc:
        return (
            None,
            None,
            "### Avatar preparation did not finish\n\n" + str(exc),
        )
    peak_memory = (
        f" Peak model-allocated VRAM: {result.peak_allocated_vram_mb:.0f} MB."
        if result.peak_allocated_vram_mb is not None
        else " Stable Fast 3D did not report its peak model memory."
    )
    return (
        result.model_path,
        result.model_path,
        f"**Your photo avatar is prepared and loaded in the live preview below.** "
        f"Reconstruction: {result.reconstruction_seconds:.1f}s at "
        f"{texture_resolution}px.{peak_memory} "
        "This is model-allocated memory, not total GPU memory. "
        "The model and rig are prepared locally; no separate Blender action or GLB "
        "reload is needed. Select **Start webcam tracking** in the live preview. "
        "Use one clear, upright full-body photo with the face, arms, and legs visible. "
        "A single photo cannot recover hidden sides or guarantee natural joint bending.",
    )


def change_animal_model(is_animal):
    global gradio_pipeline
    gradio_pipeline.clean_models()
    gradio_pipeline.init_models(is_animal=is_animal)


# assets
example_portrait_dir = "assets/examples/source"
example_video_dir = "assets/examples/driving"
#################### interface logic ####################

# Define components first
eye_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target eyes-open ratio")
lip_retargeting_slider = gr.Slider(minimum=0, maximum=0.8, step=0.01, label="target lip-open ratio")
retargeting_input_image = gr.Image(type="filepath")
output_image = gr.Image(format="png", type="numpy")
output_image_paste_back = gr.Image(format="png", type="numpy")

CYBERPUNK_CSS = r"""
:root {
    --cyber-bg: #070b13;
    --cyber-panel: #0e1624;
    --cyber-panel-strong: #111d2d;
    --cyber-line: rgba(115, 236, 255, 0.20);
    --cyber-line-hot: rgba(255, 64, 177, 0.48);
    --cyber-cyan: #65efff;
    --cyber-cyan-soft: #a8f6ff;
    --cyber-pink: #ff4db8;
    --cyber-pink-bright: #ff72c8;
    --cyber-text: #e7f5ff;
    --cyber-muted: #7d9aae;
    --cyber-shadow: 0 22px 70px rgba(0, 0, 0, 0.42);
}

body, .gradio-container {
    background:
        radial-gradient(circle at 78% 5%, rgba(0, 214, 255, 0.10), transparent 30rem),
        radial-gradient(circle at 10% 35%, rgba(255, 0, 153, 0.07), transparent 25rem),
        var(--cyber-bg) !important;
    color: var(--cyber-text) !important;
    font-family: "Segoe UI", system-ui, sans-serif !important;
}

.gradio-container {
    max-width: 1680px !important;
    padding: 28px clamp(16px, 3vw, 52px) 52px !important;
}

#cyber-header {
    border: 1px solid var(--cyber-line);
    border-radius: 22px;
    background: linear-gradient(120deg, rgba(17, 29, 45, 0.96), rgba(11, 19, 32, 0.88));
    box-shadow: var(--cyber-shadow), inset 0 1px 0 rgba(255,255,255,0.06);
    padding: 28px clamp(22px, 4vw, 64px);
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}

#cyber-header::after {
    content: "";
    position: absolute;
    inset: auto -10% -70% 35%;
    height: 210px;
    background: radial-gradient(ellipse, rgba(101, 239, 255, 0.18), transparent 68%);
    pointer-events: none;
}

.cyber-eyebrow, .cyber-kicker {
    color: var(--cyber-cyan);
    font-size: 0.70rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

#cyber-header h1 {
    color: var(--cyber-text);
    font-size: clamp(1.8rem, 4vw, 3.6rem);
    letter-spacing: -0.045em;
    line-height: 0.98;
    margin: 10px 0;
    position: relative;
    z-index: 1;
}

#cyber-header p {
    color: var(--cyber-muted);
    margin: 0;
    max-width: 760px;
    position: relative;
    z-index: 1;
}

#cyber-alerts {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 18px;
}

.cyber-alert {
    border: 1px solid var(--cyber-line);
    border-left: 3px solid var(--cyber-cyan);
    border-radius: 12px;
    background: rgba(14, 22, 36, 0.80);
    color: var(--cyber-cyan-soft);
    font-size: 0.86rem;
    line-height: 1.45;
    padding: 12px 15px;
}

.cyber-alert.hot {
    border-color: var(--cyber-line-hot);
    border-left-color: var(--cyber-pink);
    color: #ffd8ed;
}

#cyber-workspace {
    align-items: stretch;
    gap: 18px;
}

#control-sidebar, #preview-panel {
    border: 1px solid var(--cyber-line);
    border-radius: 18px;
    background: rgba(14, 22, 36, 0.88);
    box-shadow: var(--cyber-shadow), inset 0 1px 0 rgba(255,255,255,0.035);
    padding: 18px !important;
}

#control-sidebar {
    min-width: 330px;
}

#preview-panel {
    min-width: 0;
}

.cyber-section-title {
    color: var(--cyber-text);
    font-size: 1.0rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin: 6px 0 13px;
}

.cyber-section-title span {
    color: var(--cyber-cyan);
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 0.68rem;
    letter-spacing: 0.14em;
    margin-right: 9px;
}

.cyber-note {
    border: 1px solid rgba(101, 239, 255, 0.13);
    border-radius: 10px;
    color: var(--cyber-muted);
    font-size: 0.78rem;
    line-height: 1.45;
    padding: 10px 12px;
    margin: 10px 0 14px;
}

#portrait-uploader, #webcam-input, #source-video-input {
    border: 1px solid rgba(101, 239, 255, 0.25) !important;
    border-radius: 13px !important;
    overflow: hidden;
}

#portrait-uploader .upload-container,
#webcam-input .upload-container,
#source-video-input .upload-container {
    min-height: 205px;
    background: linear-gradient(145deg, rgba(101, 239, 255, 0.07), rgba(255, 77, 184, 0.04));
}

#preview-header {
    align-items: center;
    display: flex;
    justify-content: space-between;
    margin-bottom: 12px;
}

#preview-header .status {
    border: 1px solid rgba(101, 239, 255, 0.3);
    border-radius: 999px;
    color: var(--cyber-cyan);
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    padding: 6px 10px;
    text-transform: uppercase;
}

#driving-tabs {
    border: 1px solid rgba(101, 239, 255, 0.13);
    border-radius: 14px;
    padding: 10px;
}

#driving-tabs button.selected, #driving-tabs button[aria-selected="true"] {
    color: var(--cyber-cyan) !important;
    border-color: var(--cyber-cyan) !important;
}

#animation-controls {
    border: 1px solid rgba(255, 77, 184, 0.22);
    border-radius: 14px;
    background: rgba(255, 77, 184, 0.035);
    margin-top: 14px;
    padding: 12px;
}

#animate-button button, #retarget-button button {
    background: linear-gradient(100deg, var(--cyber-pink), #b62dff) !important;
    border: 0 !important;
    border-radius: 11px !important;
    box-shadow: 0 0 24px rgba(255, 77, 184, 0.27);
    color: #fff !important;
    font-weight: 850 !important;
    letter-spacing: 0.03em;
    min-height: 52px;
}

#animate-button button:hover, #retarget-button button:hover {
    box-shadow: 0 0 34px rgba(255, 77, 184, 0.48);
    transform: translateY(-1px);
}

#output-video, #output-video-secondary {
    border: 1px solid rgba(101, 239, 255, 0.27) !important;
    border-radius: 14px !important;
    background: #05080e !important;
    min-height: 390px;
    overflow: hidden;
}

#output-video video, #output-video-secondary video {
    background: #05080e;
    min-height: 360px;
    object-fit: contain;
}

#voice-shifter {
    border: 1px solid rgba(255, 77, 184, 0.28);
    border-radius: 13px;
    background: linear-gradient(145deg, rgba(255, 77, 184, 0.08), rgba(101, 239, 255, 0.04));
    margin-top: 14px;
    padding: 14px;
}

#voice-shifter h3 {
    color: var(--cyber-text);
    font-size: 0.92rem;
    margin: 0 0 4px;
}

#voice-shifter p {
    color: var(--cyber-muted);
    font-size: 0.76rem;
    line-height: 1.4;
    margin: 0 0 11px;
}

#voice-shifter label {
    color: var(--cyber-text);
    display: block;
    font-size: 0.78rem;
    font-weight: 700;
    margin: 0 0 6px;
}

#voice-shifter select {
    background: #101b29;
    border: 1px solid rgba(101, 239, 255, 0.28);
    border-radius: 9px;
    color: var(--cyber-text);
    margin-bottom: 9px;
    min-height: 40px;
    padding: 8px 10px;
    width: 100%;
}

#voice-shifter select:disabled,
#voice-shifter button:disabled {
    cursor: not-allowed;
    opacity: 0.55;
}

#voice-shifter .voice-device-help,
#voice-shifter .voice-device-status {
    color: var(--cyber-muted);
    font-size: 0.72rem;
    line-height: 1.4;
}

#voice-shifter .voice-device-status {
    margin: 7px 0 10px;
}

#voice-shifter button {
    background: transparent;
    border: 1px solid var(--cyber-pink);
    border-radius: 9px;
    color: #ffd8ed;
    cursor: pointer;
    font-weight: 750;
    padding: 9px 12px;
    width: 100%;
}

#voice-shifter button:hover {
    background: rgba(255, 77, 184, 0.16);
}

#voice-shifter button.stop {
    border-color: var(--cyber-cyan);
    color: var(--cyber-cyan-soft);
}

#voice-shifter-status {
    color: var(--cyber-muted);
    font-family: ui-monospace, SFMono-Regular, Consolas, monospace;
    font-size: 0.68rem;
    margin-top: 9px;
}

footer { display: none !important; }

@media (max-width: 900px) {
    #cyber-alerts { grid-template-columns: 1fr; }
    #cyber-workspace { flex-direction: column !important; }
    #control-sidebar { min-width: 0; }
    #output-video, #output-video-secondary { min-height: 260px; }
    #output-video video, #output-video-secondary video { min-height: 240px; }
}
"""


CYBER_HEADER_HTML = """
<header id="cyber-header">
    <div class="cyber-eyebrow">FasterLivePortrait / local inference console</div>
    <h1>Your portrait. Your motion.</h1>
    <p>Render a portrait from recorded motion, track a 3D avatar live, or work with an offline full-body clip. Choose a workspace in the navigation.</p>
</header>
"""


WORKSPACE_NAVIGATION_HTML = """
<nav aria-label="Avatar workspaces">
    <div class="workspace-brand">FasterLivePortrait<small>Local avatar studio</small></div>
    <div id="workspace-tabs" role="tablist" aria-label="Workspaces" aria-orientation="vertical">
        <button type="button" role="tab" id="nav-portrait" aria-controls="cyber-workspace" aria-selected="true" tabindex="0" data-workspace="cyber-workspace">
            <span>Portrait workspace</span><small>Recorded motion → render</small>
        </button>
        <button type="button" role="tab" id="nav-live" aria-controls="live-avatar-page" aria-selected="false" tabindex="-1" data-workspace="live-avatar-page">
            <span>Live 3D avatar</span><small>Continuous browser tracking</small>
        </button>
        <button type="button" role="tab" id="nav-full-body" aria-controls="full-body-page" aria-selected="false" tabindex="-1" data-workspace="full-body-page">
            <span>Offline full body</span><small>Check inputs → render clip</small>
        </button>
        <button type="button" role="tab" id="nav-retargeting" aria-controls="retargeting-page" aria-selected="false" tabindex="-1" data-workspace="retargeting-page">
            <span>Retargeting lab</span><small>Eye and lip adjustments</small>
        </button>
        <button type="button" role="tab" id="nav-advanced" aria-controls="advanced-page" aria-selected="false" tabindex="-1" data-workspace="advanced-page">
            <span>Advanced controls</span><small>Portrait motion and output</small>
        </button>
    </div>
    <p class="workspace-navigation-note">Switching keeps your inputs and running tasks. Stop webcam tracking or voice shifting in its workspace when finished.</p>
</nav>
"""


CYBER_ALERTS_HTML = """
<section id="cyber-alerts" aria-label="capture reminders">
    <div class="cyber-alert"><strong>FRAME LOCK //</strong> Keep your face centered, well lit, and fully inside the camera frame.</div>
    <div class="cyber-alert hot"><strong>PORTRAIT CHECK //</strong> Use a front-facing photo with visible eyes and mouth for cleaner dress and face sync.</div>
</section>
"""


AUDIO_WIDGET_HTML = """
<section id="voice-shifter" data-pitch-shift-widget>
    <h3>REAL-TIME VOICE SHIFTER</h3>
    <p>Browser microphone → +12 semitone pitch shift → local speakers. Headphones are recommended to prevent feedback.</p>
    <label for="voice-input-device">Input microphone</label>
    <select id="voice-input-device" data-voice-device disabled>
        <option value="">System default microphone</option>
    </select>
    <p class="voice-device-help">Browsers hide microphone names until you grant microphone permission. Click below to allow access and list devices; the temporary permission stream stops immediately. Stop voice shifting before changing microphones.</p>
    <button type="button" data-voice-list>Allow access &amp; list microphones</button>
    <div class="voice-device-status" data-voice-device-status aria-live="polite">Microphone names appear after permission is granted.</div>
    <button type="button" data-voice-start>Start feminine pitch</button>
    <div id="voice-shifter-status" data-voice-status>Microphone idle</div>
</section>
"""


# Gradio's HTML node hosts the controls; this launch-time script binds them and
# creates a native AudioWorklet so no server-side audio or external service is used.
js_func = r"""
() => {
    const bootWorkspaceNavigation = () => {
        const tabs = document.getElementById("workspace-tabs");
        if (!tabs || tabs.dataset.bound === "true") return;
        const buttons = Array.from(tabs.querySelectorAll("[data-workspace]"));
        const panels = buttons.map(button => document.getElementById(button.dataset.workspace));
        if (panels.some(panel => !panel)) return;
        tabs.dataset.bound = "true";
        const compact = window.matchMedia("(max-width: 767px)");
        const updateOrientation = () => {
            tabs.setAttribute("aria-orientation", compact.matches ? "horizontal" : "vertical");
            // Gradio 6 scopes media-rule selectors inside .contain; the outer
            // container is not a descendant. Apply its compact padding directly.
            const container = tabs.closest(".gradio-container");
            if (container) {
                if (compact.matches) container.style.setProperty("padding", "12px 12px 30px", "important");
                else container.style.removeProperty("padding");
            }
        };
        updateOrientation();
        compact.addEventListener("change", updateOrientation);
        panels.forEach((panel, index) => {
            panel.setAttribute("role", "tabpanel");
            panel.setAttribute("aria-labelledby", buttons[index].id);
            panel.tabIndex = 0;
        });
        const activate = (index, updateUrl = true) => {
            buttons.forEach((button, position) => {
                const selected = position === index;
                button.setAttribute("aria-selected", String(selected));
                button.tabIndex = selected ? 0 : -1;
                // Hide, never remove or re-create: Gradio state and event bindings stay intact.
                panels[position].classList.toggle("workspace-inactive", !selected);
                panels[position].hidden = !selected;
            });
            if (updateUrl) {
                const url = new URL(window.location);
                url.hash = "workspace-" + buttons[index].id.slice(4);
                window.history.replaceState(null, "", url);
                window.scrollTo({ top: 0, behavior: "instant" });
            }
        };
        const restore = () => {
            const index = buttons.findIndex(button => "#workspace-" + button.id.slice(4) === window.location.hash);
            activate(index < 0 ? 0 : index, false);
        };
        buttons.forEach((button, index) => {
            button.addEventListener("click", () => activate(index));
            button.addEventListener("keydown", event => {
                const previous = compact.matches ? "ArrowLeft" : "ArrowUp";
                const next = compact.matches ? "ArrowRight" : "ArrowDown";
                let target = index;
                if (event.key === previous) target = (index - 1 + buttons.length) % buttons.length;
                else if (event.key === next) target = (index + 1) % buttons.length;
                else if (event.key === "Home") target = 0;
                else if (event.key === "End") target = buttons.length - 1;
                else return;
                event.preventDefault();
                activate(target);
                buttons[target].focus();
                if (compact.matches) buttons[target].scrollIntoView({ block: "nearest", inline: "nearest" });
            });
        });
        window.addEventListener("hashchange", restore);
        restore();
    };
    const bootCyberpunkAudio = () => {
        const panel = document.querySelector("[data-pitch-shift-widget]");
        if (!panel || panel.dataset.bound === "true") return;
        panel.dataset.bound = "true";

        const button = panel.querySelector("[data-voice-start]");
        const deviceSelect = panel.querySelector("[data-voice-device]");
        const listButton = panel.querySelector("[data-voice-list]");
        const deviceStatus = panel.querySelector("[data-voice-device-status]");
        const status = panel.querySelector("[data-voice-status]");
        let context = null;
        let stream = null;
        let source = null;
        let shifter = null;
        let microphoneDevices = [];
        let listingDevices = false;
        let starting = false;

        const setStatus = (message, error = false) => {
            status.textContent = message;
            status.style.color = error ? "#ff8aca" : "";
        };

        const setDeviceStatus = (message, error = false) => {
            deviceStatus.textContent = message;
            deviceStatus.style.color = error ? "#ff8aca" : "";
        };

        const updateDeviceControls = () => {
            const busy = listingDevices || starting;
            deviceSelect.disabled = Boolean(stream) || busy || microphoneDevices.length === 0;
            listButton.disabled = Boolean(stream) || busy;
            button.disabled = busy;
        };

        const workletSource = `
            class CyberPitchShiftProcessor extends AudioWorkletProcessor {
                constructor() {
                    super();
                    this.size = 32768;
                    this.buffer = new Float32Array(this.size);
                    this.write = 0;
                    this.absoluteWrite = 0;
                    this.latency = 4096;
                    this.grainSize = 2048;
                    this.hop = this.grainSize / 2;
                    this.ratio = 2.0;
                    this.phaseA = 0;
                    this.phaseB = this.hop;
                    this.startA = 0;
                    this.startB = 0;
                    this.ready = false;
                }

                sampleAt(position) {
                    const wrapped = ((position % this.size) + this.size) % this.size;
                    const left = Math.floor(wrapped);
                    const right = (left + 1) % this.size;
                    const mix = wrapped - left;
                    return this.buffer[left] * (1 - mix) + this.buffer[right] * mix;
                }

                window(phase) {
                    return 0.5 - 0.5 * Math.cos((2 * Math.PI * phase) / this.grainSize);
                }

                process(inputs, outputs) {
                    const input = inputs[0];
                    const output = outputs[0];
                    if (!input || !input[0] || !output || !output[0]) return true;
                    const inChannel = input[0];
                    const outChannel = output[0];

                    for (let i = 0; i < outChannel.length; i++) {
                        const value = inChannel[i] || 0;
                        this.buffer[this.write] = value;
                        const absoluteInput = this.absoluteWrite;

                        if (!this.ready && absoluteInput > this.latency + this.grainSize) {
                            this.startA = absoluteInput - this.latency;
                            this.startB = this.startA - this.hop;
                            this.ready = true;
                        }

                        if (!this.ready) {
                            outChannel[i] = 0;
                        } else {
                            const readA = this.startA + this.phaseA * this.ratio;
                            const readB = this.startB + this.phaseB * this.ratio;
                            const ageA = absoluteInput - readA;
                            const ageB = absoluteInput - readB;
                            const validA = ageA > 8 && ageA < this.size - 8;
                            const validB = ageB > 8 && ageB < this.size - 8;
                            const weightA = this.window(this.phaseA);
                            const weightB = this.window(this.phaseB);
                            const valueA = validA ? this.sampleAt(readA) : 0;
                            const valueB = validB ? this.sampleAt(readB) : 0;
                            const weight = weightA + weightB || 1;
                            outChannel[i] = (valueA * weightA + valueB * weightB) / weight;
                        }

                        this.phaseA += 1;
                        this.phaseB += 1;
                        if (this.phaseA >= this.grainSize) {
                            this.phaseA -= this.hop;
                            this.startA = absoluteInput - this.latency;
                        }
                        if (this.phaseB >= this.grainSize) {
                            this.phaseB -= this.hop;
                            this.startB = absoluteInput - this.latency;
                        }
                        this.write = (this.write + 1) % this.size;
                        this.absoluteWrite += 1;
                    }

                    for (let channel = 1; channel < output.length; channel++) {
                        output[channel].set(outChannel);
                    }
                    return true;
                }
            }
            registerProcessor("cyber-pitch-shift", CyberPitchShiftProcessor);
        `;

        const stop = () => {
            if (source) source.disconnect();
            if (shifter) shifter.disconnect();
            if (stream) stream.getTracks().forEach(track => track.stop());
            if (context && context.state !== "closed") context.close();
            context = null;
            stream = null;
            source = null;
            shifter = null;
            button.textContent = "Start feminine pitch";
            button.classList.remove("stop");
            setStatus("Microphone idle");
            updateDeviceControls();
        };

        const permissionRecovery = () => {
            if (window.chrome?.webview || window.pywebview) {
                return "AvatarDesktop uses an embedded Edge WebView2 window, so there is no address-bar site menu. Allow any microphone prompt shown by the app. In Windows Settings > Privacy & security > Microphone, enable Microphone access and Let desktop apps access your microphone. Then click Allow access & list microphones and retry Start; if still blocked, close and reopen AvatarDesktop.";
            }
            return "In Chrome or Edge, open the site's controls beside the address bar and set Microphone to Allow. In Windows Settings > Privacy & security > Microphone, enable Microphone access and Let desktop apps access your microphone. Then click Allow access & list microphones, choose an input, and try Start again.";
        };

        const explainStartupFailure = (error, stage) => {
            const name = error?.name || "Error";
            const detail = error?.message || "Unknown startup error";
            if (stage === "audio-worklet" || /audio.?worklet/i.test(detail)) {
                return `Audio processing could not start (${name}: ${detail}). Use a current browser on HTTPS or localhost, check browser audio permissions, reload, then try Start again.`;
            }
            if (name === "NotAllowedError" || name === "PermissionDeniedError") {
                return `Microphone access was blocked (${name}). ${permissionRecovery()}`;
            }
            if (
                name === "NotFoundError" ||
                name === "DevicesNotFoundError" ||
                name === "OverconstrainedError" ||
                name === "ConstraintNotSatisfiedError"
            ) {
                return `No microphone input was found (${name}). Connect or enable a microphone, then check Windows Settings > System > Sound > Input. If the selected device was disconnected, reconnect it or choose another input with Refresh microphone list. Then try Start again.`;
            }
            if (name === "NotReadableError" || name === "TrackStartError") {
                return `The microphone is unavailable (${name}), possibly because another app is using it. Close apps such as Teams or Zoom that are capturing it, check Windows Settings > Privacy & security > Microphone > Let desktop apps access your microphone, then refresh the microphone list and try Start again.`;
            }
            if (name === "SecurityError") {
                return `The browser blocked microphone access (${name}). Open the app over HTTPS or localhost. ${permissionRecovery()}`;
            }
            return `Microphone startup failed (${detail}). Check browser site permission and Windows Settings > System > Sound > Input, refresh the microphone list, select an input, then try Start again.`;
        };

        const populateMicrophones = async () => {
            if (typeof navigator.mediaDevices.enumerateDevices !== "function") {
                throw new Error("This browser cannot list audio input devices.");
            }
            const previousDeviceId = deviceSelect.value;
            const devices = await navigator.mediaDevices.enumerateDevices();
            microphoneDevices = devices.filter(
                device =>
                    device.kind === "audioinput" &&
                    device.deviceId &&
                    device.deviceId !== "default"
            );
            listButton.textContent = "Refresh microphone list";
            const options = [new Option("System default microphone", "")];
            microphoneDevices.forEach((device, index) => {
                const label = (device.label || "").trim() || `Microphone ${index + 1} (name hidden)`;
                options.push(new Option(label, device.deviceId));
            });
            deviceSelect.replaceChildren(...options);
            if (microphoneDevices.some(device => device.deviceId === previousDeviceId)) {
                deviceSelect.value = previousDeviceId;
            } else {
                deviceSelect.value = "";
            }
            updateDeviceControls();
            if (microphoneDevices.length === 0) {
                setDeviceStatus(
                    "Windows listed no individual microphone inputs. Connect or enable a microphone in Settings > System > Sound > Input, then click Refresh microphone list. The system default remains available."
                );
                return;
            }
            const namesVisible = microphoneDevices.some(device => device.label.trim());
            setDeviceStatus(
                namesVisible
                    ? `Found ${microphoneDevices.length} microphone input${microphoneDevices.length === 1 ? "" : "s"}. Choose one, then start voice shifting.`
                    : `Found ${microphoneDevices.length} microphone input${microphoneDevices.length === 1 ? "" : "s"}, but the browser still hides their names. The numbered choices can still be selected.`
            );
        };

        listButton.addEventListener("click", async () => {
            if (stream || listingDevices || starting) return;
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                setDeviceStatus(
                    "This browser does not expose microphone capture. Use a current browser on HTTPS or localhost.",
                    true
                );
                return;
            }
            if (typeof navigator.mediaDevices.enumerateDevices !== "function") {
                setDeviceStatus(
                    "This browser cannot list microphone inputs. Use a current browser, then reload and retry.",
                    true
                );
                return;
            }

            let permissionStream = null;
            listingDevices = true;
            updateDeviceControls();
            setDeviceStatus("Requesting microphone permission…");
            try {
                permissionStream = await navigator.mediaDevices.getUserMedia({ audio: true });
                await populateMicrophones();
            } catch (error) {
                const name = error?.name || "Error";
                const message =
                    name === "NotAllowedError" ||
                    name === "PermissionDeniedError" ||
                    name === "NotFoundError" ||
                    name === "DevicesNotFoundError" ||
                    name === "NotReadableError" ||
                    name === "TrackStartError" ||
                    name === "SecurityError"
                        ? explainStartupFailure(error, "microphone")
                        : `Microphone permission was granted, but inputs could not be listed (${error?.message || name}). Refresh the list or check browser microphone permissions.`;
                setDeviceStatus(message, true);
            } finally {
                if (permissionStream) {
                    permissionStream.getTracks().forEach(track => track.stop());
                }
                listingDevices = false;
                updateDeviceControls();
            }
        });

        deviceSelect.addEventListener("change", () => {
            const selected = deviceSelect.selectedOptions[0]?.textContent;
            setDeviceStatus(
                deviceSelect.value
                    ? `Selected ${selected}. Start voice shifting to use this microphone.`
                    : "The system default microphone is selected."
            );
        });

        updateDeviceControls();

        button.addEventListener("click", async () => {
            if (stream) {
                stop();
                return;
            }
            if (listingDevices || starting) return;

            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                setStatus("This browser does not expose microphone capture.", true);
                return;
            }

            const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
            const hasAudioWorklet = Boolean(
                AudioContextCtor &&
                AudioContextCtor.prototype &&
                "audioWorklet" in AudioContextCtor.prototype &&
                typeof window.AudioWorkletNode === "function"
            );
            if (!hasAudioWorklet) {
                setStatus(
                    "This browser does not support the AudioWorklet needed for voice shifting. Use a current Chrome or Edge browser on HTTPS or localhost, then reload and try Start again.",
                    true
                );
                return;
            }

            let startupStage = "microphone";
            starting = true;
            updateDeviceControls();
            try {
                setStatus("Requesting microphone permission…");
                const audioConstraints = {
                    echoCancellation: false,
                    noiseSuppression: false,
                    autoGainControl: false
                };
                if (deviceSelect.value) {
                    audioConstraints.deviceId = { exact: deviceSelect.value };
                }
                stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
                startupStage = "audio-worklet";
                updateDeviceControls();
                context = new (window.AudioContext || window.webkitAudioContext)();
                const moduleUrl = URL.createObjectURL(
                    new Blob([workletSource], { type: "application/javascript" })
                );
                await context.audioWorklet.addModule(moduleUrl);
                URL.revokeObjectURL(moduleUrl);
                source = context.createMediaStreamSource(stream);
                shifter = new AudioWorkletNode(context, "cyber-pitch-shift");
                source.connect(shifter);
                shifter.connect(context.destination);
                await context.resume();
                button.textContent = "Stop voice shifter";
                button.classList.add("stop");
                const inputLabel = deviceSelect.selectedOptions[0]?.textContent || "System default microphone";
                setStatus(`Live // +12 semitones // ${inputLabel} // speaker output`);
            } catch (error) {
                stop();
                setStatus(explainStartupFailure(error, startupStage), true);
            } finally {
                starting = false;
                updateDeviceControls();
            }
        });

        window.addEventListener("beforeunload", stop, { once: true });
    };

    const refresh = () => {
        const url = new URL(window.location);
        if (url.searchParams.get("__theme") !== "dark") {
            url.searchParams.set("__theme", "dark");
            window.location.replace(url.href);
            return;
        }
        bootCyberpunkAudio();
        bootWorkspaceNavigation();
    };
    setTimeout(refresh, 0);
    setInterval(() => {
        bootCyberpunkAudio();
        bootWorkspaceNavigation();
    }, 1200);
}
"""


with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="pink",
            neutral_hue="slate",
            font=["Segoe UI", "system-ui", "sans-serif"],
        ),
        css=CYBERPUNK_CSS + "\n" + load_description(
            Path(__file__).parent / "assets" / "workspace_navigation.css"
        ),
        js=js_func,
        title="FasterLivePortrait // Local Console",
) as demo:
    gr.HTML(WORKSPACE_NAVIGATION_HTML, elem_id="workspace-navigation")
    gr.HTML(CYBER_HEADER_HTML)
    gr.HTML(CYBER_ALERTS_HTML)

    with gr.Row(elem_id="cyber-workspace"):
        with gr.Column(scale=1, min_width=330, elem_id="control-sidebar"):
            gr.HTML('<div class="cyber-section-title"><span>01</span>Source configuration</div>')

            webcam_toggle = gr.Checkbox(
                value=False,
                label="Use a recorded webcam clip",
                info="Record, stop recording, then Render Portrait. This is not continuous realtime tracking.",
            )
            webcam_input = gr.Video(
                sources=["webcam"],
                label="Recorded webcam motion",
                visible=False,
                elem_id="webcam-input",
            )

            with gr.Tabs():
                with gr.TabItem("Target portrait") as tab_image:
                    source_image_input = gr.Image(
                        type="filepath",
                        label="Target photo",
                        show_label=True,
                        elem_id="portrait-uploader",
                    )
                    gr.Examples(
                        examples=[
                            [osp.join(example_portrait_dir, "s9.jpg")],
                            [osp.join(example_portrait_dir, "s6.jpg")],
                            [osp.join(example_portrait_dir, "s10.jpg")],
                            [osp.join(example_portrait_dir, "s5.jpg")],
                            [osp.join(example_portrait_dir, "s7.jpg")],
                            [osp.join(example_portrait_dir, "s12.jpg")],
                        ],
                        inputs=[source_image_input],
                        cache_examples=False,
                    )
                with gr.TabItem("Target video") as tab_video:
                    source_video_input = gr.Video(
                        sources=["upload"],
                        label="Source video",
                        elem_id="source-video-input",
                    )
                    gr.Examples(
                        examples=[
                            [osp.join(example_video_dir, "d9.mp4")],
                            [osp.join(example_video_dir, "d10.mp4")],
                            [osp.join(example_video_dir, "d11.mp4")],
                            [osp.join(example_video_dir, "d12.mp4")],
                            [osp.join(example_video_dir, "d13.mp4")],
                            [osp.join(example_video_dir, "d14.mp4")],
                        ],
                        inputs=[source_video_input],
                        cache_examples=False,
                    )
            tab_selection = gr.Textbox(value="Image", visible=False)
            tab_image.select(lambda: "Image", None, tab_selection)
            tab_video.select(lambda: "Video", None, tab_selection)

            gr.HTML(
                '<div class="cyber-note"><strong>CAPTURE NOTE //</strong> '
                'Center the face, keep both eyes visible, and use a front-facing portrait '
                'for the cleanest dress and face-sync result.</div>'
            )

            with gr.Accordion("Source crop controls", open=False):
                with gr.Row():
                    flag_do_crop_input = gr.Checkbox(value=True, label="Crop source")
                    scale = gr.Number(value=2.3, label="Scale", minimum=1.8, maximum=3.2, step=0.05)
                with gr.Row():
                    vx_ratio = gr.Number(value=0.0, label="Crop X", minimum=-0.5, maximum=0.5, step=0.01)
                    vy_ratio = gr.Number(value=-0.125, label="Crop Y", minimum=-0.5, maximum=0.5, step=0.01)

            gr.HTML(AUDIO_WIDGET_HTML)

        with gr.Column(scale=3, min_width=600, elem_id="preview-panel"):
            gr.HTML(
                """
                <div id="preview-header">
                    <div>
                        <div class="cyber-kicker">02 / Render output</div>
                        <div class="cyber-section-title">Portrait preview</div>
                    </div>
                    <div class="status">RECORDED INPUT / RENDER</div>
                </div>
                """
            )

            with gr.Tabs(elem_id="driving-tabs"):
                with gr.TabItem("Driving video") as v_tab_video:
                    driving_video_input = gr.Video(
                        sources=["upload"],
                        label="Motion source",
                    )
                    gr.Examples(
                        examples=[
                            [osp.join(example_video_dir, "d9.mp4")],
                            [osp.join(example_video_dir, "d10.mp4")],
                            [osp.join(example_video_dir, "d11.mp4")],
                            [osp.join(example_video_dir, "d12.mp4")],
                            [osp.join(example_video_dir, "d13.mp4")],
                            [osp.join(example_video_dir, "d14.mp4")],
                        ],
                        inputs=[driving_video_input],
                        cache_examples=False,
                    )
                with gr.TabItem("Driving image") as v_tab_image:
                    driving_image_input = gr.Image(type="filepath", label="Motion image")
                    gr.Examples(
                        examples=[
                            [osp.join(example_portrait_dir, "s9.jpg")],
                            [osp.join(example_portrait_dir, "s6.jpg")],
                            [osp.join(example_portrait_dir, "s10.jpg")],
                            [osp.join(example_portrait_dir, "s5.jpg")],
                            [osp.join(example_portrait_dir, "s7.jpg")],
                            [osp.join(example_portrait_dir, "s12.jpg")],
                        ],
                        inputs=[driving_image_input],
                        cache_examples=False,
                    )
                with gr.TabItem("Driving pickle") as v_tab_pickle:
                    driving_pickle_input = gr.File(type="filepath", file_types=[".pkl"], label="Motion pickle")
                    gr.Examples(
                        examples=existing_examples(
                            osp.join(example_video_dir, "d2.pkl"),
                            osp.join(example_video_dir, "d8.pkl"),
                        ),
                        inputs=[driving_pickle_input],
                        cache_examples=False,
                    )
                with gr.TabItem("Driving audio") as v_tab_audio:
                    driving_audio_input = gr.Audio(
                        value=None,
                        type="filepath",
                        interactive=True,
                        show_label=False,
                        waveform_options=gr.WaveformOptions(sample_rate=24000),
                    )
                    gr.Examples(
                        examples=[[osp.join(example_video_dir, "a-01.wav")]],
                        inputs=[driving_audio_input],
                        cache_examples=False,
                    )
                with gr.TabItem("Driving text") as v_tab_text:
                    driving_text_input = gr.Textbox(
                        value="Hi, I am created by Faster LivePortrait!",
                        label="Driving text",
                    )
                    voice_dir = "checkpoints/Kokoro-82M/voices/"
                    voice_names = [
                        os.path.splitext(vname)[0]
                        for vname in os.listdir(voice_dir)
                        if vname.endswith(".pt")
                    ] if osp.isdir(voice_dir) else []
                    if not voice_names:
                        gr.Markdown(
                            "Kokoro voice assets are not installed. "
                            "Use video, image, pickle, or audio driving, "
                            "or download the optional Kokoro-82M model."
                        )
                    voice_name = gr.Dropdown(
                        choices=voice_names,
                        value=voice_names[0] if voice_names else None,
                        label="Voice name",
                        interactive=bool(voice_names),
                    )

            v_tab_selection = gr.Textbox(value="Video", visible=False)
            v_tab_video.select(lambda: "Video", None, v_tab_selection)
            v_tab_image.select(lambda: "Image", None, v_tab_selection)
            v_tab_pickle.select(lambda: "Pickle", None, v_tab_selection)
            v_tab_audio.select(lambda: "Audio", None, v_tab_selection)
            v_tab_text.select(lambda: "Text", None, v_tab_selection)

            gr.HTML('<div class="cyber-kicker">03 / Execute</div>')
            with gr.Row():
                process_button_animation = gr.Button(
                    "RENDER PORTRAIT",
                    variant="primary",
                    elem_id="animate-button",
                )
                process_button_reset = gr.ClearButton(
                    [
                        source_image_input,
                        source_video_input,
                        webcam_input,
                        webcam_toggle,
                        driving_pickle_input,
                        driving_video_input,
                        driving_image_input,
                    ],
                    value="Clear",
                )

            gr.HTML('<div class="cyber-kicker" style="margin-top:18px;">04 / Final signal</div>')
            output_video_i2v = gr.Video(
                autoplay=False,
                label="Final avatar / portrait appearance preserved",
                elem_id="output-video",
            )
            output_video_concat_i2v = gr.Video(
                autoplay=False,
                label="Diagnostic driving comparison",
                elem_id="output-video-secondary",
                visible=False,
            )
            output_image_i2i = gr.Image(
                format="png",
                type="numpy",
                label="Final avatar / portrait appearance preserved",
                visible=False,
            )
            output_image_concat_i2i = gr.Image(
                format="png",
                type="numpy",
                label="Diagnostic driving comparison",
                visible=False,
            )

    with gr.Accordion("Live 3D avatar (prototype)", open=True, elem_id="live-avatar-page"):
        gr.Markdown(
            "**Create your live avatar from one photo.** The app reconstructs a textured "
            "model and prepares its humanoid animation automatically; you do not need to "
            "open Blender or reload a rigged file. Camera frames stay in the browser. "
            "The tracker and avatar libraries are downloaded from their public CDNs on first use."
        )
        gr.Markdown(
            "### 1. Upload one full-body photo\n"
            "Choose an upright photo of one person with the face clearly visible and the "
            "arms and legs unobstructed. The app creates the textured model and its "
            "humanoid rig in one operation, then loads it into the live preview below. "
            "This local workflow does not send your photo to a 3D service.\n\n"
            "**Limits:** a single photo cannot show hidden sides or back. The generated "
            "shape and automatic joint weights are approximate; the first version tracks "
            "pose in place rather than walking or moving the feet through the room. "
            "Facial expressions and fingers depend on the generated model and its rig."
        )
        with gr.Row():
            photo_avatar_source = gr.Image(
                type="filepath",
                label="Full-body reference photo",
            )
            photo_avatar_texture_resolution = gr.Dropdown(
                choices=[512, 768, 1024],
                value=512,
                label="Texture detail",
                info="512 is the low-memory starting point; higher values preserve more texture detail.",
            )
        gr.Markdown(
            "Local NVIDIA model use is about 6 GB VRAM at the default setting. "
            "512px lowers texture memory and file size, but does not reduce "
            "the shape model's full inference memory. Stop webcam tracking and "
            "close GPU-heavy apps before generating."
        )
        photo_avatar_generate = gr.Button(
            "CREATE & LOAD MY LIVE AVATAR",
            variant="primary",
        )
        with gr.Row():
            photo_avatar_model = gr.Model3D(
                label="Prepared avatar preview",
                height=480,
            )
            photo_avatar_file = gr.File(
                label="Download prepared avatar GLB",
            )
        photo_avatar_status = gr.Markdown()
        photo_avatar_generate.click(
            fn=generate_local_photo_avatar,
            inputs=[photo_avatar_source, photo_avatar_texture_resolution],
            outputs=[photo_avatar_model, photo_avatar_file, photo_avatar_status],
            show_progress=True,
        )
        photo_avatar_file.change(
            fn=None,
            inputs=[photo_avatar_file],
            outputs=[],
            js=r"""(fileValue) => {
              const file = Array.isArray(fileValue) ? fileValue[0] : fileValue;
              if (!file?.url) return [];
              const frame = document.getElementById("live-avatar-frame");
              if (!frame?.contentWindow) return [];
              const modelUrl = new URL(file.url, window.location.href);
              if (
                modelUrl.origin !== window.location.origin ||
                !modelUrl.pathname.startsWith("/gradio_api/file=")
              ) return [];
              frame.contentWindow.postMessage(
                {
                  type: "load-photo-avatar",
                  url: modelUrl.href,
                  label: file.orig_name || "Prepared photo avatar"
                },
                window.location.origin
              );
              return [];
            }""",
        )
        gr.Markdown(
            "### 2. Start webcam tracking\n"
            "The prepared avatar loads automatically after creation. Press **Start webcam tracking** "
            "in the preview to grant camera access and begin movement tracking. The webcam tracker "
            "animates the avatar's head, chest, arms, and legs in place; the camera cannot recover "
            "hidden joints or true depth."
        )
        gr.HTML(
            '<iframe id="live-avatar-frame" title="Live 3D avatar" '
            'src="/gradio_api/file=assets/live_avatar.html" '
            'style="display:block;width:100%;height:760px;border:0;border-radius:14px;" '
            'allow="camera; fullscreen" loading="eager"></iframe>'
        )

    with gr.Accordion("Offline full-body animation", open=True, elem_id="full-body-page"):
        gr.Markdown(
            "Use a **full-body target image** and a recorded webcam pose video. "
            "This mode keeps the target's body, hands, clothing, and skin as the "
            "appearance source, then renders offline through DWPose + MimicMotion. "
            "It is separate from the realtime face pipeline and requires the "
            "optional Windows backend."
        )
        with gr.Row():
            full_body_source_image = gr.Image(
                type="filepath",
                label="Full-body target avatar",
            )
            full_body_driving_video = gr.Video(
                sources=["upload", "webcam"],
                label="Recorded webcam pose",
            )
        with gr.Row():
            full_body_check_button = gr.Button(
                "CHECK FRAMING & DURATION",
                variant="secondary",
            )
            full_body_validation_status = gr.Markdown(
                "Run the input check before starting the GPU render.",
            )
        with gr.Row():
            full_body_target_preview = gr.Image(
                label="Full-body target — check head, hands, feet, clothing, and margins",
                height=320,
                interactive=False,
            )
            full_body_driver_preview = gr.Gallery(
                label="Driver moments across the clip — select a frame to enlarge",
                height=320,
                columns=4,
                allow_preview=True,
                object_fit="contain",
                type="numpy",
                interactive=False,
            )
        gr.Markdown(
            "Up to four evenly spaced frames are shown across the driver clip; select "
            "any frame to inspect it at full size. Frames are scaled down only for "
            "display. Compare them with the target before rendering."
        )
        full_body_confirm_framing = gr.Checkbox(
            label=(
                "I confirm the target shows the head, full body, and both hands, "
                "and I reviewed any warnings above."
            ),
            value=False,
        )
        full_body_source_image.change(
            fn=lambda _value: gr.update(value=False),
            inputs=[full_body_source_image],
            outputs=[full_body_confirm_framing],
            show_progress=False,
            queue=False,
        )
        full_body_driving_video.change(
            fn=lambda _value: gr.update(value=False),
            inputs=[full_body_driving_video],
            outputs=[full_body_confirm_framing],
            show_progress=False,
            queue=False,
        )
        with gr.Row():
            full_body_resolution = gr.Dropdown(
                choices=[384, 512, 576],
                value=512,
                label="Resolution",
                info="384 uses less VRAM; 512 is the 8 GB default.",
            )
            full_body_chunk_size = gr.Number(
                value=16,
                label="Temporal chunk size",
                precision=0,
                interactive=False,
            )
            full_body_overlap = gr.Number(
                value=4,
                label="Frame overlap",
                precision=0,
            )
            full_body_steps = gr.Number(
                value=15,
                label="Denoising steps",
                precision=0,
            )
        with gr.Row():
            full_body_fps = gr.Number(value=15, label="Output FPS", precision=0)
            full_body_seed = gr.Number(value=42, label="Seed", precision=0)
        with gr.Row():
            full_body_render_button = gr.Button(
                "RENDER FULL-BODY AVATAR",
                variant="primary",
            )
            full_body_cancel_button = gr.Button(
                "CANCEL RENDER",
                variant="stop",
            )
        full_body_output = gr.Video(
            label="Full-body target-owned output",
            autoplay=False,
        )
        full_body_status = gr.Markdown(
            "Backend not started. Run `setup_full_body_windows.bat` on the Alienware.",
        )

    with gr.Accordion("Retargeting lab", open=True, elem_id="retargeting-page"):
        gr.Markdown("Fine-tune eye and lip openness against a target portrait.")
        with gr.Row():
            eye_retargeting_slider.render()
            lip_retargeting_slider.render()
        with gr.Row():
            with gr.Column():
                retargeting_input_image.render()
            with gr.Column():
                output_image.render()
            with gr.Column():
                output_image_paste_back.render()
        with gr.Row():
            process_button_retargeting = gr.Button(
                "RUN RETARGETING",
                variant="primary",
                elem_id="retarget-button",
            )
            process_button_reset_retargeting = gr.ClearButton(
                [
                    eye_retargeting_slider,
                    lip_retargeting_slider,
                    retargeting_input_image,
                    output_image,
                    output_image_paste_back,
                ],
                value="Clear retargeting",
            )

    with gr.Accordion("Advanced portrait controls", open=True, elem_id="advanced-page"):
        gr.Markdown(
            "These settings apply to **Portrait workspace** renders. Inputs and settings "
            "are kept when you switch workspaces. Return to Portrait workspace to render. "
            "The separate Live 3D and Offline full-body paths do not use these controls."
        )
        with gr.Accordion("Motion and render controls", open=True, elem_id="animation-controls"):
            with gr.Row():
                runtime_mode_control = gr.Radio(
                    ["onnx", "trt"],
                    value=runtime_mode,
                    label="Runtime mode",
                    info="Selected at launch. Restart with --mode onnx or --mode trt to change it.",
                    interactive=False,
                )
                diagnostic_mode = gr.Checkbox(
                    value=False,
                    label="Diagnostic mode",
                    info="Show the driving-person comparison output.",
                )
                strict_target_identity = gr.Checkbox(
                    value=True,
                    label="Strict target portrait identity",
                    info="The target photo supplies the intended person. Webcam-background mode may reveal the driver around the cutout if framing differs.",
                    interactive=False,
                )
                use_driver_background = gr.Checkbox(
                    value=False,
                    label="Use webcam background (experimental)",
                    info=(
                        "Locally removes the target photo background, then composites the target person "
                        "over each webcam/video frame. Requires the Stable Fast 3D environment; the first "
                        "use may download a local segmentation model. The driver's body may show around "
                        "the cutout when framing or pose differs."
                    ),
                )
                flag_virtual_camera_output = gr.Checkbox(
                    value=False,
                    label="Enable virtual camera output",
                    info=(
                        "Broadcast frames while this render runs. For continuous live webcam "
                        "output, launch camera.bat with --paste_back --virtual_camera."
                    ),
                )
            with gr.Row():
                flag_relative_input = gr.Checkbox(
                    value=True,
                    label="Identity-preserving relative motion",
                )
                flag_stitching = gr.Checkbox(value=True, label="Stitching")
                flag_remap_input = gr.Checkbox(
                    value=True,
                    label="Paste face back into target portrait",
                    info="Keeps the uploaded photo's body and background unless webcam-background mode is enabled. The photo is processed locally.",
                )
                flag_is_animal = gr.Checkbox(value=False, label="Animal model")
            with gr.Row():
                flag_color_match = gr.Checkbox(
                    value=True,
                    label="Enable color match",
                    info="Match synthesized face color to the target portrait before paste-back.",
                )
                stitching_blending_radius = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.35,
                    step=0.01,
                    label="Stitching margin / mask blending radius",
                )
            with gr.Row():
                driving_multiplier = gr.Number(value=1.0, label="Motion multiplier", minimum=0.0, maximum=2.0, step=0.02)
                cfg_scale = gr.Number(value=4.0, label="CFG scale", minimum=0.0, maximum=10.0, step=0.5)
                animation_region = gr.Radio(
                    ["exp", "pose", "lip", "eyes", "all"],
                    value="exp",
                    label="Animated region",
                    info="Expression-only keeps the portrait's hair, body, and framing unchanged.",
                )
            with gr.Row():
                flag_crop_driving_video_input = gr.Checkbox(value=False, label="Crop driving video")
                scale_crop_driving_video = gr.Number(value=2.2, label="Driving scale", minimum=1.8, maximum=3.2, step=0.05)
                vx_ratio_crop_driving_video = gr.Number(value=0.0, label="Driving X", minimum=-0.5, maximum=0.5, step=0.01)
                vy_ratio_crop_driving_video = gr.Number(value=-0.1, label="Driving Y", minimum=-0.5, maximum=0.5, step=0.01)
            with gr.Row():
                flag_video_editing_head_rotation = gr.Checkbox(value=False, label="Head rotation")
                driving_smooth_observation_variance = gr.Number(
                    value=1e-7,
                    label="Motion smooth strength",
                    minimum=1e-11,
                    maximum=1e-2,
                    step=1e-8,
                )

    webcam_toggle.change(
        lambda enabled: gr.update(visible=enabled),
        inputs=[webcam_toggle],
        outputs=[webcam_input],
    )
    flag_is_animal.change(change_animal_model, inputs=[flag_is_animal])

    process_button_retargeting.click(
        fn=gpu_wrapped_execute_image,
        inputs=[eye_retargeting_slider, lip_retargeting_slider, retargeting_input_image, flag_do_crop_input],
        outputs=[output_image, output_image_paste_back],
        show_progress=True,
    )
    process_button_animation.click(
        fn=gpu_wrapped_execute_video,
        inputs=[
            source_image_input,
            source_video_input,
            webcam_toggle,
            webcam_input,
            driving_video_input,
            driving_image_input,
            driving_pickle_input,
            driving_audio_input,
            driving_text_input,
            flag_relative_input,
            flag_do_crop_input,
            flag_remap_input,
            driving_multiplier,
            flag_stitching,
            flag_crop_driving_video_input,
            flag_video_editing_head_rotation,
            flag_is_animal,
            animation_region,
            scale,
            vx_ratio,
            vy_ratio,
            scale_crop_driving_video,
            vx_ratio_crop_driving_video,
            vy_ratio_crop_driving_video,
            driving_smooth_observation_variance,
            tab_selection,
            v_tab_selection,
            cfg_scale,
            voice_name,
            flag_color_match,
            stitching_blending_radius,
            diagnostic_mode,
            flag_virtual_camera_output,
            strict_target_identity,
            use_driver_background,
        ],
        outputs=[
            output_video_i2v,
            output_video_i2v,
            output_video_concat_i2v,
            output_video_concat_i2v,
            output_image_i2i,
            output_image_i2i,
            output_image_concat_i2i,
            output_image_concat_i2i,
        ],
        show_progress=True,
    )
    full_body_render_button.click(
        fn=execute_full_body_animation,
        inputs=[
            full_body_source_image,
            full_body_driving_video,
            full_body_resolution,
            full_body_chunk_size,
            full_body_overlap,
            full_body_steps,
            full_body_fps,
            full_body_seed,
            full_body_confirm_framing,
        ],
        outputs=[full_body_output, full_body_status],
        show_progress=True,
    )
    full_body_cancel_button.click(
        fn=cancel_full_body_animation,
        inputs=[],
        outputs=[full_body_output, full_body_status],
        show_progress=False,
        queue=False,
    )
    full_body_check_button.click(
        fn=validate_full_body_inputs,
        inputs=[full_body_source_image, full_body_driving_video],
        outputs=[
            full_body_validation_status,
            full_body_target_preview,
            full_body_driver_preview,
        ],
        show_progress=True,
    )


if __name__ == '__main__':
    demo.launch(
        server_port=args.port,
        share=False,
        server_name=args.host_ip,
        allowed_paths=["assets"],
    )
