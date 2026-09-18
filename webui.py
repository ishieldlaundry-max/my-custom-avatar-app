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
from src.pipelines.full_body_animation_pipeline import (
    FullBodyAnimationError,
    FullBodyAnimationPipeline,
    FullBodyRenderOptions,
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


runtime_mode = resolve_runtime_mode(args.mode)
if runtime_mode == "onnx":
    cfg_path = "configs/onnx_mp_infer.yaml" if args.use_mp else "configs/onnx_infer.yaml"
else:
    cfg_path = "configs/trt_mp_infer.yaml" if args.use_mp else "configs/trt_infer.yaml"
infer_cfg = OmegaConf.load(cfg_path)
gradio_pipeline = GradioLivePortraitPipeline(infer_cfg)


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
        output = FullBodyAnimationPipeline().render(
            source_image,
            driving_video,
            Path("results") / "full_body",
            options,
        )
    except FullBodyAnimationError as exc:
        raise gr.Error(str(exc), duration=12) from exc
    except (OSError, ValueError) as exc:
        raise gr.Error(f"Full-body render could not start: {exc}", duration=10) from exc
    return str(output), f"Completed full-body render: `{output}`"


def validate_full_body_inputs(source_image, driving_video):
    """Return the preflight report without starting generation."""
    return inspect_full_body_inputs(source_image, driving_video).to_markdown()


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
    font-family: Inter, ui-sans-serif, system-ui, sans-serif !important;
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
    <h1>Bring a face online.</h1>
    <p>Compose a source portrait, choose a motion signal, and render the final avatar locally on your NVIDIA workstation.</p>
</header>
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
    <button type="button" data-voice-start>Start feminine pitch</button>
    <div id="voice-shifter-status" data-voice-status>Microphone idle</div>
</section>
"""


# Gradio's HTML node hosts the controls; this launch-time script binds them and
# creates a native AudioWorklet so no server-side audio or external service is used.
js_func = r"""
() => {
    const bootCyberpunkAudio = () => {
        const panel = document.querySelector("[data-pitch-shift-widget]");
        if (!panel || panel.dataset.bound === "true") return;
        panel.dataset.bound = "true";

        const button = panel.querySelector("[data-voice-start]");
        const status = panel.querySelector("[data-voice-status]");
        let context = null;
        let stream = null;
        let source = null;
        let shifter = null;

        const setStatus = (message, error = false) => {
            status.textContent = message;
            status.style.color = error ? "#ff8aca" : "";
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
        };

        button.addEventListener("click", async () => {
            if (stream) {
                stop();
                return;
            }

            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                setStatus("This browser does not expose microphone capture.", true);
                return;
            }

            try {
                setStatus("Requesting microphone permission…");
                stream = await navigator.mediaDevices.getUserMedia({
                    audio: {
                        echoCancellation: false,
                        noiseSuppression: false,
                        autoGainControl: false
                    }
                });
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
                setStatus("Live // +12 semitones // speaker output");
            } catch (error) {
                stop();
                setStatus(`Microphone error: ${error.message}`, true);
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
    };
    setTimeout(refresh, 0);
    setInterval(bootCyberpunkAudio, 1200);
}
"""


with gr.Blocks(
        theme=gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="pink",
            neutral_hue="slate",
            font=[gr.themes.GoogleFont("Inter")],
        ),
        css=CYBERPUNK_CSS,
        js=js_func,
        title="FasterLivePortrait // Local Console",
) as demo:
    gr.HTML(CYBER_HEADER_HTML)
    gr.HTML(CYBER_ALERTS_HTML)

    with gr.Row(elem_id="cyber-workspace"):
        with gr.Column(scale=1, min_width=330, elem_id="control-sidebar"):
            gr.HTML('<div class="cyber-section-title"><span>01</span>Source configuration</div>')

            webcam_toggle = gr.Checkbox(
                value=False,
                label="Enable webcam capture",
                info="Use the local camera as motion only; the target portrait keeps its identity.",
            )
            webcam_input = gr.Video(
                sources=["webcam"],
                label="Webcam source",
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
                    <div class="status">LOCAL GPU PIPELINE</div>
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

            with gr.Accordion("Motion and render controls", open=False, elem_id="animation-controls"):
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
                        info="Locked: the driver supplies motion only; the target portrait supplies every visible pixel.",
                        interactive=False,
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
                        label="Preserve portrait body/background",
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

    with gr.Accordion("Offline full-body animation", open=False):
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
        full_body_confirm_framing = gr.Checkbox(
            label=(
                "I confirm the target shows the head, full body, and both hands, "
                "and I reviewed any warnings above."
            ),
            value=False,
        )
        with gr.Row():
            full_body_resolution = gr.Dropdown(
                choices=[512, 576],
                value=512,
                label="Resolution",
                info="Use 512 for an 8 GB RTX 4070.",
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
        full_body_render_button = gr.Button(
            "RENDER FULL-BODY AVATAR",
            variant="primary",
        )
        full_body_output = gr.Video(
            label="Full-body target-owned output",
            autoplay=False,
        )
        full_body_status = gr.Markdown(
            "Backend not started. Run `setup_full_body_windows.bat` on the Alienware.",
        )

    with gr.Accordion("Retargeting lab", open=False):
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
    full_body_check_button.click(
        fn=validate_full_body_inputs,
        inputs=[full_body_source_image, full_body_driving_video],
        outputs=[full_body_validation_status],
        show_progress=False,
    )


if __name__ == '__main__':
    demo.launch(
        server_port=args.port,
        share=False,
        server_name=args.host_ip,
    )
