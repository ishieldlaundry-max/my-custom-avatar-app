# -*- coding: utf-8 -*-
# @Author  : wenshao
# @Email   : wenshaoguo1026@gmail.com
# @Project : FasterLivePortrait
# @FileName: run.py

"""
# video
 python run.py \
 --src_image assets/examples/driving/d13.mp4 \
 --dri_video assets/examples/driving/d11.mp4 \
 --cfg configs/trt_infer.yaml \
 --paste_back \
 --animal
# pkl
 python run.py \
 --src_image assets/examples/source/s12.jpg \
 --dri_video ./results/2024-09-13-081710/d0.mp4.pkl \
 --cfg configs/trt_infer.yaml \
 --paste_back \
 --animal
"""
import os
import argparse
import pdb
import subprocess
import ffmpeg
import cv2
import time
import numpy as np
import os
import datetime
import platform
import pickle
from collections import deque
from omegaconf import OmegaConf
from tqdm import tqdm
from colorama import Fore, Back, Style
from src.pipelines.faster_live_portrait_pipeline import FasterLivePortraitPipeline
from src.utils.utils import video_has_audio
from src.utils.virtual_camera import VirtualCameraBroadcaster
from src.utils.live_tracking_preview import LiveTrackingPreview, resolve_camera_source
from src.utils.live_portrait_effects import LivePortraitEffects
from src.utils.live_face_composite import (
    build_face_mask_from_image_landmarks,
    composite_photo_face_on_driver,
)
from src.utils.realtime_capture import LatestFrameCapture
from src.utils.driver_background import (
    DriverBackgroundError,
    composite_target_over_driver_background,
    extract_target_alpha,
)

if platform.system().lower() == 'windows':
    FFMPEG = "third_party/ffmpeg-7.0.1-full_build/bin/ffmpeg.exe"
else:
    FFMPEG = "ffmpeg"


def run_with_video(args):
    is_webcam, capture_source = resolve_camera_source(args.dri_video)
    face_swap_output = bool(args.realtime and is_webcam and not args.animal)
    if face_swap_output:
        print(
            Fore.RED
            + "Live webcam: camera body/background + animated photo face | C calibrate | H head | Q exit"
            + Style.RESET_ALL
        )
    else:
        print(
            Fore.RED
            + "Live: C calibrate | H head | S select shoulders | B shoulders off | L shading | Q exit"
            + Style.RESET_ALL
        )
    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = args.paste_back or face_swap_output
    if face_swap_output:
        infer_cfg.infer_params.flag_crop_driving_video = True

    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
    ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
    if not ret:
        print(f"no face in {args.src_image}! exit!")
        exit(1)
    use_driver_background = getattr(args, "use_driver_background", False)
    if face_swap_output and pipe.is_source_video:
        raise ValueError("Live webcam face swapping requires a still source photo.")
    if face_swap_output and use_driver_background:
        print(
            "Ignoring legacy whole-person background compositing: live webcam output "
            "now keeps the camera body/background and swaps only the photo face.",
            flush=True,
        )
        use_driver_background = False
    if use_driver_background and not args.paste_back:
        raise ValueError("--use_driver_background requires --paste_back.")
    if use_driver_background and pipe.is_source_video:
        raise ValueError("--use_driver_background requires a still --src_image.")
    target_alpha = None
    if use_driver_background:
        try:
            target_alpha = extract_target_alpha(args.src_image, pipe.src_imgs[0].shape[:2])
        except DriverBackgroundError as exc:
            raise RuntimeError(str(exc)) from exc
    if args.virtual_camera and not args.realtime:
        raise ValueError("--virtual_camera requires --realtime.")
    if args.virtual_camera and not args.paste_back and not face_swap_output:
        raise ValueError("--virtual_camera requires --paste_back for a clean composited frame.")
    if is_webcam:
        # read frame from camera if no driving video input
        print(f"Live driver: camera {capture_source}", flush=True)
        vcap = cv2.VideoCapture(capture_source)
        vcap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        vcap.set(cv2.CAP_PROP_FPS, 30)
        if not vcap.isOpened():
            print("no camera found! exit!")
            exit(1)
    else:
        vcap = cv2.VideoCapture(capture_source)
    use_clean_background = use_driver_background and args.realtime and is_webcam
    clean_background = None
    background_capture_error = ""
    if use_clean_background:
        print(
            "Room background: step out of view and press G to capture; "
            "step back in and press C to start tracking.",
            flush=True,
        )
    fps = int(vcap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30
    h, w = pipe.src_imgs[0].shape[:2]
    if face_swap_output:
        face_landmarks = pipe.src_infos[0][0][1]
        source_crop_to_image = pipe.src_infos[0][0][-1]
        if hasattr(source_crop_to_image, "detach"):
            source_crop_to_image = source_crop_to_image.detach().cpu().numpy()
        face_mask = build_face_mask_from_image_landmarks(
            face_landmarks,
            source_crop_to_image,
            (512, 512),
        )
    else:
        face_mask = None
    if args.virtual_camera and face_swap_output:
        camera_width = int(vcap.get(cv2.CAP_PROP_FRAME_WIDTH))
        camera_height = int(vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if camera_width > 0 and camera_height > 0:
            w, h = camera_width, camera_height
    virtual_camera = None
    realtime_capture = None
    if args.realtime and is_webcam:
        realtime_capture = LatestFrameCapture(vcap).start()
    if args.virtual_camera:
        try:
            virtual_camera = VirtualCameraBroadcaster(w, h, fps=30).start()
        except BaseException:
            if realtime_capture is not None:
                realtime_capture.close()
            else:
                vcap.release()
            raise
        print(f"Virtual camera active: {w}x{h} at 30 FPS")
    save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # render output video
    if not args.realtime:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vsave_crop_path = os.path.join(save_dir,
                                       f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-crop.mp4")
        vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512 * 2, 512))
        vsave_org_path = os.path.join(save_dir,
                                      f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-org.mp4")
        vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

    infer_times = deque(maxlen=300) if args.realtime else []
    motion_lst = []
    c_eyes_lst = []
    c_lip_lst = []

    frame_ind = 0
    live_preview = None
    try:
        if args.realtime and is_webcam:
            live_preview = LiveTrackingPreview(
                capture_source,
                allow_background_capture=use_clean_background,
                face_swap_mode=face_swap_output,
            )
            if not face_swap_output:
                live_preview.effects = LivePortraitEffects(
                    pipe.src_imgs[0], pipe.src_infos[0][0][1]
                )
            if use_clean_background:
                live_preview.status = (
                    "Step out of frame, then press G to capture the clean room background."
                )
            infer_cfg.infer_params.live_motion_smoothing = True
            if not face_swap_output:
                cv2.imshow("Render", cv2.cvtColor(pipe.src_imgs[0], cv2.COLOR_RGB2BGR))
        while realtime_capture is not None or vcap.isOpened():
            if realtime_capture is not None:
                ret, frame = realtime_capture.read(timeout=1.0)
                if not ret:
                    if realtime_capture.ended:
                        break
                    continue
            else:
                ret, frame = vcap.read()
            if not ret:
                break
            if live_preview is not None:
                live_preview.show_camera(frame)
                if not live_preview.poll():
                    break
                if use_clean_background and live_preview.background_capture_requested:
                    live_preview.background_capture_requested = False
                    background_capture_error = ""
                    try:
                        clean_background = live_preview.capture_clean_background(
                            realtime_capture,
                            frame,
                            pipe.src_imgs[0].shape[:2],
                        )
                    except DriverBackgroundError as exc:
                        background_capture_error = f"{exc} Press G to retry."
                        live_preview.status = background_capture_error
                        continue
                    finally:
                        live_preview.background_capture_requested = False
                    if clean_background is None:
                        break
                    live_preview.background_captured = True
                    live_preview.status = (
                        "Room captured. Step back in and press C to track; G to recapture."
                    )
                    cv2.imshow(
                        "Render",
                        cv2.cvtColor(pipe.src_imgs[0], cv2.COLOR_RGB2BGR),
                    )
                    continue
                effects = live_preview.effects
                if effects is not None:
                    effects.handle_key(live_preview.last_key)
                    if live_preview.last_key in (ord("s"), ord("S")):
                        live_preview.needs_reference = True
                        live_preview.calibration_requested = False
                    if effects.selecting:
                        live_preview.calibration_requested = False
                infer_cfg.infer_params.live_head_motion = live_preview.head_enabled
                if use_clean_background and clean_background is None:
                    cv2.imshow(
                        "Render",
                        cv2.cvtColor(pipe.src_imgs[0], cv2.COLOR_RGB2BGR),
                    )
                    continue
                if not live_preview.ready:
                    cv2.imshow(
                        "Render",
                        frame.copy()
                        if face_swap_output
                        else effects.selection_image(pipe.src_imgs[0]),
                    )
                    continue
                # Re-detect rather than following stale landmarks on the background.
                # This also reacquires the face after it leaves the camera view.
                pipe.src_lmk_pre = None
            t0 = time.time()
            first_frame = frame_ind == 0 or (
                live_preview is not None and live_preview.needs_reference
            )
            dri_crop, out_crop, out_org, dri_motion_info = pipe.run(
                frame,
                pipe.src_imgs[0],
                pipe.src_infos[0],
                first_frame=first_frame,
                realtime=args.realtime,
            )
            frame_ind += 1
            if live_preview is not None:
                live_preview.tracking_result(out_crop is not None)
                live_preview.show_camera(frame)
            if out_crop is None:
                print(f"no face in driving frame:{frame_ind}")
                if face_swap_output:
                    cv2.imshow("Render", frame.copy())
                continue
            live_face_output = None
            if face_swap_output:
                crop_to_driver = pipe.last_driver_crop_to_original
                if crop_to_driver is None or face_mask is None:
                    raise RuntimeError(
                        "The live face crop transform is unavailable; refusing to "
                        "replace the webcam body/background with the source photo."
                    )
                live_face_output = composite_photo_face_on_driver(
                    frame,
                    out_crop,
                    face_mask,
                    crop_to_driver,
                )
            elif live_preview is not None and live_preview.effects is not None:
                out_org = live_preview.effects.apply(
                    frame, out_org, dri_motion_info[0], reset=first_frame,
                    head_enabled=live_preview.head_enabled,
                )
            if target_alpha is not None:
                out_org = composite_target_over_driver_background(
                    out_org,
                    clean_background if use_clean_background else frame,
                    target_alpha,
                )

            if not args.realtime:
                motion_lst.append(dri_motion_info[0])
                c_eyes_lst.append(dri_motion_info[1])
                c_lip_lst.append(dri_motion_info[2])

            infer_times.append(time.time() - t0)
            if not args.realtime:
                dri_crop = cv2.resize(dri_crop, (512, 512))
                out_crop = np.concatenate([dri_crop, out_crop], axis=1)
                out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
                vout_crop.write(out_crop)
                out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                vout_org.write(out_org)
            else:
                if infer_cfg.infer_params.flag_pasteback:
                    if face_swap_output:
                        if virtual_camera is not None:
                            virtual_camera.send(
                                cv2.cvtColor(live_face_output, cv2.COLOR_BGR2RGB)
                            )
                        display_frame = live_face_output
                    else:
                        if virtual_camera is not None:
                            virtual_camera.send(out_org)
                        out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                        display_frame = out_org
                    cv2.imshow("Render", display_frame)
                else:
                    dri_crop = cv2.resize(dri_crop, (512, 512))
                    out_crop = np.concatenate([dri_crop, out_crop], axis=1)
                    out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
                    cv2.imshow('Render', out_crop)
                if live_preview is None and cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    except BaseException:
        if realtime_capture is not None:
            realtime_capture.close()
        else:
            vcap.release()
        if virtual_camera is not None:
            virtual_camera.close()
        if not args.realtime:
            vout_crop.release()
            vout_org.release()
        cv2.destroyAllWindows()
        raise
    if realtime_capture is not None:
        realtime_capture.close()
    else:
        vcap.release()
    if virtual_camera is not None:
        virtual_camera.close()
    if not args.realtime:
        vout_crop.release()
        vout_org.release()
        if video_has_audio(args.dri_video):
            vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_crop_path, "-i", args.dri_video,
                 "-b:v", "10M", "-c:v",
                 "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac",
                 "-pix_fmt", "yuv420p", vsave_crop_path_new, "-y", "-shortest"])
            vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_org_path, "-i", args.dri_video,
                 "-b:v", "10M", "-c:v",
                 "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac",
                 "-pix_fmt", "yuv420p", vsave_org_path_new, "-y", "-shortest"])

            print(vsave_crop_path_new)
            print(vsave_org_path_new)
        else:
            print(vsave_crop_path)
            print(vsave_org_path)
    else:
        cv2.destroyAllWindows()

    print(
        "inference median time: {} ms/frame, mean time: {} ms/frame".format(np.median(infer_times) * 1000,
                                                                            np.mean(infer_times) * 1000))
    if not args.realtime:
        # Offline renders can preserve every motion frame for later replay.
        template_dct = {
            'n_frames': len(motion_lst),
            'output_fps': fps,
            'motion': motion_lst,
            'c_eyes_lst': c_eyes_lst,
            'c_lip_lst': c_lip_lst,
        }
        template_pkl_path = os.path.join(
            save_dir, f"{os.path.basename(args.dri_video)}.pkl"
        )
        with open(template_pkl_path, "wb") as fw:
            pickle.dump(template_dct, fw)
        print(f"save driving motion pkl file at : {template_pkl_path}")


def run_with_pkl(args):
    infer_cfg = OmegaConf.load(args.cfg)
    infer_cfg.infer_params.flag_pasteback = args.paste_back

    pipe = FasterLivePortraitPipeline(cfg=infer_cfg, is_animal=args.animal)
    ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
    if not ret:
        print(f"no face in {args.src_image}! exit!")
        return
    with open(args.dri_video, "rb") as fin:
        dri_motion_infos = pickle.load(fin)

    fps = int(dri_motion_infos["output_fps"])
    h, w = pipe.src_imgs[0].shape[:2]
    save_dir = f"./results/{datetime.datetime.now().strftime('%Y-%m-%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    # render output video
    if not args.realtime:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vsave_crop_path = os.path.join(save_dir,
                                       f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-crop.mp4")
        vout_crop = cv2.VideoWriter(vsave_crop_path, fourcc, fps, (512, 512))
        vsave_org_path = os.path.join(save_dir,
                                      f"{os.path.basename(args.src_image)}-{os.path.basename(args.dri_video)}-org.mp4")
        vout_org = cv2.VideoWriter(vsave_org_path, fourcc, fps, (w, h))

    infer_times = []
    motion_lst = dri_motion_infos["motion"]
    c_eyes_lst = dri_motion_infos["c_eyes_lst"] if "c_eyes_lst" in dri_motion_infos else dri_motion_infos[
        "c_d_eyes_lst"]
    c_lip_lst = dri_motion_infos["c_lip_lst"] if "c_lip_lst" in dri_motion_infos else dri_motion_infos["c_d_lip_lst"]

    frame_num = len(motion_lst)
    for frame_ind in tqdm(range(frame_num)):
        t0 = time.time()
        first_frame = frame_ind == 0
        dri_motion_info_ = [motion_lst[frame_ind], c_eyes_lst[frame_ind], c_lip_lst[frame_ind]]
        out_crop, out_org = pipe.run_with_pkl(dri_motion_info_, pipe.src_imgs[0], pipe.src_infos[0],
                                              first_frame=first_frame)
        if out_crop is None:
            print(f"no face in driving frame:{frame_ind}")
            continue

        infer_times.append(time.time() - t0)
        # print(time.time() - t0)
        out_crop = cv2.cvtColor(out_crop, cv2.COLOR_RGB2BGR)
        if not args.realtime:
            vout_crop.write(out_crop)
            out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
            vout_org.write(out_org)
        else:
            if infer_cfg.infer_params.flag_pasteback:
                out_org = cv2.cvtColor(out_org, cv2.COLOR_RGB2BGR)
                cv2.imshow('Render,  Q > exit,  S > Stitching,  Z > RelativeMotion,  X > AnimationRegion,  C > CropDrivingVideo, KL > AdjustSourceScale, NM > AdjustDriverScale,  Space > Webcamassource,  R > SwitchRealtimeWebcamUpdate',out_org)
            else:
                # image show in realtime mode
                cv2.imshow('Render,  Q > exit,  S > Stitching,  Z > RelativeMotion,  X > AnimationRegion,  C > CropDrivingVideo, KL > AdjustSourceScale, NM > AdjustDriverScale,  Space > Webcamassource,  R > SwitchRealtimeWebcamUpdate', out_crop)
            # Press the 'q' key to exit the loop, r to switch realtime src_webcam update, spacebar to switch sourceisWebcam
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            # Key for Interesting Params    
            if k == ord('s'):
                infer_cfg.infer_params.flag_stitching = not infer_cfg.infer_params.flag_stitching
                print('flag_stitching:'+str(infer_cfg.infer_params.flag_stitching))
            if k == ord('z'):
                infer_cfg.infer_params.flag_relative_motion = not infer_cfg.infer_params.flag_relative_motion
                print('flag_relative_motion:'+str(infer_cfg.infer_params.flag_relative_motion))                
            if k == ord('x'):
                if infer_cfg.infer_params.animation_region == "all": infer_cfg.infer_params.animation_region = "exp", print('animation_region = "exp"')
                else:infer_cfg.infer_params.animation_region = "all", print('animation_region = "all"')
            if k == ord('c'):
                infer_cfg.infer_params.flag_crop_driving_video = not infer_cfg.infer_params.flag_crop_driving_video
                print('flag_crop_driving_video:'+str(infer_cfg.infer_params.flag_crop_driving_video))  
            if k == ord('v'):
                infer_cfg.infer_params.flag_pasteback = not infer_cfg.infer_params.flag_pasteback
                print('flag_pasteback:'+str(infer_cfg.infer_params.flag_pasteback)) 
                
            if k == ord('a'):
                infer_cfg.infer_params.flag_normalize_lip = not infer_cfg.infer_params.flag_normalize_lip
                print('flag_normalize_lip:'+str(infer_cfg.infer_params.flag_normalize_lip))  
            if k == ord('d'):
                infer_cfg.infer_params.flag_source_video_eye_retargeting = not infer_cfg.infer_params.flag_source_video_eye_retargeting
                print('flag_source_video_eye_retargeting:'+str(infer_cfg.infer_params.flag_source_video_eye_retargeting))  
            if k == ord('f'):
                infer_cfg.infer_params.flag_video_editing_head_rotation = not infer_cfg.infer_params.flag_video_editing_head_rotation
                print('flag_video_editing_head_rotation:'+str(infer_cfg.infer_params.flag_video_editing_head_rotation))                 
            if k == ord('g'):
                infer_cfg.infer_params.flag_eye_retargeting = not infer_cfg.infer_params.flag_eye_retargeting
                print('flag_eye_retargeting:'+str(infer_cfg.infer_params.flag_eye_retargeting)) 
                
            if k == ord('k'):
                infer_cfg.crop_params.src_scale -= 0.1
                ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
                print('src_scale:'+str(infer_cfg.crop_params.src_scale))                
            if k == ord('l'):
                infer_cfg.crop_params.src_scale += 0.1
                ret = pipe.prepare_source(args.src_image, realtime=args.realtime)
                print('src_scale:'+str(infer_cfg.crop_params.src_scale))  
            if k == ord('n'):
                infer_cfg.crop_params.dri_scale -= 0.1
                print('dri_scale:'+str(infer_cfg.crop_params.dri_scale))                
            if k == ord('m'):
                infer_cfg.crop_params.dri_scale += 0.1
                print('dri_scale:'+str(infer_cfg.crop_params.dri_scale))

    if not args.realtime:
        vout_crop.release()
        vout_org.release()
        if video_has_audio(args.dri_video):
            vsave_crop_path_new = os.path.splitext(vsave_crop_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_crop_path, "-i", args.dri_video,
                 "-b:v", "10M", "-c:v",
                 "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac",
                 "-pix_fmt", "yuv420p", vsave_crop_path_new, "-y", "-shortest"])
            vsave_org_path_new = os.path.splitext(vsave_org_path)[0] + "-audio.mp4"
            subprocess.call(
                [FFMPEG, "-i", vsave_org_path, "-i", args.dri_video,
                 "-b:v", "10M", "-c:v",
                 "libx264", "-map", "0:v", "-map", "1:a",
                 "-c:a", "aac",
                 "-pix_fmt", "yuv420p", vsave_org_path_new, "-y", "-shortest"])

            print(vsave_crop_path_new)
            print(vsave_org_path_new)
        else:
            print(vsave_crop_path)
            print(vsave_org_path)
    else:
        cv2.destroyAllWindows()

    print(
        "inference median time: {} ms/frame, mean time: {} ms/frame".format(np.median(infer_times) * 1000,
                                                                            np.mean(infer_times) * 1000))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Faster Live Portrait Pipeline')
    parser.add_argument('--src_image', required=False, type=str, default="assets/examples/source/s12.jpg",
                        help='source image')
    parser.add_argument('--dri_video', required=False, type=str, default="assets/examples/driving/d14.mp4",
                        help='driving video')
    parser.add_argument('--cfg', required=False, type=str, default="configs/onnx_infer.yaml", help='inference config')
    parser.add_argument('--realtime', action='store_true', help='realtime inference')
    parser.add_argument('--require-nvidia', action='store_true',
                        help='Refuse CPU/Intel inference; require NVIDIA CUDA')
    parser.add_argument('--animal', action='store_true', help='use animal model')
    parser.add_argument('--paste_back', action='store_true', default=False, help='paste back to origin image')
    parser.add_argument(
        '--use_driver_background',
        action='store_true',
        help='composite the target cutout over a captured webcam room or each video frame',
    )
    parser.add_argument('--virtual_camera', action='store_true',
                        help='broadcast composited realtime frames as a system webcam')
    args, unknown = parser.parse_known_args()
    if args.require_nvidia or os.name == "nt":
        from src.utils.nvidia_runtime import enforce_nvidia_runtime
        enforce_nvidia_runtime()

    if args.dri_video.endswith(".pkl"):
        run_with_pkl(args)
    else:
        run_with_video(args)
