from pathlib import Path
from typing import Union

import cv2
import numpy as np
import pyrealsense2 as rs

from .paths.paths import EVAL, NAMES, TRAIN


def overlay_3d_keypoints_on_video(
    mp4_file: Union[str, Path],
    npy_file: Union[str, Path],
    bag_file: Union[str, Path],
    output_mp4: Union[str, Path],
):
    # Load 3D keypoints from .npy file
    keypoints_3d = np.load(npy_file)  # Shape: (n_frames, 17, 3)

    # Initialize RealSense pipeline to get camera intrinsics
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device_from_file(str(bag_file), repeat_playback=False)
    config.enable_stream(rs.stream.color)

    try:
        profile = pipeline.start(config)
        color_stream = profile.get_stream(rs.stream.color)
        video_profile = color_stream.as_video_stream_profile()
        intrinsics = video_profile.get_intrinsics()
        fps = video_profile.fps()
        width, height = intrinsics.width, intrinsics.height

        # Open input video
        cap = cv2.VideoCapture(str(mp4_file))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video file {mp4_file}")

        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_mp4), fourcc, fps, (width, height))

        frame_idx = 0
        while cap.isOpened() and frame_idx < len(keypoints_3d):
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB (if necessary)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get 3D keypoints for current frame
            frame_keypoints = keypoints_3d[frame_idx]

            # Project 3D points to 2D
            for keypoint in frame_keypoints:
                x, y, z = keypoint
                if z == 0:  # Skip invalid points
                    continue
                # Project 3D point to 2D pixel coordinates
                pixel = rs.rs2_project_point_to_pixel(intrinsics, [x, y, z])
                u, v = int(pixel[0]), int(pixel[1])

                # Draw circle at the projected point
                if 0 <= u < width and 0 <= v < height:
                    cv2.circle(frame_rgb, (u, v), 5, (0, 255, 0), -1)  # Green circle

            # Write frame to output video
            out.write(frame_rgb)
            frame_idx += 1

        print(f"Output video saved to {output_mp4}")

    except Exception as e:
        print(f"Error: {e}")
        raise e
    finally:
        cap.release()
        out.release()
        pipeline.stop()


def main(make_dir=True, eval=False):
    BASE = EVAL if eval else TRAIN
    if make_dir:
        BASE.check_tree()

    for person in NAMES:
        print(person)
        continue
        bn = person["samples"][0]["bag"]
        on = person["samples"][0]["out"]
        ev = person["samples"][0]["eval"]

        if eval and (ev is None or ev == ""):
            continue

        mp4_file = BASE.MP4 / f"{on}.mp4"
        npy_file = BASE.KP_3D / f"{on}.npy"
        bag_file = BASE.BAG / f"{bn}.bag"
        output_mp4 = BASE.MP4 / f"{on}_keypoints.mp4"  # Output video with keypoints

        if not output_mp4.exists():
            overlay_3d_keypoints_on_video(mp4_file, npy_file, bag_file, output_mp4)
        else:
            if not output_mp4.is_file():
                raise RuntimeError(f"Path '{output_mp4}' exists but is not a file!")


if __name__ == "__main__":
    main(make_dir=True, eval=False)
