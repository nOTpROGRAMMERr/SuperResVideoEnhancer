import os
import cv2
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from basicsr.utils import img2tensor
from torchvision.transforms.functional import normalize
from gfpgan import GFPGANer
from basicsr.utils.download_util import load_file_from_url
from moviepy.editor import VideoFileClip, AudioFileClip
import argparse
import sys
import subprocess

def load_models(model_choice):
    models = {}
    if 'gfpgan' in model_choice.lower():
        models['gfpgan'] = GFPGANer(
            model_path='experiments/pretrained_models/GFPGANv1.3.pth',
            upscale=2,
            arch='clean',
            channel_multiplier=2,
            bg_upsampler=None
        )
    return models


def process_frame(frame, models, model_choice):
    if frame.shape[2] == 4:  # Remove alpha channel if present
        frame = frame[:, :, :3]

    enhanced_frame = frame.copy()

    if 'gfpgan' in model_choice.lower():
        _, _, enhanced_frame = models['gfpgan'].enhance(
            enhanced_frame,
            has_aligned=False,
            only_center_face=False,
            paste_back=True
        )

    return enhanced_frame


def enhance_video(args):
    if 'codeformer' in args.superres.lower():
        # Run the CodeFormer script for enhancement
        command = [
            'python',
            'CodeFormer/inference_codeformer.py',
            '--bg_upsampler', 'realesrgan',
            '--face_upsample',
            '-w', '1.0',
            '-i', args.input_video,
            '-o', args.output
        ]
        try:
            subprocess.run(command, check=True)
            print("CodeFormer enhancement completed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error executing CodeFormer script: {e}")
        return

    models = load_models(args.superres)

    video = cv2.VideoCapture(args.input_video)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    temp_output = 'temp_output.mp4'
    out = None

    try:
        pbar = tqdm(total=total_frames)
        frame_number = 0

        while True:
            ret, frame = video.read()
            if not ret:
                break

            enhanced_frame = process_frame(frame, models, args.superres)

            if out is None:
                height, width = enhanced_frame.shape[:2]
                out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

            out.write(enhanced_frame)
            frame_number += 1
            pbar.update(1)

    finally:
        video.release()
        if out is not None:
            out.release()
        pbar.close()

    # Combine video and audio
    video_clip = VideoFileClip(temp_output)
    audio_clip = AudioFileClip(args.input_audio)

    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(args.output)

    video_clip.close()
    audio_clip.close()
    if os.path.exists(temp_output):
        os.remove(temp_output)

    print(f"Video enhancement completed. Output saved to {args.output}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video enhancement script")
    parser.add_argument('--superres', type=str, required=True, help="Super-resolution model to use (GFPGAN or CodeFormer)")
    parser.add_argument('-iv', '--input_video', type=str, required=True, help="Path to input video file")
    parser.add_argument('-ia', '--input_audio', type=str, required=True, help="Path to input audio file")
    parser.add_argument('-o', '--output', type=str, required=True, help="Path to save the output video")

    args = parser.parse_args()

    enhance_video(args)


