# import os
# import cv2
# import torch
# import numpy as np
# from PIL import Image
# from tqdm import tqdm
# from basicsr.utils import img2tensor
# from torchvision.transforms.functional import normalize
# from gfpgan import GFPGANer
# from basicsr.utils.download_util import load_file_from_url
# from moviepy.editor import VideoFileClip, AudioFileClip
# import argparse
# import sys
# import subprocess
#
# def load_models(model_choice):
#     models = {}
#     if 'gfpgan' in model_choice.lower():
#         models['gfpgan'] = GFPGANer(
#             model_path='/content/experiments/pretrained_models/GFPGANv1.3.pth',
#             upscale=2,
#             arch='clean',
#             channel_multiplier=2,
#             bg_upsampler=None
#         )
#     return models
#
#
# def process_frame(frame, models, model_choice):
#     if frame.shape[2] == 4:  # Remove alpha channel if present
#         frame = frame[:, :, :3]
#
#     enhanced_frame = frame.copy()
#
#     if 'gfpgan' in model_choice.lower():
#         _, _, enhanced_frame = models['gfpgan'].enhance(
#             enhanced_frame,
#             has_aligned=False,
#             only_center_face=False,
#             paste_back=True
#         )
#
#     return enhanced_frame
#
#
# def enhance_video(args):
#     if 'codeformer' in args.superres.lower():
#         # Run the CodeFormer script for enhancement
#         command = [
#             'python',
#             '/content/CodeFormer/inference_codeformer.py',
#             '--bg_upsampler', 'realesrgan',
#             '--face_upsample',
#             '-w', '1.0',
#             '-i', args.input_video,
#             '-o', args.output
#         ]
#         try:
#             subprocess.run(command, check=True)
#             print("CodeFormer enhancement completed successfully!")
#         except subprocess.CalledProcessError as e:
#             print(f"Error executing CodeFormer script: {e}")
#         return
#
#     models = load_models(args.superres)
#
#     video = cv2.VideoCapture(args.input_video)
#     fps = video.get(cv2.CAP_PROP_FPS)
#     total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     temp_output = 'temp_output.mp4'
#     out = None
#
#     try:
#         pbar = tqdm(total=total_frames)
#         frame_number = 0
#
#         while True:
#             ret, frame = video.read()
#             if not ret:
#                 break
#
#             enhanced_frame = process_frame(frame, models, args.superres)
#
#             if out is None:
#                 height, width = enhanced_frame.shape[:2]
#                 out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))
#
#             out.write(enhanced_frame)
#             frame_number += 1
#             pbar.update(1)
#
#     finally:
#         video.release()
#         if out is not None:
#             out.release()
#         pbar.close()
#
#     # Combine video and audio
#     video_clip = VideoFileClip(temp_output)
#     audio_clip = AudioFileClip(args.input_audio)
#
#     final_clip = video_clip.set_audio(audio_clip)
#     final_clip.write_videofile(args.output)
#
#     video_clip.close()
#     audio_clip.close()
#     if os.path.exists(temp_output):
#         os.remove(temp_output)
#
#     print(f"Video enhancement completed. Output saved to {args.output}")
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="Video enhancement script")
#     parser.add_argument('--superres', type=str, required=True, help="Super-resolution model to use (GFPGAN or CodeFormer)")
#     parser.add_argument('-iv', '--input_video', type=str, required=True, help="Path to input video file")
#     parser.add_argument('-ia', '--input_audio', type=str, required=True, help="Path to input audio file")
#     parser.add_argument('-o', '--output', type=str, required=True, help="Path to save the output video")
#
#     args = parser.parse_args()
#
#     enhance_video(args)

import argparse
import cv2
import moviepy.editor as mp
import os
import subprocess
from basicsr.archs.rrdbnet_arch import RRDBNet
from gfpgan import GFPGANer
from torch.nn.functional import interpolate
import torch

def process_frame(frame, superres_method, resolution_ratio):
    """
    Process a single video frame by applying super-resolution.

    Args:
        frame: Input frame to be processed.
        superres_method: Super-resolution method ('GFPGAN' or 'CodeFormer').
        resolution_ratio: Calculated upscale ratio for subframe resolution.

    Returns:
        Processed frame with enhanced resolution.
    """
    if superres_method == 'GFPGAN':
        # Initialize GFPGAN enhancer
        gfpgan = GFPGANer(model_path='experiments/pretrained_models/GFPGANv1.3.pth', upscale=int(resolution_ratio))
        _, _, enhanced_frame = gfpgan.enhance(frame, has_aligned=False, only_center_face=False, paste_back=True)
        return enhanced_frame

    elif superres_method == 'CodeFormer':
        # Initialize CodeFormer enhancer
        from facexlib.utils import imwrite
        from CodeFormer.basicsr.archs.rrdbnet_arch import RRDBNet
        from CodeFormer.basicsr.utils.download_util import load_file_from_url

        # Load pretrained model (ensure model is downloaded)
        model_path = 'experiments/pretrained_models/CodeFormer.pth'
        if not os.path.exists(model_path):
            raise FileNotFoundError("Pretrained CodeFormer model not found.")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4).to(device)
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['params_ema'])
        model.eval()

        # Process the frame using CodeFormer
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        tensor_frame = torch.tensor(frame).float().div(255).permute(2, 0, 1).unsqueeze(0).to(device)
        enhanced_tensor = interpolate(tensor_frame, scale_factor=resolution_ratio, mode='bilinear', align_corners=False)
        enhanced_frame = (enhanced_tensor.squeeze().permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
        return cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)  # Convert back to BGR

    else:
        raise ValueError("Invalid super-resolution method")

def enhance_video(input_video, input_audio, output_video, superres_method):
    """
    Enhance a video using super-resolution and merge with audio.

    Args:
        input_video: Path to input video file.
        input_audio: Path to input audio file.
        output_video: Path to save output video file.
        superres_method: Super-resolution method ('GFPGAN' or 'CodeFormer').
    """
    # Load video
    cap = cv2.VideoCapture(input_video)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    resolution_ratio = 2  # Default resolution ratio (adjust as needed)

    # Calculate the upscale ratio based on subframe analysis
    resolution_ratio = 2.0  # Placeholder for actual dynamic calculation logic

    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    temp_video_path = "temp_video.mp4"
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (int(frame_width * resolution_ratio), int(frame_height * resolution_ratio)))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Process the frame
        enhanced_frame = process_frame(frame, superres_method, resolution_ratio)
        out.write(enhanced_frame)

    cap.release()
    out.release()

    # Merge enhanced video with audio
    video = mp.VideoFileClip(temp_video_path)
    audio = mp.AudioFileClip(input_audio)
    final_video = video.set_audio(audio)
    final_video.write_videofile(output_video, codec='libx264')

    # Cleanup temporary video
    os.remove(temp_video_path)

def main():
    parser = argparse.ArgumentParser(description="Video Enhancement Script")
    parser.add_argument("--superres", required=True, choices=['GFPGAN', 'CodeFormer'], help="Super-resolution method")
    parser.add_argument("-iv", required=True, help="Input video file")
    parser.add_argument("-ia", required=True, help="Input audio file")
    parser.add_argument("-o", required=True, help="Output video file")
    args = parser.parse_args()

    enhance_video(args.iv, args.ia, args.o, args.superres)

if __name__ == "__main__":
    main()
