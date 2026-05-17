import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
import os

def upscale_video(input_path, output_path):
    # Initialize the Real-ESRGAN model
    # Note: This requires the 'realesrgan' and 'basicsr' packages
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
        model=model,
        tile=400,
        tile_pad=10,
        pre_pad=0,
        half=True if torch.cuda.is_available() else False # Use GPU if available
    )

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) * 4)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 4)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Restoring and upscaling: {input_path}...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Upscale the frame
        output, _ = upsampler.enhance(frame, outscale=4)
        out.write(output)

    cap.release()
    out.release()
    print(f"Restoration complete. Saved to: {output_path}")

if __name__ == "__main__":
    # Ensure results folder exists
    if not os.path.exists('results'): os.makedirs('results')
    upscale_video("data/sample_low_res.mp4", "results/restored_high_res.mp4")