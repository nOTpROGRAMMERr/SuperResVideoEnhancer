# SuperResVideoEnhancer

# Video Enhancement Tool

A Python-based tool for enhancing video quality using **GFPGAN** and **CodeFormer** super-resolution models. This tool processes input video frames, enhances them, and seamlessly combines them with audio to produce a high-quality output video.

## Features
- **GFPGAN** for facial detail enhancement.
- **CodeFormer** for face and background enhancement.
- Combines enhanced video with original audio.
- Easy-to-use command-line interface.

## Installation

### 1. Clone the Repository
```bash
https://github.com/nOTpROGRAMMERr/SuperResVideoEnhancer.git
cd SuperResVideoEnhancer
```

## 2. Install Dependencies
Using Python Virtual Environment (Recommended)
Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
```
Install required packages:
```bash
pip install -r requirements.txt
```

## 3. Install Additional Tools
-**GFPGAN:**
```bash
git clone https://github.com/TencentARC/GFPGAN.git
cd GFPGAN
pip install -r requirements.txt
python setup.py develop
cd ..
```
-**CodeFormer:**
```bash
git clone https://github.com/sczhou/CodeFormer.git
cd CodeFormer
conda create -n codeformer python=3.8 -y
conda activate codeformer
pip install -r requirements.txt
python basicsr/setup.py develop
conda install -c conda-forge dlib  # Optional for face cropping
cd ..
```
## 4. Download Pretrained Models
-**GFPGAN Pretrained Model:**
```bash
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth -P experiments/pretrained_models
```
-**CodeFormer Pretrained Model:**
```bash
python CodeFormer/scripts/download_pretrained_models.py
```
## Usage
Run the script with the following command:
```bash
python x.py --superres [GFPGAN/CodeFormer] -iv <input_video_path> -ia <input_audio_path> -o <output_video_path>
```
## Example:
```bash
python x.py --superres GFPGAN -iv  Inputs/input.mp4 -ia Inputs/input.mp3 -o output.mp4
```




