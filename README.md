# Digital Restoration & Motion Capture: Preserving the Essence of Noh Theater via AI 🎭🤖

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-ff69b4)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-red)

**An interdisciplinary AI pipeline that restores historical Noh archives using GANs and quantifies *Ma* (間) and *Kamae* (構え) via MediaPipe kinematic tracking.**

---

## 📈 Results at a Glance

*GAN‑based restoration: original (left) vs. upscaled (right)*

<img width="800" alt="gan_restoration_comparison" src="https://via.placeholder.com/800x400?text=GAN+Restoration+Example+(add+your+image+here)" />

*MediaPipe pose overlay on a *Shakkyō* performance frame*

<img width="800" alt="mediapipe_pose_overlay" src="https://via.placeholder.com/800x400?text=33+Landmark+Overlay+(add+your+image+here)" />

*Center of Gravity trajectory over one *Ma* interval*

<img width="800" alt="cog_trajectory_ma" src="https://via.placeholder.com/800x400?text=CoG+Trajectory+Analysis+(add+your+image+here)" />

---

## 🧠 How It Works

### 1. AI Restomodding (GANs)
We use **Real‑ESRGAN** to upscale and denoise historical Noh performance videos. This preserves fine details of costumes, masks, and stage geometry that are essential for kinematic analysis.

### 2. Kinematic Motion Tracking (MediaPipe + OpenCV)
A **33‑landmark pose model** extracts body keypoints from each frame. We track:
- **Center of Gravity (CoG)** – per‑frame calculation to detect weight shifts  

  $$X_{CoG} = \frac{\sum m_i x_i}{\sum m_i}$$

- **Ma (間)** – quantified as the duration of minimal CoG displacement (the “pregnant pause”)
- **Kamae (構え)** – stability index from landmark variance (posture rigidity)

All processing is done with respect to **Kita school** aesthetics and the *Shakkyō* (石橋) play’s dramatic structure.

---

## 🏗️ Repository Structure
```
noh-motion-restoration/
├── analysis/ # CoG, Ma interval, Kamae stability
│ ├── compute_cog.py
│ └── ma_interval_detection.py
├── motion_tracking/ # MediaPipe + OpenCV pipeline
│ ├── pose_extractor.py
│ └── landmark_visualizer.py
├── restoration/ # Real‑ESRGAN upscaling
│ ├── esrgan_upscale.py
│ └── video_reassembly.py
├── data/ # Sample frames & landmarks (no full videos)
│ ├── sample_frames.zip
│ └── cog_trajectory.csv
├── results/ # Output GIFs, charts, overlay samples
├── docs/ # Ethics statement, technical abstract, BibTeX
├── requirements/ # requirements.txt, environment.yml
├── .github/workflows/ # (future) CI smoke test
├── README.md # This file (trilingual)
└── LICENSE
```

---

## 🚀 Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/johannesbambang/noh-motion-restoration.git
cd noh-motion-restoration
python -m venv .venv
# Activate virtual environment:
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements/requirements.txt
```

### 2. Run Motion Tracking on a Sample Frame
```bash
python motion_tracking/pose_extractor.py --input data/sample_frames.zip --output results/pose_overlay/
```

### 3. Compute Center of Gravity Trajectory
```bash
python analysis/compute_cog.py --landmarks data/landmarks_raw.json --output results/cog_plot.png
```

### 4. (Optional) GAN Restoration on a Low‑Res Clip
```bash
python restoration/esrgan_upscale.py --input path/to/lowres_video.mp4 --output results/restored.mp4
```

📚 Datasets & Fair Use
We use only promotional excerpts (fair use) for academic research:

Shiotsu Keisuke (Kita school) – YouTube

WCP2018 Shakkyō – YouTube

Full videos are not stored in this repository. See docs/ethics_statement.md for full data provenance and permissions.

📖 Trilingual Documentation
English – full technical details (above)

Bahasa Indonesia – ringkasan proyek dan petunjuk cepat (di bawah)

日本語 – プロジェクト概要とクイックスタート (下記)

🇮🇩 Bahasa Indonesia
Ringkasan Proyek
Proyek ini menggunakan GAN (Real‑ESRGAN) untuk merestorasi video arsip Noh dan MediaPipe untuk melacak 33 titik kerangka tubuh guna menganalisis Ma (jeda) dan Kamae (postur). Pusat Gravitasi dihitung per frame dengan rumus di atas.

Mulai Cepat
bash
git clone https://github.com/johannesbambang/noh-motion-restoration.git
cd noh-motion-restoration
pip install -r requirements/requirements.txt
python motion_tracking/pose_extractor.py --input data/sample_frames.zip
Catatan: Video penuh tidak disertakan. Lihat docs/ethics_statement.md untuk detail etika.

🇯🇵 日本語
プロジェクト概要
本プロジェクトは Real‑ESRGAN による能楽映像の修復と、MediaPipe による33点の身体ランドマーク追跡を組み合わせ、間 と 構え を定量化します。重心の計算式は上記の通りです。

クイックスタート
bash
git clone https://github.com/johannesbambang/noh-motion-restoration.git
cd noh-motion-restoration
pip install -r requirements/requirements.txt
python motion_tracking/pose_extractor.py --input data/sample_frames.zip
注記: 完全な動画は含まれません。倫理声明は docs/ethics_statement.md をご覧ください。

📄 License & Citation
This project is released under the MIT License.
If you use this code or data in your research, please cite:

bibtex
@software{wirawan2025noh,
  author = {Wirawan, Johannes Bambang},
  title = {Digital Restoration \& Motion Capture: Preserving the Essence of Noh Theater via AI},
  year = {2025},
  url = {https://github.com/johannesbambang/noh-motion-restoration},
  license = {MIT}
}
🤝 Contact & Event
Researcher: Johannes Bambang Wirawan

Affiliation: Master in Robotics, Tech Global University

Email: jbambangwirawan@gmail.com

Instagram: @johanneswirawan

Event: 1SKS Students Edition 2026 – Instagram Live @JF_Jakarta

<p align="center"> 🎭 <i>“Preserving the unseen pause between movements.”</i> 🎭<br> Made with 🔬 by <a href="https://github.com/johannesbambang">johannesbambang</a> </p> ```
