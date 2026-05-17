# 🎭 Digital Restoration & Motion Capture: Preserving the Essence of Noh Theater via AI

🌍 **Language / Bahasa / 言語**: [English](#english-en) | [Bahasa Indonesia](#bahasa-indonesia-id) | [日本語](#日本語-ja)

![Python](https://img.shields.io/badge/python-3.10-blue)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose-ff69b4)
![OpenCV](https://img.shields.io/badge/OpenCV-4.9-red)

---

## English (EN)

**An interdisciplinary AI pipeline that restores historical Noh archives using GANs and quantifies *Ma* (間) and *Kamae* (構え) via MediaPipe kinematic tracking.**

### 📈 Results at a Glance

*Kinematic Stability Analysis (Kamae): Proving 'Stillness in Motion'*
<img width="3036" height="1651" alt="cog_stability_plot" src="https://github.com/user-attachments/assets/6a832cfd-97f3-48ec-8815-e44418aed59b" />

*Detection of Ma (間): Quantifying Dramatic Tension via Kinematic Velocity*
<img width="3528" height="1884" alt="ma_detection_plot" src="https://github.com/user-attachments/assets/ce5d2843-61cf-43b1-a7ea-0e21dd082878" />

*MediaPipe pose overlay locked onto the Lion Dancer (Overcoming Costume Occlusion)*
<img width="640" height="360" alt="pose_overlay_sample" src="https://github.com/user-attachments/assets/c698ff51-8e7c-4052-8d06-a57551d6e476" />


*GAN‑based restoration (Real-ESRGAN)*
> *[Phase 2: Archival upscaling in progress. Restored visual comparisons coming soon.]*

### 🧠 How It Works
1. **AI Restomodding (GANs):** We use **Real‑ESRGAN** to upscale and denoise historical Noh performance videos. This preserves fine details of costumes, masks, and stage geometry.
2. **Kinematic Motion Tracking:** A custom **33‑landmark pose model** extracts body keypoints to track:
   - **Center of Gravity (CoG)** – per‑frame calculation to detect weight shifts.
   - **Ma (間)** – quantified as the duration of minimal CoG displacement (the "pregnant pause").
   - **Kamae (構え)** – stability index from landmark variance (posture rigidity).

### 🚀 Quick Start
```bash
git clone [https://github.com/johannesbambang/noh-motion-restoration.git](https://github.com/johannesbambang/noh-motion-restoration.git)
cd noh-motion-restoration
python -m venv noh_env
# Activate virtual environment (Windows): .\noh_env\Scripts\activate
pip install -r requirements.txt

# Run Pose Extraction
python motion_tracking/final_pose.py --input data/shakkyo_cropped_strict.mp4<img width="3036" height="1651" alt="cog_stability_plot" src="https://github.com/user-attachments/assets/bc945ed3-c09b-4ab5-b47a-404b8baae6f8" />

```

---

## Bahasa Indonesia (ID)

**Pipeline AI interdisipliner yang merestorasi arsip sejarah Noh menggunakan GAN dan mengukur Ma (間) serta Kamae (構え) melalui pelacakan kinematika MediaPipe.**

### 📈 Sekilas Hasil

*Analisis Stabilitas Kinematik (Kamae): Membuktikan 'Ketenangan dalam Gerakan'*
<img width="3036" height="1651" alt="cog_stability_plot" src="https://github.com/user-attachments/assets/6a832cfd-97f3-48ec-8815-e44418aed59b" />

*Deteksi Ma (間): Mengukur Ketegangan Dramatis melalui Kecepatan Kinematik*
<img width="3528" height="1884" alt="ma_detection_plot" src="https://github.com/user-attachments/assets/ce5d2843-61cf-43b1-a7ea-0e21dd082878" />

*Pelacakan Pose MediaPipe pada Penari Singa (Mengatasi Halangan Kostum)*
<img width="640" height="360" alt="pose_overlay_sample" src="https://github.com/user-attachments/assets/c698ff51-8e7c-4052-8d06-a57551d6e476" />

### 🧠 Cara Kerja
1. **Restorasi AI (GANs):** Kami menggunakan **Real‑ESRGAN** untuk meningkatkan resolusi (upscale) dan mengurangi noise pada video arsip pertunjukan Noh. Hal ini melestarikan detail kostum, topeng, dan geometri panggung.
2. **Pelacakan Gerak Kinematik:** Model pose dengan **33 titik tubuh (landmarks)** mengekstrak titik-titik penting tubuh untuk melacak:

- **Pusat Gravitasi (CoG)** – dihitung per frame untuk mendeteksi perpindahan berat badan.
- **Ma (間)** – diukur sebagai durasi pergerakan CoG paling minimal ("jeda dramatis").
- **Kamae (構え)** – indeks stabilitas dari variasi titik tubuh (kekakuan postur).

### 🚀 Mulai Cepat
```bash
git clone [https://github.com/johannesbambang/noh-motion-restoration.git](https://github.com/johannesbambang/noh-motion-restoration.git)
cd noh-motion-restoration
python -m venv noh_env
# Aktivasi virtual environment (Windows): .\noh_env\Scripts\activate
pip install -r requirements.txt

# Ekstraksi Pose
python motion_tracking/final_pose.py --input data/shakkyo_cropped_strict.mp4

```

---
## 日本語 (JA)

**GANを用いて歴史的な能楽アーカイブを修復し、MediaPipeの運動学トラッキングを通じて「間」と「構え」を定量化する学際的なAIパイプライン。**

### 📈 結果の概要
*運動学的安定性分析（構え）：「動の中の静」の証明*
<img width="3036" height="1651" alt="cog_stability_plot" src="https://github.com/user-attachments/assets/6a832cfd-97f3-48ec-8815-e44418aed59b" />

*「間」の検出：運動速度による劇的緊張の定量化*
<img width="3528" height="1884" alt="ma_detection_plot" src="https://github.com/user-attachments/assets/ce5d2843-61cf-43b1-a7ea-0e21dd082878" />

*獅子舞へのMediaPipeポーズオーバーレイ（衣装によるオクルージョンの克服）*
<img width="640" height="360" alt="pose_overlay_sample" src="https://github.com/user-attachments/assets/c698ff51-8e7c-4052-8d06-a57551d6e476" />

### 🧠 仕組み
1. **AIによる修復（GAN）:** **Real-ESRGAN** を使用して、歴史的な能楽映像をアップスケールし、ノイズを除去します。これにより、装束、面、舞台の精細なディテールが保存されます。

2. **運動学的モーショントラッキング:** カスタムの**33ランドマークポーズ**モデルが身体のキーポイントを抽出し、以下を追跡します：

- **重心（CoG）**: 体重移動を検出するためのフレームごとの計算。

- **間（Ma）**: 重心の移動が最小となる期間（劇的な休止）として定量化。

- **構え（Kamae）**: ランドマークの分散から算出される安定性指標（姿勢の剛性）。

### 🚀 クイックスタート
```bash
git clone [https://github.com/johannesbambang/noh-motion-restoration.git](https://github.com/johannesbambang/noh-motion-restoration.git)
cd noh-motion-restoration
python -m venv noh_env
# 仮想環境の有効化 (Windows): .\noh_env\Scripts\activate
pip install -r requirements.txt

# ポーズ抽出の実行
python motion_tracking/final_pose.py --input data/shakkyo_cropped_strict.mp4
```

---
## 📚 Datasets & Fair Use
We use only promotional excerpts for academic research under fair use:

- **Shiotsu Keisuke** (Kita school)
- **WCP2018 Shakkyō**

Full videos are not stored in this repository. See docs/ethics_statement.md for full data provenance and permissions.

## 📄 License & Citation

This project is released under the MIT License. If you use this code or data in your research, please cite:
```bash
@software{wirawan2026noh,
  author = {Wirawan, Johannes Bambang},
  title = {Digital Restoration & Motion Capture: Preserving the Essence of Noh Theater via AI},
  year = {2026},
  url = {[https://github.com/johannesbambang/noh-motion-restoration](https://github.com/johannesbambang/noh-motion-restoration)},
  license = {MIT}
}
```

---
## 🤝 Contact & Event
- **Researcher:** Johannes Bambang Wirawan
- **Affiliation:** Master in Robotics, Tech Global University
- **Event:** 1SKS Students Edition 2026 – Instagram Live @JF_Jakarta
