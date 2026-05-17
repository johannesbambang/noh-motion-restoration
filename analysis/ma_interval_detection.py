import json
import os
import math
import numpy as np
import matplotlib.pyplot as plt

def detect_ma():
    json_path = '../results/final_landmarks.json'
    
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    W_HEAD, W_TRUNK, W_ARMS, W_THIGHS, W_CALVES = 0.08, 0.50, 0.10, 0.20, 0.12

    frames = []
    x_cog_list = []
    y_cog_list = []

    print("Extracting CoG data for Ma detection...")

    for frame_str, lms in data.items():
        if not frame_str.isdigit() or not lms or len(lms) < 33:
            continue
            
        frame_idx = int(frame_str)
        
        def get_avg(indices, axis='x'):
            return sum([lms[i][axis] for i in indices]) / len(indices)

        head_x, head_y = lms[0]['x'], lms[0]['y']
        trunk_x, trunk_y = get_avg([11, 12, 23, 24], 'x'), get_avg([11, 12, 23, 24], 'y')
        arms_x, arms_y = get_avg([13, 14, 15, 16], 'x'), get_avg([13, 14, 15, 16], 'y')
        thighs_x, thighs_y = get_avg([23, 24, 25, 26], 'x'), get_avg([23, 24, 25, 26], 'y')
        calves_x, calves_y = get_avg([25, 26, 27, 28], 'x'), get_avg([25, 26, 27, 28], 'y')

        X_cog = (head_x*W_HEAD + trunk_x*W_TRUNK + arms_x*W_ARMS + thighs_x*W_THIGHS + calves_x*W_CALVES)
        Y_cog = (head_y*W_HEAD + trunk_y*W_TRUNK + arms_y*W_ARMS + thighs_y*W_THIGHS + calves_y*W_CALVES)

        frames.append(frame_idx)
        x_cog_list.append(X_cog)
        y_cog_list.append(Y_cog)

    if not frames:
        print("No valid data found.")
        return

    # 1. Calculate 2D Velocity (Euclidean distance between CoG frames)
    velocities = [0.0] # First frame has 0 velocity
    for i in range(1, len(frames)):
        dx = x_cog_list[i] - x_cog_list[i-1]
        dy = y_cog_list[i] - y_cog_list[i-1]
        v = math.sqrt(dx**2 + dy**2)
        velocities.append(v)

    # 2. Apply a Moving Average Filter to smooth MediaPipe "jitter"
    window_size = 5
    smoothed_v = np.convolve(velocities, np.ones(window_size)/window_size, mode='same')

    # 3. Define the threshold for 'Ma' (Near Zero Movement)
    STILLNESS_THRESHOLD = 0.002
    
    # Generate the Ma Detection Plot
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Plot Velocity
    ax1.plot(frames, smoothed_v, color='#1976D2', label='Kinematic Velocity (Smoothed)')
    ax1.axhline(y=STILLNESS_THRESHOLD, color='black', linestyle=':', label='Ma Threshold')
    
    # Highlight the Ma (Stillness) intervals
    in_ma = False
    for i in range(len(frames)):
        if smoothed_v[i] < STILLNESS_THRESHOLD:
            if not in_ma:
                start_ma = frames[i]
                in_ma = True
        else:
            if in_ma:
                ax1.axvspan(start_ma, frames[i], color='#FFCDD2', alpha=0.5)
                in_ma = False
                
    # Close any trailing intervals
    if in_ma:
        ax1.axvspan(start_ma, frames[-1], color='#FFCDD2', alpha=0.5, label='Detected Ma (Pause)')

    # Fix duplicate labels in legend
    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax1.legend(by_label.values(), by_label.keys(), loc='upper right')

    plt.title('Detection of Ma (間) - Kinematic Velocity Analysis', fontsize=15, fontweight='bold')
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Velocity (Movement Intensity)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate the graph for the presentation
    plt.text(0.02, 0.95, '*Pink zones indicate intense dramatic tension (Zero CoG Variance)', 
             transform=ax1.transAxes, fontsize=10, style='italic', bbox=dict(facecolor='white', alpha=0.8))

    output_img = '../results/ma_detection_plot.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Success! Ma detection graph saved to: {output_img}")

if __name__ == "__main__":
    detect_ma()