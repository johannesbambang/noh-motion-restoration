import json
import matplotlib.pyplot as plt
import os

def calculate_cog():
    json_path = '../results/final_landmarks.json'
    
    if not os.path.exists(json_path):
        print(f"Error: Could not find {json_path}. Run this script from inside the 'analysis' folder.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Biomechanical Mass Distribution (Dempster's parameters)
    W_HEAD = 0.08
    W_TRUNK = 0.50
    W_ARMS = 0.10
    W_THIGHS = 0.20
    W_CALVES = 0.12

    frames = []
    x_cog_list = []
    y_cog_list = []

    print("Analyzing kinematic data...")

    # Iterate through the dictionary keys (frame numbers) and values (landmarks)
    for frame_str, lms in data.items():
        # Ensure the key is a frame number
        if not frame_str.isdigit():
            continue
            
        frame_idx = int(frame_str)
        
        # Skip if no pose was detected in this frame (empty list)
        if not lms or len(lms) < 33:
            continue

        # Helper function to get average coordinates for segments
        def get_avg(indices, axis='x'):
            return sum([lms[i][axis] for i in indices]) / len(indices)

        # 1. Head (Nose)
        head_x, head_y = lms[0]['x'], lms[0]['y']
        
        # 2. Trunk (Avg of Shoulders 11,12 and Hips 23,24)
        trunk_x = get_avg([11, 12, 23, 24], 'x')
        trunk_y = get_avg([11, 12, 23, 24], 'y')
        
        # 3. Arms (Avg of Elbows 13,14 and Wrists 15,16)
        arms_x = get_avg([13, 14, 15, 16], 'x')
        arms_y = get_avg([13, 14, 15, 16], 'y')
        
        # 4. Thighs (Avg of Hips 23,24 and Knees 25,26)
        thighs_x = get_avg([23, 24, 25, 26], 'x')
        thighs_y = get_avg([23, 24, 25, 26], 'y')
        
        # 5. Calves/Feet (Avg of Knees 25,26 and Ankles 27,28)
        calves_x = get_avg([25, 26, 27, 28], 'x')
        calves_y = get_avg([25, 26, 27, 28], 'y')

        # Execute CoG Formula: Sum(m_i * x_i) / Sum(m_i)
        X_cog = (head_x*W_HEAD + trunk_x*W_TRUNK + arms_x*W_ARMS + thighs_x*W_THIGHS + calves_x*W_CALVES)
        Y_cog = (head_y*W_HEAD + trunk_y*W_TRUNK + arms_y*W_ARMS + thighs_y*W_THIGHS + calves_y*W_CALVES)

        frames.append(frame_idx)
        x_cog_list.append(X_cog)
        y_cog_list.append(Y_cog)

    if not frames:
        print("Error: No frames with valid pose data found to plot. Check the JSON file.")
        return

    # Generate the Stability Plot
    plt.figure(figsize=(12, 6))
    
    # We plot the Y (Vertical) CoG to show the stability of the Kamae stance
    plt.plot(frames, y_cog_list, label='Vertical Center of Gravity', color='#D32F2F', linewidth=2)
    
    plt.title('Kinematic Stability Analysis - Shakkyō (Lion Dance)', fontsize=14, fontweight='bold')
    plt.xlabel('Frame Number', fontsize=12)
    plt.ylabel('Vertical Displacement (Normalized)', fontsize=12)
    
    # MediaPipe Y-axis starts at 0 at the top, so we invert it for readability
    plt.gca().invert_yaxis() 
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    output_img = '../results/cog_stability_plot.png'
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    print(f"Success! Kinematic graph saved to: {output_img}")

if __name__ == "__main__":
    calculate_cog()