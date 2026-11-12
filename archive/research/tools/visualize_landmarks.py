import cv2
import pandas as pd
import numpy as np
from pathlib import Path

class MediaPipeVisualizer:
    # MediaPipe connection indices (correct format)
    POSE_CONNECTIONS = [
        # Head
        (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
        # Torso
        (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        # Left arm
        (11, 23), (23, 24), (24, 26),
        # Right arm
        (12, 24), (24, 25), (25, 27),
        # Left leg
        (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
        # Right leg
        (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
    ]

    FACE_CONNECTIONS = [
        # Lips
        (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321), (321, 375), (375, 291),
        # Left eye
        (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133), (133, 33),
        # Right eye
        (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362), (362, 263),
        # Left eyebrow
        (46, 53), (53, 52), (52, 65), (65, 55), (55, 107), (107, 66), (66, 107),
        # Right eyebrow
        (276, 283), (283, 282), (282, 295), (295, 285), (285, 336), (336, 296), (296, 336),
        # Face contour
        (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 150), (70, 63), (63, 105), (105, 66), (66, 107),
    ]

    HAND_CONNECTIONS = [
        # Thumb
        (0, 1), (1, 2), (2, 3), (3, 4),
        # Index
        (0, 5), (5, 6), (6, 7), (7, 8),
        # Middle
        (0, 9), (9, 10), (10, 11), (11, 12),
        # Ring
        (0, 13), (13, 14), (14, 15), (15, 16),
        # Pinky
        (0, 17), (17, 18), (18, 19), (19, 20),
    ]

    # Colors in BGR format
    COLORS = {
        'pose': (0, 255, 0),        # Green
        'face': (0, 255, 255),      # Yellow
        'left_hand': (255, 0, 255), # Magenta
        'right_hand': (255, 255, 0) # Cyan
    }

    def __init__(self, csv_path, output_path='output.mp4', fps=30, frame_size=(1280, 720)):
        """
        Initialize the visualizer.
        
        Args:
            csv_path: Path to the MediaPipe CSV file
            output_path: Path to save the output video
            fps: Frames per second for output video
            frame_size: (width, height) of output frames
        """
        self.csv_path = csv_path
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.data = None
        self.frames_dict = None
        
    def load_csv(self):
        """Load and parse the CSV file."""
        print(f"Loading CSV from {self.csv_path}...")
        self.data = pd.read_csv(self.csv_path)
        print(f"Loaded {len(self.data)} landmarks")
        
    def organize_frames(self):
        """Organize landmarks by frame and type."""
        self.frames_dict = {}
        
        for _, row in self.data.iterrows():
            frame = int(row['frame'])
            landmark_type = row['type']
            landmark_index = int(row['landmark_index'])
            x = row['x']
            y = row['y']
            z = row['z']
            
            # Skip rows with missing coordinates
            if pd.isna(x) or pd.isna(y):
                continue
                
            if frame not in self.frames_dict:
                self.frames_dict[frame] = {
                    'pose': [],
                    'face': [],
                    'left_hand': [],
                    'right_hand': []
                }
            
            self.frames_dict[frame][landmark_type].append({
                'index': landmark_index,
                'x': float(x),
                'y': float(y),
                'z': float(z) if not pd.isna(z) else 0
            })
        
        print(f"Organized into {len(self.frames_dict)} frames")
        return sorted(self.frames_dict.keys())
    
    def draw_landmarks(self, frame, landmarks, color, radius=5):
        """Draw landmarks on frame."""
        for landmark in landmarks:
            x = int(landmark['x'] * self.frame_size[0])
            y = int(landmark['y'] * self.frame_size[1])
            cv2.circle(frame, (x, y), radius, color, -1)
    
    def draw_connections(self, frame, landmarks, connections, color, thickness=2):
        """Draw connections between landmarks."""
        landmark_dict = {l['index']: l for l in landmarks}
        
        for start_idx, end_idx in connections:
            if start_idx in landmark_dict and end_idx in landmark_dict:
                start = landmark_dict[start_idx]
                end = landmark_dict[end_idx]
                
                start_pos = (int(start['x'] * self.frame_size[0]), int(start['y'] * self.frame_size[1]))
                end_pos = (int(end['x'] * self.frame_size[0]), int(end['y'] * self.frame_size[1]))
                
                cv2.line(frame, start_pos, end_pos, color, thickness)
    
    def draw_frame(self, frame_num, frame_data):
        """Draw all landmarks and connections for a frame."""
        canvas = np.zeros((self.frame_size[1], self.frame_size[0], 3), dtype=np.uint8)
        
        # Draw pose
        if frame_data['pose']:
            self.draw_connections(canvas, frame_data['pose'], self.POSE_CONNECTIONS, 
                                self.COLORS['pose'], thickness=2)
            self.draw_landmarks(canvas, frame_data['pose'], self.COLORS['pose'], radius=4)
        
        # Draw face
        if frame_data['face']:
            self.draw_connections(canvas, frame_data['face'], self.FACE_CONNECTIONS, 
                                self.COLORS['face'], thickness=1)
            self.draw_landmarks(canvas, frame_data['face'], self.COLORS['face'], radius=2)
        
        # Draw left hand
        if frame_data['left_hand']:
            self.draw_connections(canvas, frame_data['left_hand'], self.HAND_CONNECTIONS, 
                                self.COLORS['left_hand'], thickness=2)
            self.draw_landmarks(canvas, frame_data['left_hand'], self.COLORS['left_hand'], radius=3)
        
        # Draw right hand
        if frame_data['right_hand']:
            self.draw_connections(canvas, frame_data['right_hand'], self.HAND_CONNECTIONS, 
                                self.COLORS['right_hand'], thickness=2)
            self.draw_landmarks(canvas, frame_data['right_hand'], self.COLORS['right_hand'], radius=3)
        
        # Add frame number
        cv2.putText(canvas, f"Frame: {frame_num}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return canvas
    
    def create_video(self):
        """Create video from frames."""
        if self.data is None:
            self.load_csv()
        
        frame_nums = self.organize_frames()
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
        
        print(f"Creating video: {self.output_path}")
        
        for i, frame_num in enumerate(frame_nums):
            frame_data = self.frames_dict[frame_num]
            canvas = self.draw_frame(frame_num, frame_data)
            out.write(canvas)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(frame_nums)} frames")
        
        out.release()
        print(f"Video saved to {self.output_path}")
    
    def show_interactive(self):
        """Show interactive visualization with keyboard controls."""
        if self.data is None:
            self.load_csv()
        
        frame_nums = self.organize_frames()
        current_idx = 0
        is_playing = False
        
        print("Controls:")
        print("  SPACE - Play/Pause")
        print("  LEFT/RIGHT - Previous/Next frame")
        print("  Q - Quit")
        
        while True:
            frame_num = frame_nums[current_idx]
            frame_data = self.frames_dict[frame_num]
            canvas = self.draw_frame(frame_num, frame_data)
            
            # Add playback info
            status = "PLAYING" if is_playing else "PAUSED"
            cv2.putText(canvas, status, (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('MediaPipe Visualizer', canvas)
            
            key = cv2.waitKey(30 if is_playing else 0) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                is_playing = not is_playing
            elif key == 83:  # Right arrow
                current_idx = min(current_idx + 1, len(frame_nums) - 1)
            elif key == 81:  # Left arrow
                current_idx = max(current_idx - 1, 0)
        
        cv2.destroyAllWindows()


def main():
    """Example usage."""
    csv_file = 'Kaggle_Data/100015657.csv'  # Change this to your CSV file path
    
    # Create visualizer
    viz = MediaPipeVisualizer(
        csv_path=csv_file,
        output_path='Kaggle_Data/100015657.mp4',
        fps=30,
        frame_size=(1280, 720)
    )
    
    # Option 1: Create video file
    # viz.create_video()
    
    # Option 2: Show interactive visualization
    viz.show_interactive()


if __name__ == '__main__':
    main()