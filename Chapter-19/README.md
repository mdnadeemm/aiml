# Chapter 19: Video Analysis and Tracking

## Learning Objectives
By the end of this chapter, students will be able to:
- Understand video data structures and processing challenges
- Implement motion detection and background subtraction algorithms
- Apply object tracking techniques (Kalman filters, particle filters)
- Analyze temporal patterns and activity recognition in videos
- Build real-time video analysis systems for surveillance and monitoring

## Table of Contents
1. [Introduction to Video Analysis](#introduction)
2. [Motion Detection and Background Subtraction](#motion-detection)
3. [Object Tracking Algorithms](#tracking)
4. [Multi-Object Tracking](#multi-tracking)
5. [Activity Recognition](#activity)
6. [Real-Time Processing](#real-time)

## 1. Introduction to Video Analysis {#introduction}

Video analysis extends image processing to temporal sequences, adding the dimension of time and motion. This enables understanding of dynamic scenes, object trajectories, and behavioral patterns.

### Video Data Structure

**Frame Sequence**: Video as sequence of images (frames) captured at regular intervals
**Temporal Resolution**: Frames per second (fps) - typically 24-60 fps
**Spatial Resolution**: Width × height of each frame
**Color Channels**: RGB, YUV, or grayscale representations

### Key Challenges

**Computational Complexity**: Processing multiple frames requires significant computation
**Real-time Constraints**: Many applications require immediate processing
**Motion Blur**: Fast-moving objects may appear blurred
**Illumination Changes**: Lighting variations across frames
**Camera Motion**: Distinguishing object motion from camera movement
**Occlusions**: Objects disappearing behind others temporarily

### Applications

**Surveillance**: Intrusion detection, crowd monitoring, behavioral analysis
**Autonomous Vehicles**: Pedestrian tracking, lane detection, obstacle avoidance
**Sports Analysis**: Player tracking, performance metrics, strategy analysis
**Medical Imaging**: Cardiac motion analysis, surgery guidance
**Human-Computer Interaction**: Gesture recognition, gaze tracking

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans

class VideoAnalysisDemo:
    """Video analysis and tracking demonstrations"""
    
    def __init__(self):
        self.create_synthetic_video()
        self.setup_trackers()
        
    def create_synthetic_video(self):
        """Create synthetic video sequence for demonstration"""
        frames = []
        frame_count = 50
        frame_size = (300, 400)
        
        # Create moving objects
        ball1_path = [(50 + t*4, 100 + int(20*np.sin(t*0.3))) for t in range(frame_count)]
        ball2_path = [(350 - t*3, 50 + t*2) for t in range(frame_count)]
        
        for i in range(frame_count):
            # Create background
            frame = np.ones(frame_size, dtype=np.uint8) * 50
            
            # Add noise
            noise = np.random.normal(0, 5, frame_size)
            frame = np.clip(frame + noise, 0, 255).astype(np.uint8)
            
            # Add moving objects
            if i < len(ball1_path):
                center1 = ball1_path[i]
                if 0 <= center1[1] < frame_size[0] and 0 <= center1[0] < frame_size[1]:
                    cv2.circle(frame, center1, 15, 200, -1)
            
            if i < len(ball2_path):
                center2 = ball2_path[i]
                if 0 <= center2[1] < frame_size[0] and 0 <= center2[0] < frame_size[1]:
                    cv2.circle(frame, center2, 12, 180, -1)
            
            frames.append(frame)
        
        self.synthetic_video = frames
        self.ground_truth_tracks = {
            'object1': ball1_path,
            'object2': ball2_path
        }
    
    def setup_trackers(self):
        """Initialize tracking algorithms"""
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50)
        
        # Kalman filter for tracking
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                 [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
    
    def demonstrate_frame_differencing(self):
        """Show basic frame differencing for motion detection"""
        print("=== FRAME DIFFERENCING FOR MOTION DETECTION ===")
        
        # Calculate frame differences
        diff_frames = []
        for i in range(1, min(5, len(self.synthetic_video))):
            current = self.synthetic_video[i].astype(float)
            previous = self.synthetic_video[i-1].astype(float)
            diff = np.abs(current - previous)
            diff_frames.append(diff)
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Show original frames
        for i in range(4):
            if i < len(self.synthetic_video):
                axes[0, i].imshow(self.synthetic_video[i], cmap='gray')
                axes[0, i].set_title(f'Frame {i}')
                axes[0, i].axis('off')
        
        # Show difference frames
        for i in range(3):
            if i < len(diff_frames):
                axes[1, i].imshow(diff_frames[i], cmap='hot')
                axes[1, i].set_title(f'Diff {i} vs {i+1}')
                axes[1, i].axis('off')
        
        # Thresholded difference
        if len(diff_frames) > 0:
            thresh_diff = (diff_frames[0] > 30).astype(np.uint8) * 255
            axes[1, 3].imshow(thresh_diff, cmap='gray')
            axes[1, 3].set_title('Thresholded Motion')
            axes[1, 3].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("Frame differencing characteristics:")
        print("• Simple and fast motion detection")
        print("• Sensitive to noise and illumination changes")
        print("• Only detects moving edges, not entire objects")
        print("• Good for initial motion detection")

# Initialize and demonstrate video analysis
video_demo = VideoAnalysisDemo()
video_demo.demonstrate_frame_differencing()
```

## 2. Motion Detection and Background Subtraction {#motion-detection}

Background subtraction separates moving foreground objects from the static background scene. This fundamental technique enables object detection and tracking in video sequences.

### Background Modeling Approaches

**Running Average**: Simple adaptive background model
B(x,y,t) = αI(x,y,t) + (1-α)B(x,y,t-1)

**Gaussian Mixture Models (GMM)**: Model each pixel as mixture of Gaussians
**Median Background**: Use median pixel values over time
**CodeBook**: Quantize pixel values into representative codewords

### Foreground Detection

**Threshold-based**: |I(x,y,t) - B(x,y,t)| > T
**Statistical**: Compare with background model statistics
**Adaptive**: Adjust parameters based on scene characteristics

```python
class MotionDetection:
    """Motion detection and background subtraction implementations"""
    
    def __init__(self, video_frames):
        self.frames = video_frames
        self.background_models = {}
        
    def running_average_background(self, alpha=0.1):
        """Implement running average background subtraction"""
        print("=== RUNNING AVERAGE BACKGROUND SUBTRACTION ===")
        
        # Initialize background with first frame
        background = self.frames[0].astype(np.float32)
        foreground_masks = []
        
        for i, frame in enumerate(self.frames[1:], 1):
            frame_float = frame.astype(np.float32)
            
            # Calculate foreground mask
            diff = np.abs(frame_float - background)
            foreground_mask = (diff > 30).astype(np.uint8) * 255
            foreground_masks.append(foreground_mask)
            
            # Update background model
            background = alpha * frame_float + (1 - alpha) * background
            
            if i == 1:  # Store first background for visualization
                self.background_models['running_avg'] = background.copy()
        
        # Visualize results
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        
        # Show background evolution
        backgrounds = [self.frames[0], self.background_models['running_avg']]
        titles = ['Initial Background', f'Background (α={alpha})']
        
        for i, (bg, title) in enumerate(zip(backgrounds, titles)):
            axes[0, i].imshow(bg, cmap='gray')
            axes[0, i].set_title(title)
            axes[0, i].axis('off')
        
        # Show original frames and foreground masks
        frame_indices = [5, 15]
        for i, idx in enumerate(frame_indices):
            if idx < len(self.frames):
                axes[0, i+2].imshow(self.frames[idx], cmap='gray')
                axes[0, i+2].set_title(f'Frame {idx}')
                axes[0, i+2].axis('off')
                
                if idx-1 < len(foreground_masks):
                    axes[1, i+2].imshow(foreground_masks[idx-1], cmap='gray')
                    axes[1, i+2].set_title(f'Foreground {idx}')
                    axes[1, i+2].axis('off')
        
        # Show foreground detection evolution
        if len(foreground_masks) >= 2:
            axes[1, 0].imshow(foreground_masks[0], cmap='gray')
            axes[1, 0].set_title('Early Foreground')
            axes[1, 0].axis('off')
            
            axes[1, 1].imshow(foreground_masks[-1], cmap='gray')
            axes[1, 1].set_title('Later Foreground')
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return foreground_masks
    
    def gaussian_mixture_background(self):
        """Demonstrate Gaussian Mixture Model background subtraction"""
        print("\n=== GAUSSIAN MIXTURE MODEL BACKGROUND SUBTRACTION ===")
        
        # Use OpenCV's MOG2 background subtractor
        bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50)
        
        mog2_masks = []
        
        for frame in self.frames:
            # Apply MOG2 background subtraction
            fg_mask = bg_subtractor.apply(frame)
            mog2_masks.append(fg_mask)
        
        # Get learned background
        background_mog2 = bg_subtractor.getBackgroundImage()
        
        # Visualize MOG2 results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Show learned background
        if background_mog2 is not None:
            axes[0, 0].imshow(background_mog2, cmap='gray')
            axes[0, 0].set_title('MOG2 Background Model')
            axes[0, 0].axis('off')
        
        # Show sample frames and masks
        sample_indices = [10, 25]
        for i, idx in enumerate(sample_indices):
            if idx < len(self.frames) and idx < len(mog2_masks):
                axes[0, i+1].imshow(self.frames[idx], cmap='gray')
                axes[0, i+1].set_title(f'Original Frame {idx}')
                axes[0, i+1].axis('off')
                
                axes[1, i+1].imshow(mog2_masks[idx], cmap='gray')
                axes[1, i+1].set_title(f'MOG2 Foreground {idx}')
                axes[1, i+1].axis('off')
        
        # Show mask evolution
        if len(mog2_masks) > 0:
            axes[1, 0].imshow(mog2_masks[0], cmap='gray')
            axes[1, 0].set_title('Initial MOG2 Mask')
            axes[1, 0].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("MOG2 advantages:")
        print("• Handles multimodal backgrounds (waving trees, etc.)")
        print("• Adapts to gradual illumination changes")
        print("• Shadow detection capabilities")
        print("• Robust to noise and small movements")
        
        return mog2_masks

# Apply motion detection to synthetic video
motion_detector = MotionDetection(video_demo.synthetic_video)
running_avg_masks = motion_detector.running_average_background()
mog2_masks = motion_detector.gaussian_mixture_background()
```

## 3. Object Tracking Algorithms {#tracking}

Object tracking maintains the identity and location of objects across video frames. Effective tracking handles occlusions, appearance changes, and complex motion patterns.

### Tracking Approaches

**Template Matching**: Track objects by matching appearance templates
**Feature-based Tracking**: Track distinctive features and group them
**Model-based Tracking**: Use motion models to predict object locations
**Learning-based Tracking**: Adapt appearance models during tracking

### Kalman Filter Tracking

The Kalman filter provides optimal state estimation for linear systems with Gaussian noise.

**State Vector**: [x, y, vx, vy] - position and velocity
**Prediction Step**: Predict next state using motion model
**Update Step**: Correct prediction using measurements

```python
class ObjectTracking:
    """Object tracking algorithm implementations"""
    
    def __init__(self, video_frames):
        self.frames = video_frames
        self.tracks = []
        
    def template_tracking_demo(self):
        """Demonstrate template-based tracking"""
        print("=== TEMPLATE-BASED TRACKING ===")
        
        # Select initial object region from first frame
        first_frame = self.frames[0]
        
        # Define initial bounding box (manually for demo)
        initial_bbox = (35, 85, 30, 30)  # x, y, width, height
        x, y, w, h = initial_bbox
        template = first_frame[y:y+h, x:x+w]
        
        tracked_positions = []
        tracked_positions.append((x + w//2, y + h//2))
        
        # Track through subsequent frames
        for i, frame in enumerate(self.frames[1:6]):  # Track first 5 frames
            # Template matching
            result = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            # Update position
            new_x, new_y = max_loc
            center = (new_x + w//2, new_y + h//2)
            tracked_positions.append(center)
            
            # Update template (adaptive tracking)
            if max_val > 0.7:  # High confidence update
                template = frame[new_y:new_y+h, new_x:new_x+w]
        
        # Visualize tracking results
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Show initial template
        axes[0, 0].imshow(template, cmap='gray')
        axes[0, 0].set_title('Initial Template')
        axes[0, 0].axis('off')
        
        # Show tracking on sample frames
        for i in range(5):
            if i < len(self.frames) and i < len(tracked_positions):
                frame_copy = self.frames[i].copy()
                center = tracked_positions[i]
                
                # Draw tracking box
                top_left = (center[0] - w//2, center[1] - h//2)
                cv2.rectangle(frame_copy, top_left, 
                            (top_left[0] + w, top_left[1] + h), 255, 2)
                cv2.circle(frame_copy, center, 3, 255, -1)
                
                row, col = (i+1) // 3, (i+1) % 3
                if row < 2 and col < 3:
                    axes[row, col].imshow(frame_copy, cmap='gray')
                    axes[row, col].set_title(f'Frame {i}')
                    axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        return tracked_positions
    
    def kalman_filter_tracking(self):
        """Implement Kalman filter for object tracking"""
        print("\n=== KALMAN FILTER TRACKING ===")
        
        # Initialize Kalman filter
        kalman = cv2.KalmanFilter(4, 2)  # 4 state vars, 2 measurement vars
        
        # State transition matrix (constant velocity model)
        kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy  
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=np.float32)
        
        # Measurement matrix
        kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],  # measure x
            [0, 1, 0, 0]   # measure y
        ], dtype=np.float32)
        
        # Process noise covariance
        kalman.processNoiseCov = 0.1 * np.eye(4, dtype=np.float32)
        
        # Measurement noise covariance
        kalman.measurementNoiseCov = 10 * np.eye(2, dtype=np.float32)
        
        # Initial state (first detection)
        initial_detection = (50, 100)  # x, y
        kalman.statePre = np.array([
            initial_detection[0], initial_detection[1], 0, 0
        ], dtype=np.float32)
        
        # Simulate noisy detections (add noise to ground truth)
        np.random.seed(42)
        true_positions = [(50 + t*4, 100 + int(20*np.sin(t*0.3))) 
                         for t in range(20)]
        
        noisy_detections = []
        for pos in true_positions:
            noise_x = np.random.normal(0, 5)
            noise_y = np.random.normal(0, 5)
            noisy_detections.append((pos[0] + noise_x, pos[1] + noise_y))
        
        # Track with Kalman filter
        kalman_predictions = []
        kalman_estimates = []
        
        for i, detection in enumerate(noisy_detections):
            # Prediction step
            prediction = kalman.predict()
            kalman_predictions.append((prediction[0], prediction[1]))
            
            # Update step with measurement
            measurement = np.array([[detection[0]], [detection[1]]], dtype=np.float32)
            estimate = kalman.correct(measurement)
            kalman_estimates.append((estimate[0, 0], estimate[1, 0]))
        
        # Visualize tracking performance
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot trajectories
        true_x, true_y = zip(*true_positions)
        noisy_x, noisy_y = zip(*noisy_detections)
        pred_x, pred_y = zip(*kalman_predictions)
        est_x, est_y = zip(*kalman_estimates)
        
        axes[0].plot(true_x, true_y, 'g-', label='Ground Truth', linewidth=2)
        axes[0].plot(noisy_x, noisy_y, 'r.', label='Noisy Detections', markersize=8)
        axes[0].plot(est_x, est_y, 'b-', label='Kalman Estimates', linewidth=2)
        axes[0].set_title('Kalman Filter Tracking')
        axes[0].set_xlabel('X Position')
        axes[0].set_ylabel('Y Position')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot tracking errors
        estimation_errors = [euclidean(true_positions[i], kalman_estimates[i]) 
                           for i in range(len(true_positions))]
        detection_errors = [euclidean(true_positions[i], noisy_detections[i]) 
                          for i in range(len(true_positions))]
        
        axes[1].plot(detection_errors, 'r-', label='Detection Error', linewidth=2)
        axes[1].plot(estimation_errors, 'b-', label='Kalman Error', linewidth=2)
        axes[1].set_title('Tracking Error Over Time')
        axes[1].set_xlabel('Frame')
        axes[1].set_ylabel('Position Error (pixels)')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        avg_detection_error = np.mean(detection_errors)
        avg_kalman_error = np.mean(estimation_errors)
        
        print(f"Tracking Performance:")
        print(f"• Average detection error: {avg_detection_error:.2f} pixels")
        print(f"• Average Kalman error: {avg_kalman_error:.2f} pixels")
        print(f"• Error reduction: {((avg_detection_error - avg_kalman_error) / avg_detection_error * 100):.1f}%")
        
        return kalman_estimates

# Apply tracking algorithms
tracker = ObjectTracking(video_demo.synthetic_video)
template_positions = tracker.template_tracking_demo()
kalman_positions = tracker.kalman_filter_tracking()
```

## Summary

This chapter covered essential video analysis and tracking techniques:

### Key Concepts:
1. **Video Data Structure**: Temporal sequences and processing challenges
2. **Motion Detection**: Frame differencing and background subtraction
3. **Object Tracking**: Template matching, Kalman filtering, feature tracking
4. **Performance Evaluation**: Accuracy metrics and error analysis

### Practical Applications:
- **Surveillance Systems**: Security monitoring and intrusion detection
- **Autonomous Vehicles**: Object tracking for navigation safety
- **Sports Analytics**: Player performance and movement analysis
- **Medical Imaging**: Organ motion tracking during procedures

### Best Practices:
- Choose tracking methods based on object characteristics and motion patterns
- Handle occlusions and appearance changes with robust algorithms
- Optimize for real-time performance when required
- Validate tracking accuracy with ground truth data
- Consider multi-object tracking for complex scenes

---

## Exercises

1. **Multi-Object Tracking**: Implement tracking multiple objects simultaneously
2. **Particle Filter**: Develop particle filter tracking for non-linear motion
3. **Real-Time System**: Build real-time video tracking application
4. **Activity Recognition**: Analyze motion patterns for behavior classification
5. **Performance Optimization**: Optimize tracking algorithms for mobile devices

---

*Video analysis and tracking enable understanding of dynamic scenes and temporal patterns, forming the foundation for advanced applications in surveillance, robotics, and human behavior analysis.* 