import cv2, time
import pose_module as pm
from scipy.spatial.distance import cosine, euclidean
from fastdtw import fastdtw
import numpy as np
import os
import logging

# Configure logging
logger = logging.getLogger(__name__)

def process_frame(image, detector, is_user=False):
    """Process a single frame with pose detection and visualization"""
    # Resize frame
    image = cv2.resize(image, (720, 640))
    
    # Detect pose
    image = detector.findPose(image)
    lmList = detector.findPosition(image)
    
    if is_user and (not lmList or len(lmList) < 25):
        cv2.putText(image, "POSE NOT DETECTED", (40, 600),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return image, lmList

def compare_positions(benchmark_video):
    # Check if benchmark video exists
    if not os.path.exists(benchmark_video):
        logger.error(f"Error: Benchmark video not found at {benchmark_video}")
        return 0, 0, 0, None, None

    logger.info(f"Opening benchmark video: {benchmark_video}")
    # Capture benchmark video from file and user video from the live camera
    benchmark_cam = cv2.VideoCapture(benchmark_video)
    
    # Check video properties
    if benchmark_cam.isOpened():
        width = benchmark_cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = benchmark_cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = benchmark_cam.get(cv2.CAP_PROP_FPS)
        frame_count = benchmark_cam.get(cv2.CAP_PROP_FRAME_COUNT)
        logger.info(f"Benchmark video properties - Width: {width}, Height: {height}, FPS: {fps}, Total Frames: {frame_count}")
    else:
        logger.error(f"Error: Could not open benchmark video at {benchmark_video}")
        return 0, 0, 0, None, None

    # Try different webcam indices
    user_cam = None
    for i in range(3):  # Try first 3 camera indices
        user_cam = cv2.VideoCapture(i)
        if user_cam.isOpened():
            logger.info(f"Successfully opened webcam at index {i}")
            break
        else:
            logger.warning(f"Could not open webcam at index {i}")
            user_cam.release()

    if not user_cam or not user_cam.isOpened():
        logger.error("Error: Could not open any webcam")
        benchmark_cam.release()
        return 0, 0, 0, None, None

    logger.info("Successfully opened both video sources")
    fps_time = 0  # Initialize fps time
    detector_1 = pm.poseDetector(trackCon=0.7)  # Increased tracking confidence
    detector_2 = pm.poseDetector(trackCon=0.7)
    frame_counter = 0
    correct_frames = 0
    error_lst = 0
    acc = 0
    n = 0
    
    # For temporal smoothing
    error_window = []
    window_size = 5
    
    # Constants
    error_threshold = 0.3  # Lower threshold for more strict comparison
    min_required_landmarks = 25  # Minimum landmarks needed for valid comparison

    try:
        while benchmark_cam.isOpened() and user_cam.isOpened():
            # Read a frame from the live user video
            ret_user, image_user = user_cam.read()
            if not ret_user:
                logger.warning("Failed to grab user frame")
                break

            # Process user frame
            image_user, lmList_user = process_frame(image_user, detector_1, is_user=True)
            
            # Validate user landmarks
            if not lmList_user or len(lmList_user) < min_required_landmarks:
                logger.warning("Could not detect user pose properly")
                continue

            # Read a frame from the benchmark video
            ret_bench, image_bench = benchmark_cam.read()
            if not ret_bench:
                logger.info("Reached end of benchmark video, restarting")
                benchmark_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_bench, image_bench = benchmark_cam.read()
                if not ret_bench:
                    logger.error("Failed to grab benchmark frame after restart")
                    break

            # Process benchmark frame
            image_bench, lmList_benchmark = process_frame(image_bench, detector_2)
            
            # Validate benchmark landmarks
            if not lmList_benchmark or len(lmList_benchmark) < min_required_landmarks:
                logger.warning("Could not detect benchmark pose properly")
                continue
            
            frame_counter += 1
            
            try:
                # Convert landmarks to numpy arrays for comparison
                user_landmarks = np.array([(lm[1], lm[2]) for lm in lmList_user])
                benchmark_landmarks = np.array([(lm[1], lm[2]) for lm in lmList_benchmark])
                
                # Calculate multiple error metrics
                error, _ = fastdtw(user_landmarks, benchmark_landmarks, dist=cosine)
                
                # Add to error window for smoothing
                error_window.append(error)
                if len(error_window) > window_size:
                    error_window.pop(0)
                
                # Use smoothed error
                smoothed_error = sum(error_window) / len(error_window)
                error_lst += smoothed_error * 100
                n += 1
                
                # Overlay error percentage on user frame
                cv2.putText(image_user, f'Error: {round(100 * float(smoothed_error), 2)}%', (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Mark the step as correct or incorrect based on error threshold
                if smoothed_error < error_threshold:
                    cv2.putText(image_user, "CORRECT STEPS", (40, 600),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    correct_frames += 1
                else:
                    cv2.putText(image_user, "INCORRECT STEPS", (40, 600),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            except Exception as e:
                logger.error(f"Error in pose comparison: {str(e)}")
                continue

            # Calculate and display FPS
            current_time = time.time()
            fps = 1.0 / (current_time - fps_time) if fps_time > 0 else 0
            fps_time = current_time
            cv2.putText(image_user, f"FPS: {fps:.1f}", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display dynamic accuracy (percentage of correct frames)
            if frame_counter > 0:
                current_acc = 100 * correct_frames / frame_counter
                cv2.putText(image_user, f"Dance Steps Accurately Done: {round(current_acc, 2)}%", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                acc += current_acc

            # Return the processed frames
            yield image_user, image_bench, error_lst, acc, n

    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")
    finally:
        benchmark_cam.release()
        user_cam.release()
        
        # Return final values
        if n == 0:
            logger.warning("No frames were processed successfully")
            return 0, 0, 0, None, None
        logger.info(f"Processed {n} frames successfully")
        return error_lst, acc, n, None, None