from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from move_comparison import compare_positions
import threading
import time
import os
import logging
import queue
import pose_module as pm
import mediapipe as mp
import csv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Global variables to store the comparison results
current_error = 0
current_status = "WAITING"
current_accuracy = 0
frame_queue = queue.Queue(maxsize=2)  # Queue to store processed frames
comparison_thread = None
should_stop = False

# Global variables for hand mudra detection
hand_mudra_cap = None
hand_mudra_thread = None
hand_mudra_running = False
hand_mudra_class_name = "Pataka"  # Default mudra class name

# Get the absolute path to the benchmark video
current_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_video = os.path.join(current_dir, "dance_videos", "benchmarkOdissi.mp4")

# Verify the video file exists
if not os.path.exists(benchmark_video):
    logger.error(f"Benchmark video not found at: {benchmark_video}")
else:
    logger.info(f"Found benchmark video at: {benchmark_video}")

def generate_frames():
    """Generate frames with comparison results"""
    while True:
        try:
            # Get frames from the queue if available
            user_frame, benchmark_frame, error, acc, n = frame_queue.get(timeout=1.0)
            
            # Resize frames to match
            user_frame = cv2.resize(user_frame, (720, 640))
            benchmark_frame = cv2.resize(benchmark_frame, (720, 640))
            
            # Add error and accuracy text to user frame
            cv2.putText(user_frame, f"Error: {error:.2f}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(user_frame, f"Accuracy: {acc*100:.2f}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Combine frames side by side
            combined_frame = np.hstack((user_frame, benchmark_frame))
            
            ret, buffer = cv2.imencode('.jpg', combined_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except queue.Empty:
            # If no frames in queue, show waiting message
            frame = np.zeros((640, 1440, 3), dtype=np.uint8)
            cv2.putText(frame, "Waiting for comparison to start...", (400, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            logger.error(f"Error in frame generation: {str(e)}")
            continue

def run_comparison():
    """Run the comparison process and update global variables"""
    global current_error, current_status, current_accuracy, should_stop
    try:
        logger.info("Starting comparison process...")
        current_status = "COMPARING"
        should_stop = False
        
        # Run the comparison and process frames
        for user_frame, benchmark_frame, error, acc, n in compare_positions(benchmark_video):
            if should_stop:
                logger.info("Stopping comparison process...")
                break
                
            # Update global variables
            current_error = error
            current_accuracy = acc
            
            # Put frames in the queue
            try:
                frame_queue.put((user_frame, benchmark_frame, error, acc, n), timeout=1.0)
            except queue.Full:
                # If queue is full, remove old frames and try again
                try:
                    frame_queue.get_nowait()  # Remove old frame
                    frame_queue.put((user_frame, benchmark_frame, error, acc, n), timeout=1.0)
                except queue.Empty:
                    continue
            
    except Exception as e:
        logger.error(f"Error in comparison process: {str(e)}")
        current_status = "ERROR"
        current_error = 0
        current_accuracy = 0
    finally:
        current_status = "COMPLETED"
        should_stop = False

def generate_hand_mudra_frames():
    """Generate frames for hand mudra detection"""
    global hand_mudra_cap, hand_mudra_running
    hand_mudra_cap = cv2.VideoCapture(0)
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while hand_mudra_running and hand_mudra_cap.isOpened():
            ret, frame = hand_mudra_cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw right hand landmarks
            mp_drawing.draw_landmarks(
                image, 
                results.right_hand_landmarks, 
                mp_holistic.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )

            try:
                rhand = results.right_hand_landmarks.landmark
                rhand_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] 
                                         for landmark in rhand]).flatten())
                
                row = rhand_row
                row.insert(0, hand_mudra_class_name)
                with open("Pataka.csv", mode='a', newline='') as f:
                    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
            except:
                pass

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    if hand_mudra_cap is not None:
        hand_mudra_cap.release()

@app.route('/')
def index():
    logger.info("Rendering index page")
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Stream the combined video feed"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_comparison')
def start_comparison():
    global comparison_thread, should_stop
    logger.info("Starting comparison thread")
    try:
        # Clear the frame queue
        while not frame_queue.empty():
            frame_queue.get_nowait()
            
        # Start the comparison in a separate thread
        comparison_thread = threading.Thread(target=run_comparison)
        comparison_thread.daemon = True
        comparison_thread.start()
        return jsonify({"status": "started"})
    except Exception as e:
        logger.error(f"Error starting comparison: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/stop_comparison')
def stop_comparison():
    global should_stop
    logger.info("Stopping comparison...")
    should_stop = True
    return jsonify({"status": "stopped"})

@app.route('/get_status')
def get_status():
    return jsonify({
        "error": current_error,
        "status": current_status,
        "accuracy": current_accuracy
    })

@app.route('/hand_mudra')
def hand_mudra():
    return render_template('hand_mudra.html')

@app.route('/hand_mudra_feed')
def hand_mudra_feed():
    return Response(generate_hand_mudra_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_hand_mudra')
def start_hand_mudra():
    global hand_mudra_running
    hand_mudra_running = True
    return jsonify({"status": "started"})

@app.route('/stop_hand_mudra')
def stop_hand_mudra():
    global hand_mudra_running, hand_mudra_cap
    hand_mudra_running = False
    if hand_mudra_cap is not None:
        hand_mudra_cap.release()
        hand_mudra_cap = None
    return jsonify({"status": "stopped"})

if __name__ == '__main__':
    logger.info("Starting Flask server...")
    app.run(host='0.0.0.0', port=5001, debug=True) 