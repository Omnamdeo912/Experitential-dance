from flask import Flask, redirect, url_for, request, render_template, Response
import cv2
import mediapipe as mp
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

def generate_frames(gestureName):
    # If possible, consider loading the model once outside the loop for efficiency.
    while True:
        with open("./models/" + gestureName + '.pkl', 'rb') as f:
            model_body = pickle.load(f)
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame using Mediapipe's Holistic model
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Draw landmarks for right hand
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                )

            # Draw landmarks for left hand
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                )

            try:
                # Extract right-hand features if available
                right_features = []
                if results.right_hand_landmarks:
                    rHand = results.right_hand_landmarks.landmark
                    right_features = list(np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in rHand]
                    ).flatten())

                # Extract left-hand features if available
                left_features = []
                if results.left_hand_landmarks:
                    lHand = results.left_hand_landmarks.landmark
                    left_features = list(np.array(
                        [[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in lHand]
                    ).flatten())

                # Combine features from both hands if available
                if right_features and left_features:
                    features = right_features + left_features
                elif right_features:
                    features = right_features
                elif left_features:
                    features = left_features
                else:
                    features = []  # No landmarks detected

                # If features are available, predict the gesture and display the result.
                if features:
                    X4 = pd.DataFrame([features])
                    prediction = model_body.predict(X4)[0]
                    prob = model_body.predict_proba(X4)[0][np.argmax(model_body.predict_proba(X4)[0])]
                    cv2.putText(image, f"{gestureName}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Prob: {prob:.2f}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            except Exception as e:
                # Uncomment the following line to debug any errors during prediction:
                # print("Error:", e)
                pass

            ret, buffer = cv2.imencode('.jpg', image)
            img = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

@app.route('/')
def index():
    return render_template("index.html", title="Home")

@app.route('/hands/')
def hand():
    return render_template("hand_home.html", title="Hand Gesture Home")

@app.route('/handGesture/', methods=['POST'])
def handGesture():
    gestureName = ""
    # The form is expected to send the gesture name as a key.
    for key in request.form:
        gestureName = key
    print("Gesture selected:", gestureName)
    return render_template("hand.html", title="Hand Gesture", data=gestureName)

@app.route('/video_feed')
def video_feed():
    gestureName = request.args.get("gestureName")
    return Response(generate_frames(gestureName), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
