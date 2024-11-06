import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import use

# Set the backend to 'Agg' to avoid Tkinter dependency
use('Agg')

# Load the Haar Cascades for car and person detection
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
person_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

# Initialize the Kalman Filter for tracking
kalman = cv2.KalmanFilter(4, 2)
kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0.5, 0], [0, 0, 0, 0.5]], np.float32) * 0.1
kalman.measurementNoiseCov = np.array([[0.1, 0], [0, 0.1]], np.float32)

# Load the provided video file
cap = cv2.VideoCapture('./data/sample_video.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create a VideoWriter object to save the output video
output = cv2.VideoWriter('output_video_slow.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Prepare arrays for plot
car_counts = []
person_counts = []
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for the cascade classifiers
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect cars and people in the frame
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)
    people = person_cascade.detectMultiScale(gray, 1.1, 3)

    # Track detections
    car_counts.append(len(cars))
    person_counts.append(len(people))
    frames.append(len(frames))

    # Process detected cars and people
    for (x, y, w, h) in cars:
        measured = np.array([[np.float32(x + w / 2)], [np.float32(y + h / 2)]])
        kalman.correct(measured)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "Car", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    for (x, y, w, h) in people:
        measured = np.array([[np.float32(x + w / 2)], [np.float32(y + h / 2)]])
        kalman.correct(measured)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Predict and draw the predicted location
    predicted = kalman.predict()
    predicted_center = (int(predicted[0]), int(predicted[1]))
    cv2.circle(frame, predicted_center, 5, (0, 0, 255), -1)
    cv2.putText(frame, "Predicted", predicted_center, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    
    # Write the frame to the output video
    output.write(frame)
    
    # Show the result in slow motion
    cv2.imshow("Car and Person Detection and Tracking", frame)
    if cv2.waitKey(200) & 0xFF == 27:  # Press ESC to exit, with a delay for slow motion
        break

# Plot after video processing
plt.figure()
plt.plot(frames, car_counts, label='Cars Detected')
plt.plot(frames, person_counts, label='People Detected')
plt.xlabel('Frame')
plt.ylabel('Count')
plt.legend(loc='upper right')
plt.savefig('detection_counts.png')  # Save the plot as an image

# Release resources
cap.release()
output.release()
cv2.destroyAllWindows()
