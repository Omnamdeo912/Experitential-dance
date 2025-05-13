# Dance Pose Detection and Comparison

This project aims to provide dance learners with a virtual teacher that can compare their dance movements with a benchmark video. By analyzing the poses in real-time, the module identifies errors in the learner's movements and provides recommendations to correct their poses. The system supports both uploaded videos and live webcam feeds for comparison.

## Features
- Compare dance movements with a benchmark video.
- Real-time feedback on accuracy and errors.
- Dynamic display of accuracy percentage and error points.
- Support for live webcam feed or pre-recorded videos.
- Easy-to-use web interface for interaction.

## Prerequisites
- Docker installed on your system.
- A benchmark video for comparison.

## Instructions to Run the Application

1. **Pull the Docker Image**  
   Run the following command to pull the pre-built Docker image:
   ```sh
   docker pull omprakash/dance-comparison
