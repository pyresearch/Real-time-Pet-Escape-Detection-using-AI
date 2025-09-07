import cv2
from ultralytics import YOLO
import urllib.request
import os
import typer
import numpy as np
import pyresearch

# Initialize Typer for command-line interface
app = typer.Typer()

# Load the YOLO model
model = YOLO("best.pt")  # Replace with your trained model for catss and dogss

def download_video(url, temp_file="temp_video.mp4"):
    """Download video from a URL to a temporary file."""
    try:
        print(f"Downloading video from {url}...")
        urllib.request.urlretrieve(url, temp_file)
        print(f"Video downloaded to {temp_file}")
        return temp_file
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None

def draw_dashed_line(img, start, end, color, thickness=2, dash_length=10):
    """Draw a dashed line on the image."""
    x1, y1 = start
    x2, y2 = end
    length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dashes = int(length / (2 * dash_length))
    if dashes == 0:
        dashes = 1
    dx = (x2 - x1) / dashes
    dy = (y2 - y1) / dashes
    for i in range(dashes):
        start_x = int(x1 + i * dx)
        start_y = int(y1 + i * dy)
        end_x = int(x1 + (i + 0.5) * dx)
        end_y = int(y1 + (i + 0.5) * dy)
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, thickness)

def process_video(input_source, output_file="yolov8n_demo2.mp4"):
    """Process video to count catss and dogss crossing a horizontal detection line."""
    # Handle URL input
    temp_file = None
    if input_source.startswith("http://") or input_source.startswith("https://"):
        temp_file = download_video(input_source)
        if temp_file is None:
            return
        input_source = temp_file
    elif input_source.lower() == "webcam":
        input_source = 0  # Use default webcam

    # Load the video
    cap = cv2.VideoCapture(input_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {input_source}.")
        return

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video writer
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Set the detection line position (horizontal line at y = frame_height // 2)
    detection_line_y = frame_height // 2  # Center row of the frame

    # Initialize tracking and counts
    tracked_ids = set()
    catss_in = 0
    catss_out = 0
    dogss_in = 0
    dogss_out = 0
    prev_positions = {}  # Store previous y-positions for each track_id

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or error reading frame.")
            break

        # Run detection with tracking (assuming classes 0=catss, 1=dogss)
        results = model.track(frame, persist=True, show=False, classes=[0, 1], conf=0.3, verbose=False)

        # Count and draw bounding boxes
        current_catss = 0
        current_dogss = 0
        current_positions = {}
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                track_id = int(box.id.item()) if box.id is not None else None
                class_id = int(box.cls.item()) if box.cls is not None else None
                class_name = "cats" if class_id == 0 else "dogs" if class_id == 1 else "unknown"

                # Draw bounding box with class-specific color
                color = (0, 255, 0) if class_name == "cats" else (255, 0, 0)  # Green for catss, blue for dogss
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
                cv2.putText(frame, f"{class_name} ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                # Update current counts
                if class_name == "cats":
                    current_catss += 1
                elif class_name == "dogs":
                    current_dogss += 1

                # Track direction for in/out counting
                if track_id is not None:
                    box_center_y = (y1 + y2) // 2
                    current_positions[track_id] = box_center_y

                    if track_id not in tracked_ids and track_id in prev_positions:
                        prev_y = prev_positions[track_id]
                        if prev_y > detection_line_y and box_center_y <= detection_line_y:  # Moving up (in)
                            tracked_ids.add(track_id)
                            if class_name == "cats":
                                catss_in += 1
                            elif class_name == "dogs":
                                dogss_in += 1
                        elif prev_y < detection_line_y and box_center_y >= detection_line_y:  # Moving down (out)
                            tracked_ids.add(track_id)
                            if class_name == "cats":
                                catss_out += 1
                            elif class_name == "dogs":
                                dogss_out += 1

        # Update previous positions
        prev_positions = current_positions.copy()

        # Draw semi-transparent overlay for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 250), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Draw the horizontal dashed detection line
        draw_dashed_line(frame, (0, detection_line_y), (frame_width, detection_line_y), (0, 0, 255), 4)

        # Add text labels for counts
        cv2.putText(frame, f"cat in frame: {current_catss}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"dog in frame: {current_dogss}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"cat entered home: {catss_in}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"cat exited home: {catss_out}", (20, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"dog entered home: {dogss_in}", (20, 170),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(frame, f"dog exited home: {dogss_out}", (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # Write to output and display
        out.write(frame)
        cv2.imshow("catss and dogss Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Clean up temporary file if downloaded
    if temp_file and os.path.exists(temp_file):
        os.remove(temp_file)
        print(f"Temporary file {temp_file} removed.")

    # Print final counts
    print(f"Total unique catss entered home: {catss_in}")
    print(f"Total unique catss exited home: {catss_out}")
    print(f"Total unique dogss entered home: {dogss_in}")
    print(f"Total unique dogss exited home: {dogss_out}")

@app.command()
def process(input_source: str = "test2.mp4", output_file: str = "yolov8n_demo2.mp4"):
    """Process video from a webcam, local file, or URL to count catss and dogss."""
    typer.echo(f"Starting video processing for {input_source}...")
    process_video(input_source, output_file)

if __name__ == "__main__":
    app()