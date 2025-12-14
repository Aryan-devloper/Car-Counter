import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt") 
video_path = "road.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"ERROR: Could not open video file '{video_path}'.")
    print("Please verify the file exists and is not corrupted.")
    exit()

original_fps = cap.get(cv2.CAP_PROP_FPS)
delay_ms = int(1000 / original_fps) if original_fps > 0 else 30 
print(f"Video FPS: {original_fps:.2f}. Setting waitKey delay to {delay_ms}ms.")

counted_track_ids = set() 
counted_cars_total = 0 
COUNTING_ZONE_BUFFER = 10 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Video stream ended.")
        break

    H, W, _ = frame.shape
    counting_line_y = int(H * 0.70) 
    
    cv2.line(frame, (0, counting_line_y), (W, counting_line_y), (255, 0, 0), 2)
    
    results = model.track(frame, persist=True, tracker="bytetrack.yaml") 
    
    
    if results and results[0].boxes.id is not None:
        
        for r in results:
            boxes = r.boxes
            track_ids = boxes.id.tolist() 

            for i, box in enumerate(boxes):
                cls = int(box.cls[0])
                label = model.names[cls]
                track_id = track_ids[i] 
                
                if label in ["car", "truck", "bus", "motorbike"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cy_bottom = y2 
                    
                    
                    if (counting_line_y - COUNTING_ZONE_BUFFER) < cy_bottom < (counting_line_y + COUNTING_ZONE_BUFFER):
                        
                        
                        if track_id not in counted_track_ids:
                            counted_cars_total += 1
                            counted_track_ids.add(track_id)
                            
                            cv2.line(frame, (0, counting_line_y), (W, counting_line_y), (0, 255, 0), 5) 

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} ID: {track_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"Total Cars Counted: {counted_cars_total}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 0, 255), 3)

    cv2.imshow("Car Counting System", frame)

    if cv2.waitKey(delay_ms) & 0xFF == 27: 
        break

cap.release()
cv2.destroyAllWindows()