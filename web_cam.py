from ultralytics import YOLO
import cv2


def webcam():
    # Define class names (modify according to your classes)
    class_names = ["Fire", "default", "smoke"]
    model = YOLO("best.pt")
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            results = model(frame)
            annotated_frame = results[0].plot()
            new_frame = annotated_frame.copy()
            for a in results:
                b = a.boxes.data.tolist()
                if len(b)>0:
                    print(b[0])
                    x1, y1, x2, y2, conf, cls = b[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    label = class_names[int(cls)]
                    # score = f"{conf:.2f}"
                    color = (0, 255, 0) if label == "default" else (0, 0, 255)
                    cv2.rectangle(new_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(new_frame, f"{label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            out.write(new_frame)
            cv2.imshow("video22",new_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
