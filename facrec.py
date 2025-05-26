import cv2
import os
import face_recognition
import pickle
import pandas as pd
from datetime import datetime

# === Step 1: Register user and capture images ===
def register_user(name):
    folder = f'dataset/{name}'
    os.makedirs(folder, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0

    print(f"[INFO] Starting image capture for user: {name}")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to access the webcam.")
            break

        count += 1
        img_path = f"{folder}/{name}_{count}.jpg"
        cv2.imwrite(img_path, frame)
        cv2.imshow("Registering User - Press ESC to stop", frame)

        # Stop on ESC key or after 20 images
        if cv2.waitKey(1) == 27 or count >= 20:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] {count} images saved to '{folder}'")


# === Step 2: Encode faces from dataset images ===
def encode_faces(dataset_dir="dataset"):
    known_encodings = []
    known_names = []

    if not os.path.exists(dataset_dir):
        print(f"[ERROR] Dataset directory '{dataset_dir}' not found. Please register users first.")
        return None

    print("[INFO] Encoding faces from dataset...")
    for user in os.listdir(dataset_dir):
        user_folder = os.path.join(dataset_dir, user)
        if not os.path.isdir(user_folder):
            continue

        for img_name in os.listdir(user_folder):
            img_path = os.path.join(user_folder, img_name)
            image = cv2.imread(img_path)
            if image is None:
                print(f"[WARNING] Unable to read image: {img_path}")
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            boxes = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, boxes)

            for enc in encodings:
                known_encodings.append(enc)
                known_names.append(user)

    data = {"encodings": known_encodings, "names": known_names}

    with open("encodings.pickle", "wb") as f:
        pickle.dump(data, f)

    print("[INFO] Face encoding complete and saved to 'encodings.pickle'.")
    return data


# === Step 3: Mark attendance for recognized faces ===
def mark_attendance(name):
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y")
    time_str = now.strftime("%H:%M:%S")

    try:
        df = pd.read_csv("attendance.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    # Check if attendance already marked today for the user
    if not ((df['Name'] == name) & (df['Date'] == date_str)).any():
        new_row = {"Name": name, "Date": date_str, "Time": time_str}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv("attendance.csv", index=False)
        print(f"[INFO] Attendance marked for {name}")


# === Step 4: Recognize faces in real-time from webcam ===
def recognize_faces():
    # Load encodings file
    try:
        with open("encodings.pickle", "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("[ERROR] 'encodings.pickle' not found. Run the encoding step first.")
        return

    cap = cv2.VideoCapture(0)
    print("[INFO] Starting real-time face recognition. Press ESC to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        # Validate frame shape and type
        if frame is None or frame.size == 0:
            print("[WARNING] Empty frame received, skipping...")
            continue
        if len(frame.shape) != 3 or frame.shape[2] != 3:
            print("[WARNING] Frame is not a 3-channel image, skipping...")
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Validate rgb image
        if rgb.dtype != 'uint8' or len(rgb.shape) != 3 or rgb.shape[2] != 3:
            print("[WARNING] Invalid RGB image, skipping...")
            continue

        boxes = face_recognition.face_locations(rgb)
        encodings = face_recognition.face_encodings(rgb, boxes)

        for encoding, box in zip(encodings, boxes):
            matches = face_recognition.compare_faces(data["encodings"], encoding)
            name = "Unknown"

            if True in matches:
                matched_idxs = [i for i, b in enumerate(matches) if b]
                counts = {}

                for i in matched_idxs:
                    matched_name = data["names"][i]
                    counts[matched_name] = counts.get(matched_name, 0) + 1

                name = max(counts, key=counts.get)
                mark_attendance(name)

            top, right, bottom, left = box
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

        cv2.imshow("Face Recognition Attendance", frame)

        if cv2.waitKey(1) == 27:  # ESC key to break
            break

    cap.release()
    cv2.destroyAllWindows()


# === Example usage ===
if __name__ == "__main__":
    # Step 1: Uncomment to register user
    # register_user("Pravallika")

    # Step 2: Encode dataset faces (run after registration)
    encode_faces()

    # Step 3 & 4: Run face recognition and attendance
    recognize_faces()
