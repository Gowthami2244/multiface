import cv2

# Load the pre-trained face classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the input image or video
image_path = 'path_to_image_or_video'
cap = cv2.VideoCapture(image_path)  # Use 0 for webcam

# Initialize face count
face_count = 0

while True:
    # Read the current frame
    ret, frame = cap.read()

    # Break the loop if the video is finished
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the faces and count them
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_count += 1

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()

# Print the total number of faces detected
print("Total faces detected:", face_count)
