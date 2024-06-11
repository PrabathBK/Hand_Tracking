import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
 
genai.configure(api_key="AIzaSyAAL8wvdcYFisZ3fC0_AMmJqMekO-1j2Nc")
model = genai.GenerativeModel('gemini-1.5-flash')
 
# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 600)
 
# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.9, minTrackCon=0.5)
 
def getHandInfo(img):
    hands, img = detector.findHands(img, draw=True, flipType=False)
    if hands:
        hand = hands[0]  # Get the first hand detected
        lmList = hand["lmList"]  # List of 21 landmarks for the first hand
        fingers = detector.fingersUp(hand)
        return fingers, lmList
    else:
        return None
 
def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [1, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]
        if prev_pos is None: prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [0, 0, 0, 0, 0]:
        canvas = np.zeros_like(img)
    return current_pos, canvas
 
def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return None

def display_text_in_window(window_name, text):
    blank_image = np.zeros((200, 1280, 3), np.uint8)
    y0, dy = 50, 30
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(blank_image, line, (10, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(window_name, blank_image)

prev_pos = None
canvas = None
output_text = ""
text_window_name = "AI Output"
text_window_open = False

# Continuously get frames from the webcam
while True:
    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
 
    if canvas is None:
        canvas = np.zeros_like(img)
 
    info = getHandInfo(img)
    if info:
        fingers, lmList = info
        prev_pos, canvas = draw(info, prev_pos, canvas)
        new_output_text = sendToAI(model, canvas, fingers)
        
        if new_output_text:
            output_text = new_output_text
            display_text_in_window(text_window_name, output_text)
        elif cv2.getWindowProperty(text_window_name, cv2.WND_PROP_VISIBLE) >= 1:
            if fingers == [0, 0, 0, 0, 0] or fingers == [1, 1, 1, 1, 0]:
                cv2.destroyWindow(text_window_name)

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    cv2.imshow("Webcam", image_combined)
 
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
