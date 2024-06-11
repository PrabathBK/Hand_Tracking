import streamlit as st
import cv2
import numpy as np
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
import google.generativeai as genai

# Configure the AI model
genai.configure(api_key="AIzaSyAAL8wvdcYFisZ3fC0_AMmJqMekO-1j2Nc")
model = genai.GenerativeModel('gemini-1.5-flash')

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
        cv2.line(canvas, tuple(current_pos), tuple(prev_pos), (255, 0, 255), 10)
    elif fingers == [0, 0, 0, 0, 0]:
        canvas = np.zeros_like(img)
    return current_pos, canvas

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        return response.text
    return None

# Initialize Streamlit
st.title("Hand Gesture Math Solver with AI")
stframe = st.empty()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

prev_pos = None
canvas = None
output_text = ""

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

    image_combined = cv2.addWeighted(img, 0.7, canvas, 0.3, 0)
    stframe.image(image_combined, channels="BGR")

    if output_text:
        st.text_area("AI Output", value=output_text, height=200)

cap.release()
