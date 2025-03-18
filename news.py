import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter
import cv2
from PIL import Image, ImageTk
import joblib
import numpy as np
import speech_recognition as sr
import nltk
from nltk.corpus import stopwords
import re
from pydub import AudioSegment
import os
from customtkinter import CTkImage
import threading

nltk.download('stopwords')

# Load your models
naivebayes_model = "nb_model.pkl"
vectorizer_filename = "vectorizer (13).pkl"

try:
    model = joblib.load(naivebayes_model)
    vectorizer = joblib.load(vectorizer_filename)
except FileNotFoundError:
    print("Model or vectorizer file not found. Ensure the .pkl files are in the same directory.")
    exit()

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    text = [word for word in text if word not in stop_words]
    return ' '.join(text)

def predict_news(text):
    if not text:
        return "No text extracted"
    preprocessed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([preprocessed_text])
    prediction = model.predict(vectorized_text)
    return "Real News" if prediction[0] == 1 else "Fake News"

def extract_text_from_video(video_file):
    recognizer = sr.Recognizer()
    try:
        audio = AudioSegment.from_file(video_file)
        audio = audio.set_channels(1).set_frame_rate(16000)
        temp_audio_file = "temp_audio.wav"
        audio.export(temp_audio_file, format="wav")

        with sr.AudioFile(temp_audio_file) as source:
            audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Audio unclear, unable to extract text."
    except sr.RequestError as e:
        return f"Error with speech recognition service: {e}"
    except Exception as e:
        return f"Error: {e}"
    finally:
        if os.path.exists(temp_audio_file):
            os.remove(temp_audio_file)

def animate_prediction_label(prediction_text, text_color):
    result_label.configure(text="", text_color=text_color)

    def fade_in_and_scale():
        opacity = 0.0
        scale_factor = 1.0

        def animate_step():
            nonlocal opacity, scale_factor
            if opacity < 1.0:
                opacity += 0.05
                scale_factor += 0.02
                result_label.configure(text=prediction_text, font=("Arial", int(14 * scale_factor)), text_color=text_color)
                result_label.update_idletasks()
                result_label.after(30, animate_step)

        animate_step()

    fade_in_and_scale()

def open_video():
    global video_file
    video_file = filedialog.askopenfilename(filetypes=[('Video Files', '*.mp4 *.avi *.mov *.mkv'), ('All Files', '*.*')])
    if not video_file:
        return

    try:
        video_capture = cv2.VideoCapture(video_file)
        video_thread = threading.Thread(target=play_video, args=(video_capture,))
        video_thread.start()

        progress_bar.start()
        video_text = extract_text_from_video(video_file)
        progress_bar.stop()

        if video_text:
            prediction = predict_news(video_text)
            text_color = "#00FF00" if "Real" in prediction else "#FF0000"
            animate_prediction_label(f"Prediction: {prediction}", text_color)
        else:
            animate_prediction_label("No text extracted.", "#FF0000")

    except Exception as e:
        messagebox.showerror("Error", f"Error loading the video: {e}")

def apply_rounded_corners(frame, radius=30):
    rows, cols = frame.shape[:2]
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.rectangle(mask, (radius, 0), (cols-radius, rows), 255, -1)
    cv2.rectangle(mask, (0, radius), (cols, rows-radius), 255, -1)
    cv2.circle(mask, (radius, radius), radius, 255, -1)
    cv2.circle(mask, (cols-radius, radius), radius, 255, -1)
    cv2.circle(mask, (radius, rows-radius), radius, 255, -1)
    cv2.circle(mask, (cols-radius, rows-radius), radius, 255, -1)
    
    # Apply the mask to the frame
    frame = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Add transparency to the corners (optional)
    frame_with_alpha = cv2.merge((frame, mask))
    return frame_with_alpha

def play_video(video_capture):
    ret, frame = video_capture.read()
    if ret:
        frame = cv2.resize(frame, (600, 500))
        frame_with_corners = apply_rounded_corners(frame)  # Apply rounded corners
        
        image = cv2.cvtColor(frame_with_corners, cv2.COLOR_BGRA2RGBA)
        image = Image.fromarray(image)
        ctk_image = CTkImage(light_image=image, size=(600, 400))

        label.configure(image=ctk_image)
        label.image = ctk_image

        label.after(10, play_video, video_capture)
    else:
        video_capture.release()
        label.configure(text="End of Video")

def switch_to_video_frame():
    info_frame.pack_forget()
    frame_1.pack(pady=20, padx=20, fill="both", expand=True)

def switch_to_info_frame():
    frame_1.pack_forget()
    info_frame.pack(pady=20, padx=20, fill="both", expand=True)

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

app = customtkinter.CTk()
app.geometry("800x600")
app.title("Video Player with Fake News Detection")


# Background Image
background_image = Image.open("img.jpg")
bg_image = ImageTk.PhotoImage(background_image.resize((1500, 1500), Image.Resampling.LANCZOS))
background_label = tk.Label(app, image=bg_image)
background_label.place(relwidth=1, relheight=1)

# Info Frame
info_frame = customtkinter.CTkFrame(master=app, corner_radius=15, fg_color=("Grey", "Black"))  # Dark blue-gray gradient
info_frame.pack(pady=20, padx=20, fill="both", expand=True)



university_label = customtkinter.CTkLabel(
    master=info_frame,
    text="Sir Syed University Of Engineering & Technology\nSoftware Engineering Department",
    font=("Arial", 18, "bold"),
    text_color="white",
    anchor="center"
)
university_label.pack(pady=(10, 15))

course_label = customtkinter.CTkLabel(
    master=info_frame,
    text="Course Name: Artificial Intelligence",
    font=("Arial", 18, "italic"),
    text_color="white",
    anchor="center"
)
course_label.pack(pady=(5, 15))

details_label = customtkinter.CTkLabel(
    master=info_frame,
    text="Batch: 2022\nSection: C\nSemester: 5",
    font=("Arial", 18),
    text_color="white",
    anchor="center"
)
details_label.pack(pady=(5, 15))

project_title_label = customtkinter.CTkLabel(
    master=info_frame,
    text="Fake News Detection",
    font=("Arial", 24, "bold"),
    text_color="white",
    anchor="center"
)
project_title_label.pack(pady=(15, 20))



# Button to Continue
continue_button = customtkinter.CTkButton(
    master=info_frame,
    text="Continue",
    command=switch_to_video_frame,
    hover_color="#1abc9c",
    font=("Arial", 18)
)
continue_button.pack(pady=20)


# Open the logo image
logo_image_university = Image.open("University_logo.jpg")  # Replace with your logo file path
logo_image_university = logo_image_university.resize((150, 150))
logo_photo_university = ImageTk.PhotoImage(logo_image_university)
logo_label_university = customtkinter.CTkLabel(master=info_frame, image=logo_photo_university, text="")
logo_label_university.image = logo_photo_university
logo_label_university.place(relx=0.0, rely=0.0, anchor='nw', x=10, y=10)  # Top left corner with some padding

# Place the department logo on the top right of the info frame
logo_image_department = Image.open("Dpartment_logo.jpg")  # Replace with your logo file path
logo_image_department = logo_image_department.resize((150, 150))
logo_photo_department = ImageTk.PhotoImage(logo_image_department)
logo_label_department = customtkinter.CTkLabel(master=info_frame, image=logo_photo_department, text="")
logo_label_department.image = logo_photo_department
logo_label_department.place(relx=1.0, rely=0.0, anchor='ne', x=-10, y=10) 


# Video Player Frame
frame_1 = customtkinter.CTkFrame(master=app, corner_radius=15, fg_color="transparent")
frame_1.pack_forget()

back_button = customtkinter.CTkButton(
    master=frame_1, text="←", corner_radius=8, command=switch_to_info_frame, hover_color="#1abc9c"
)
back_button.place(relx=0.0, rely=0.0, anchor='nw', x=10, y=10)  # Place the button in the top-left corner with padding


button_1 = customtkinter.CTkButton(master=frame_1, text="Open Video", corner_radius=8, command=open_video, hover_color="#1abc9c")
button_1.pack(pady=10, padx=10)

label = customtkinter.CTkLabel(master=frame_1, text="Video Frame")
label.pack(expand=True, fill="both", padx=10, pady=10)

progress_bar = customtkinter.CTkProgressBar(master=frame_1)
progress_bar.pack(pady=10, padx=10)
progress_bar.set(0)

result_label = customtkinter.CTkLabel(
    master=frame_1, text="Prediction: ", font=("Arial", 14), anchor="center"
)
result_label.pack(pady=10, fill="x", padx=10)


app.mainloop()
