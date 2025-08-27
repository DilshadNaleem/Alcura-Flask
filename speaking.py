import tkinter as tk
import speech_recognition as sr
import pyttsx3

# Create the main window
root = tk.Tk()
root.title("Speech to Text")
root.geometry("400x200")

# Initialize the pyttsx3 engine
engine = pyttsx3.init()

# Function to recognize speech, display it, and speak it out
def listen_and_display():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        # Adjust the recognizer sensitivity to ambient noise
        recognizer.adjust_for_ambient_noise(source)
        print("Listening... Speak now!")

        # Record the audio from the microphone with extended timeout and phrase_time_limit
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=10)  # Adjust the timeout and phrase_time_limit

            # Use Google's online speech recognition service
            text = recognizer.recognize_google(audio)
            print(f"You said: {text}")

            # Clear the text box before inserting new text
            text_box.delete(1.0, tk.END)  # Clear existing text
            text_box.insert(tk.END, text)  # Insert the recognized text

            # Speak the recognized text out loud
            engine.say(text)
            engine.runAndWait()

        except sr.UnknownValueError:
            print("Sorry, I could not understand the audio.")
        except sr.RequestError:
            print("Sorry, the speech service is down. Please try again later.")
        except Exception as e:
            print(f"Error: {e}")

# Create a text box in the GUI to show the recognized text
text_box = tk.Text(root, height=5, width=40)
text_box.pack(pady=20)

# Create a button to start listening
listen_button = tk.Button(root, text="Start Listening", command=listen_and_display)
listen_button.pack()

# Run the main loop to display the GUI
root.mainloop()
