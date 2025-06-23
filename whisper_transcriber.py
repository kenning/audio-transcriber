#!/usr/bin/env python3
"""
Whisper Audio Transcriber - A lightweight desktop app for audio transcription
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import queue
import time
import tempfile
import os
import pyperclip
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from pathlib import Path

# Try to import whisper, handle if not installed
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False


class WhisperTranscriber:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Whisper Transcriber")
        self.root.geometry("600x500")
        self.root.resizable(True, True)

        # Application state
        self.model = None
        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000  # Whisper works best with 16kHz
        self.recording_thread = None
        self.transcription_queue = queue.Queue()

        # Create GUI
        self.create_widgets()

        # Start checking for transcription results
        self.check_transcription_queue()

        # Check if whisper is available
        if not WHISPER_AVAILABLE:
            messagebox.showerror(
                "Missing Dependencies",
                "Whisper is not installed. Please install it with:\npip install openai-whisper",
            )

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Configure grid weight
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # Model section
        model_frame = ttk.LabelFrame(main_frame, text="Model Management", padding="10")
        model_frame.grid(
            row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10)
        )
        model_frame.columnconfigure(1, weight=1)

        ttk.Label(model_frame, text="Model:").grid(
            row=0, column=0, sticky=tk.W, padx=(0, 10)
        )

        self.model_status = ttk.Label(model_frame, text="Not loaded", foreground="red")
        self.model_status.grid(row=0, column=1, sticky=tk.W)

        self.load_button = ttk.Button(
            model_frame, text="Load Model", command=self.load_model
        )
        self.load_button.grid(row=0, column=2, padx=(10, 0))

        self.unload_button = ttk.Button(
            model_frame,
            text="Unload Model",
            command=self.unload_model,
            state="disabled",
        )
        self.unload_button.grid(row=0, column=3, padx=(10, 0))

        # Recording section
        record_frame = ttk.LabelFrame(main_frame, text="Recording", padding="10")
        record_frame.grid(
            row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10)
        )

        self.record_button = ttk.Button(
            record_frame,
            text="🎤 Start Recording",
            command=self.toggle_recording,
            state="disabled",
        )
        self.record_button.grid(row=0, column=0, padx=(0, 10))

        self.recording_status = ttk.Label(record_frame, text="Ready to record")
        self.recording_status.grid(row=0, column=1, sticky=tk.W)

        # Progress bar
        self.progress = ttk.Progressbar(record_frame, mode="indeterminate")
        self.progress.grid(
            row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0)
        )

        # Transcription section
        transcription_frame = ttk.LabelFrame(
            main_frame, text="Transcription", padding="10"
        )
        transcription_frame.grid(
            row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10)
        )
        transcription_frame.columnconfigure(0, weight=1)
        transcription_frame.rowconfigure(0, weight=1)

        self.transcription_text = scrolledtext.ScrolledText(
            transcription_frame, height=10, wrap=tk.WORD
        )
        self.transcription_text.grid(
            row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10)
        )

        # Buttons frame
        buttons_frame = ttk.Frame(transcription_frame)
        buttons_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        buttons_frame.columnconfigure(0, weight=1)

        self.copy_button = ttk.Button(
            buttons_frame, text="📋 Copy to Clipboard", command=self.copy_to_clipboard
        )
        self.copy_button.grid(row=0, column=0, padx=(0, 10))

        self.clear_button = ttk.Button(
            buttons_frame, text="🗑️ Clear", command=self.clear_transcription
        )
        self.clear_button.grid(row=0, column=1)

        # Status bar
        self.status_bar = ttk.Label(
            main_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W
        )
        self.status_bar.grid(
            row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0)
        )

        # Configure grid weights for resizability
        main_frame.rowconfigure(2, weight=1)

    def load_model(self):
        """Load the Whisper tiny model in a separate thread"""
        if not WHISPER_AVAILABLE:
            messagebox.showerror("Error", "Whisper is not installed!")
            return

        def load_model_thread():
            try:
                self.update_status("Loading Whisper tiny model...")
                self.load_button.config(state="disabled")
                self.progress.start()

                # Load the model
                self.model = whisper.load_model("tiny")

                # Update UI
                self.root.after(0, self.model_loaded_successfully)

            except Exception as e:
                self.root.after(0, lambda: self.model_load_failed(str(e)))

        threading.Thread(target=load_model_thread, daemon=True).start()

    def model_loaded_successfully(self):
        """Called when model loading succeeds"""
        self.progress.stop()
        self.model_status.config(text="Loaded (tiny)", foreground="green")
        self.load_button.config(state="disabled")
        self.unload_button.config(state="normal")
        self.record_button.config(state="normal")
        self.update_status("Model loaded successfully! Ready to transcribe.")

    def model_load_failed(self, error):
        """Called when model loading fails"""
        self.progress.stop()
        self.load_button.config(state="normal")
        self.update_status("Failed to load model")
        messagebox.showerror("Error", f"Failed to load model: {error}")

    def unload_model(self):
        """Unload the model and free memory"""
        self.model = None
        self.model_status.config(text="Not loaded", foreground="red")
        self.load_button.config(state="normal")
        self.unload_button.config(state="disabled")
        self.record_button.config(state="disabled")
        self.update_status("Model unloaded")

    def toggle_recording(self):
        """Start or stop audio recording"""
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        """Start recording audio"""
        if not self.model:
            messagebox.showerror("Error", "Please load the model first!")
            return

        self.is_recording = True
        self.audio_data = []
        self.record_button.config(text="🛑 Stop Recording")
        self.recording_status.config(text="Recording... (press Stop when done)")
        self.update_status("Recording audio...")

        def record_audio():
            """Record audio in a separate thread"""
            try:
                # Start recording
                with sd.InputStream(
                    samplerate=self.sample_rate, channels=1, dtype=np.float32
                ) as stream:
                    while self.is_recording:
                        data, overflowed = stream.read(1024)
                        if overflowed:
                            print("Audio buffer overflowed")
                        self.audio_data.append(data.flatten())
                        time.sleep(0.01)  # Small delay to prevent excessive CPU usage

            except Exception as e:
                self.root.after(
                    0,
                    lambda: messagebox.showerror(
                        "Recording Error", f"Failed to record audio: {e}"
                    ),
                )
                self.root.after(0, self.stop_recording)

        self.recording_thread = threading.Thread(target=record_audio, daemon=True)
        self.recording_thread.start()

    def stop_recording(self):
        """Stop recording and transcribe the audio"""
        if not self.is_recording:
            return

        self.is_recording = False
        self.record_button.config(text="🎤 Start Recording")
        self.recording_status.config(text="Processing...")
        self.update_status("Processing audio...")
        self.progress.start()

        # Transcribe in a separate thread
        threading.Thread(target=self.transcribe_audio, daemon=True).start()

    def transcribe_audio(self):
        """Transcribe the recorded audio"""
        try:
            if not self.audio_data:
                self.transcription_queue.put(("error", "No audio data recorded"))
                return

            # Combine audio data
            audio_array = np.concatenate(self.audio_data)

            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                wav.write(
                    temp_file.name,
                    self.sample_rate,
                    (audio_array * 32767).astype(np.int16),
                )
                temp_filename = temp_file.name

            try:
                # Transcribe with Whisper
                result = self.model.transcribe(temp_filename)
                transcription = result["text"].strip()

                if transcription:
                    self.transcription_queue.put(("success", transcription))
                else:
                    self.transcription_queue.put(("error", "No speech detected"))

            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_filename)
                except:
                    pass

        except Exception as e:
            self.transcription_queue.put(("error", f"Transcription failed: {e}"))

    def check_transcription_queue(self):
        """Check for transcription results and update UI"""
        try:
            while True:
                result_type, message = self.transcription_queue.get_nowait()

                self.progress.stop()
                self.recording_status.config(text="Ready to record")

                if result_type == "success":
                    # Add transcription to text widget
                    self.transcription_text.insert(tk.END, message + "\n\n")
                    self.transcription_text.see(tk.END)

                    # Automatically copy to clipboard
                    try:
                        pyperclip.copy(message)
                        self.update_status(f"Transcribed and copied to clipboard!")
                    except Exception as e:
                        self.update_status(f"Transcribed (clipboard copy failed: {e})")

                elif result_type == "error":
                    self.update_status(f"Error: {message}")
                    messagebox.showerror("Transcription Error", message)

        except queue.Empty:
            pass

        # Schedule next check
        self.root.after(100, self.check_transcription_queue)

    def copy_to_clipboard(self):
        """Copy current transcription to clipboard"""
        text = self.transcription_text.get(1.0, tk.END).strip()
        if text:
            try:
                pyperclip.copy(text)
                self.update_status("Copied to clipboard!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy to clipboard: {e}")
        else:
            messagebox.showwarning("Warning", "No text to copy!")

    def clear_transcription(self):
        """Clear the transcription text"""
        self.transcription_text.delete(1.0, tk.END)
        self.update_status("Transcription cleared")

    def update_status(self, message):
        """Update the status bar"""
        self.status_bar.config(text=message)

    def run(self):
        """Start the application"""
        self.root.mainloop()


def main():
    """Main entry point"""
    app = WhisperTranscriber()
    app.run()


if __name__ == "__main__":
    main()
