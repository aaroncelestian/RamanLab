import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import subprocess
import os
import threading
import sys
import queue
import time
from PIL import Image, ImageTk

class RamanGUIManager:
    def __init__(self, master):
        self.master = master
        master.title("Raman Spectroscopy Workflow GUI")
        master.geometry("1200x800")  # Increased window size for better visibility

        # --- Add logo at the top ---
        try:
            logo_img = Image.open("nhm_logo.png")
            logo_img = logo_img.resize((250, 250), Image.ANTIALIAS)
            self.logo_photo = ImageTk.PhotoImage(logo_img)
            logo_label = tk.Label(master, image=self.logo_photo, bg="black")
            logo_label.pack(pady=(10, 0))
        except Exception as e:
            print(f"Could not load logo: {e}")
        # --- End logo ---

        # Store directory paths
        self.class_a_dir = ""  # Previously plastic_dir
        self.class_b_dir = ""  # Previously non_plastic_dir
        self.unknown_dir = ""
        
        # Instructions
        tk.Label(master, text="Workflow Steps:", font=("Arial", 16)).pack(pady=10) 