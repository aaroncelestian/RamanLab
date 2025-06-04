import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import subprocess
import os
import threading
import sys
import queue
import time
import re

def run_script(script_name, args=None, output_widget=None, progress_widget=None, progress_var=None):
    """Runs a Python script and captures its output."""
    if output_widget:
        output_widget.insert(tk.END, f"Running {script_name}...\n")
        output_widget.see(tk.END)
    
    command = ["python", script_name]
    if args:
        command.extend(args)
    
    try:
        # Create a process with pipes for stdout and stderr
        process = subprocess.Popen(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True, 
            bufsize=1, 
            universal_newlines=True
        )
        
        # Non-blocking read of stdout and stderr with improved real-time display
        def stream_output(stream, prefix, output_queue):
            """Read from stream and put lines into the queue with a prefix."""
            for line in iter(stream.readline, ''):
                # Filter out read errors
                line_stripped = line.strip()
                if line_stripped.startswith("read error"):
                    continue
                if line_stripped.startswith("Processing files:"):
                    # Extract progress information
                    match = re.search(r'(\d+)%\|.*?(\d+)/(\d+)', line_stripped)
                    if match and progress_widget and progress_var:
                        percent = int(match.group(1))
                        current = int(match.group(2))
                        total = int(match.group(3))
                        # Update progress bar and label using the main thread
                        def update_progress():
                            progress_widget.configure(value=percent)
                            progress_var.set(f"Processing: {current}/{total} files ({percent}%)")
                        if output_widget:
                            output_widget.after(0, update_progress)
                    continue
                output_queue.put((prefix, line))
            stream.close()
            
        # Create a queue for thread-safe output handling
        output_queue = queue.Queue()
        
        # Start threads to read stdout and stderr
        stdout_thread = threading.Thread(
            target=stream_output, 
            args=(process.stdout, "", output_queue),  # No prefix for stdout
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=stream_output, 
            args=(process.stderr, "ERROR: ", output_queue),  # Only prefix stderr
            daemon=True
        )
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Function to update UI with output from the queue
        def update_output():
            # Process all current items in the queue
            try:
                while True:  # Process all available items
                    prefix, line = output_queue.get_nowait()
                    if output_widget:
                        output_widget.insert(tk.END, f"{prefix}{line}")
                        output_widget.see(tk.END)
                    output_queue.task_done()
            except queue.Empty:
                pass  # Queue is empty, continue
            
            # Schedule to check queue again after a short delay if process is still running
            if process.poll() is None or not output_queue.empty():
                output_widget.after(100, update_output)
            else:  # Process has completed and queue is empty
                stdout_thread.join()
                stderr_thread.join()
                
                if output_widget:
                    output_widget.insert(tk.END, f"{script_name} finished with exit code {process.returncode}.\n")
                    output_widget.see(tk.END)
                
                if progress_widget:
                    def update_complete():
                        progress_widget.configure(value=100)
                        progress_var.set("Complete")
                    output_widget.after(0, update_complete)
                
                if process.returncode != 0:
                    messagebox.showerror("Script Error", f"{script_name} failed. Check output for details.")
                elif "visualize" not in script_name.lower():  # Only show success message for non-visualization scripts
                    messagebox.showinfo("Script Complete", f"{script_name} completed successfully.")
        
        # Start output processing if we have a widget
        if output_widget:
            output_widget.after(100, update_output)
        else:  # Fallback if no output widget
            process.wait()  # Block until completion
            if process.returncode != 0:
                messagebox.showerror("Script Error", f"{script_name} failed.")
            elif "visualize" not in script_name.lower():  # Only show success message for non-visualization scripts
                messagebox.showinfo("Script Complete", f"{script_name} completed successfully.")

    except FileNotFoundError:
        messagebox.showerror("Error", f"Python interpreter or script {script_name} not found.")
        if output_widget:
            output_widget.insert(tk.END, f"Error: Python interpreter or script {script_name} not found.\n")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
        if output_widget:
            output_widget.insert(tk.END, f"An error occurred: {str(e)}\n")

class RamanGUIManager:
    def __init__(self, master):
        self.master = master
        master.title("Raman Map Component Analysis Workflow")
        master.geometry("800x900")  # window size
        
        # Initialize directory paths
        self.base_dir = ""
        self.class_a_dir = ""
        self.class_b_dir = ""
        self.unknown_dir = ""
        
        # Instructions
        tk.Label(master, text="Workflow Steps:", font=("Arial", 16)).pack(pady=10)

        # Step 1: Data Organization
        data_org_frame = tk.LabelFrame(master, text="Step 1: Data Organization", padx=10, pady=10)
        data_org_frame.pack(padx=10, pady=5, fill="x")
        
        # Directory selection instructions
        tk.Label(data_org_frame, justify=tk.LEFT, text=(
            "Select the main directory containing your Raman spectra folders:\n"
            "  - Class A spectra: Known as the positive samples\n"
            "  - Class B spectra: Known as the negative samples\n"
            "  - Unknown spectra: Samples to be classified")).pack(pady=5)
        
        # Directory selection button
        dir_button_frame = tk.Frame(data_org_frame)
        dir_button_frame.pack(fill="x", pady=5)
        
        tk.Label(dir_button_frame, text="Main Directory:", width=20, anchor="w").pack(side=tk.LEFT)
        self.base_dir_label = tk.Label(dir_button_frame, text="Not selected", anchor="w")
        self.base_dir_label.pack(side=tk.LEFT, fill="x", expand=True)
        tk.Button(dir_button_frame, text="Select", command=self.select_base_dir, width=10).pack(side=tk.RIGHT)
        
        # Status labels for subdirectories
        self.status_frame = tk.Frame(data_org_frame)
        self.status_frame.pack(fill="x", pady=5)
        
        # Class A status
        class_a_status = tk.Frame(self.status_frame)
        class_a_status.pack(fill="x", pady=2)
        tk.Label(class_a_status, text="Class A Directory:", width=20, anchor="w").pack(side=tk.LEFT)
        self.class_a_status = tk.Label(class_a_status, text="Not found", fg="red", anchor="w")
        self.class_a_status.pack(side=tk.LEFT, fill="x", expand=True)
        
        # Class B status
        class_b_status = tk.Frame(self.status_frame)
        class_b_status.pack(fill="x", pady=2)
        tk.Label(class_b_status, text="Class B Directory:", width=20, anchor="w").pack(side=tk.LEFT)
        self.class_b_status = tk.Label(class_b_status, text="Not found", fg="red", anchor="w")
        self.class_b_status.pack(side=tk.LEFT, fill="x", expand=True)
        
        # Unknown status
        unknown_status = tk.Frame(self.status_frame)
        unknown_status.pack(fill="x", pady=2)
        tk.Label(unknown_status, text="Unknown Directory:", width=20, anchor="w").pack(side=tk.LEFT)
        self.unknown_status = tk.Label(unknown_status, text="Not found", fg="red", anchor="w")
        self.unknown_status.pack(side=tk.LEFT, fill="x", expand=True)
        
        # Button frames with fixed widths
        button_frames_container = tk.Frame(master)
        button_frames_container.pack(padx=10, pady=5, fill="x")
        
        # Step 2: Model Training
        train_frame = tk.LabelFrame(button_frames_container, text="Step 2: Train Classifier Model", padx=10, pady=10)
        train_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5)
        train_button = tk.Button(train_frame, text="Train Model", command=self.train_model, width=15, height=2)
        train_button.pack(pady=5)
        
        # Step 3: Predict Unknown Samples
        predict_frame = tk.LabelFrame(button_frames_container, text="Step 3: Predict Unknown Samples", padx=10, pady=10)
        predict_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5)
        predict_button = tk.Button(predict_frame, text="Predict Unknown", command=self.predict_unknown, width=15, height=2)
        predict_button.pack(pady=5)
        
        # Step 4: Visualize Results
        visualize_frame = tk.LabelFrame(button_frames_container, text="Step 4: Visualize Results", padx=10, pady=10)
        visualize_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5)
        visualize_button = tk.Button(visualize_frame, text="Show Grid Plot", command=self.visualize_results, width=15, height=2)
        visualize_button.pack(pady=5)
        
        # Output Console
        console_frame = tk.LabelFrame(master, text="Output Console", padx=10, pady=10)
        console_frame.pack(padx=10, pady=5, fill="both", expand=True)
        
        # Add a clear button for the console
        clear_button = tk.Button(console_frame, text="Clear Console", command=self.clear_console, width=15, height=1)
        clear_button.pack(anchor="ne", padx=5, pady=5)
        
        # Scrolled text widget for output
        self.output_text = scrolledtext.ScrolledText(console_frame, wrap=tk.WORD, width=80, height=20)
        self.output_text.pack(fill="both", expand=True)
        
        # Progress bar
        self.progress_var = tk.StringVar(value="Ready")
        self.progress_bar = ttk.Progressbar(console_frame, mode='determinate', length=300)
        self.progress_bar.pack(fill="x", padx=5, pady=5)
        self.progress_label = tk.Label(console_frame, textvariable=self.progress_var)
        self.progress_label.pack(pady=2)
        
        # Configure tags for different types of output
        self.output_text.tag_configure("info", foreground="black")
        self.output_text.tag_configure("error", foreground="red")
        self.output_text.tag_configure("success", foreground="green")
        self.output_text.tag_configure("warning", foreground="orange")
        
        # Print welcome message
        self.output_text.insert(tk.END, "Welcome to Raman Spectroscopy Map Component Analysis Workflow GUI\n", "info")
        self.output_text.insert(tk.END, "Please follow the workflow steps from top to bottom.\n\n", "info")
        self.output_text.see(tk.END)
    
    def clear_console(self):
        """Clear the output console."""
        self.output_text.delete(1.0, tk.END)
        self.output_text.insert(tk.END, "Console cleared.\n", "info")
        self.output_text.see(tk.END)
        self.progress_bar['value'] = 0
        self.progress_var.set("Ready")
    
    def select_base_dir(self):
        """Select the main directory containing all required folders."""
        dir_path = filedialog.askdirectory(title="Select Main Directory")
        if dir_path:
            self.base_dir = dir_path
            self.base_dir_label.config(text=dir_path)
            
            # Look for required directories
            self.find_required_directories()
            
    def find_required_directories(self):
        """Find the required directories within the base directory."""
        # Common variations of directory names
        class_a_variations = ['class_a', 'classa', 'positive', 'pos']
        class_b_variations = ['class_b', 'classb', 'negative', 'neg']
        unknown_variations = ['unknown', 'test', 'predict']
        
        # Reset paths
        self.class_a_dir = ""
        self.class_b_dir = ""
        self.unknown_dir = ""
        
        # Find directories
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                item_lower = item.lower()
                
                # Check for Class A
                if any(var in item_lower for var in class_a_variations):
                    self.class_a_dir = item_path
                    self.class_a_status.config(text=f"Found: {item}", fg="green")
                
                # Check for Class B
                elif any(var in item_lower for var in class_b_variations):
                    self.class_b_dir = item_path
                    self.class_b_status.config(text=f"Found: {item}", fg="green")
                
                # Check for Unknown
                elif any(var in item_lower for var in unknown_variations):
                    self.unknown_dir = item_path
                    self.unknown_status.config(text=f"Found: {item}", fg="green")
        
        # Update status for not found directories
        if not self.class_a_dir:
            self.class_a_status.config(text="Not found", fg="red")
        if not self.class_b_dir:
            self.class_b_status.config(text="Not found", fg="red")
        if not self.unknown_dir:
            self.unknown_status.config(text="Not found", fg="red")
            
        # Show message if any required directory is missing
        if not all([self.class_a_dir, self.class_b_dir, self.unknown_dir]):
            messagebox.showwarning("Missing Directories", 
                "Some required directories were not found. Please ensure your directory structure includes:\n"
                "- A directory for Class A/positive samples\n"
                "- A directory for Class B/negative samples\n"
                "- A directory for unknown/test samples")

    def train_model(self):
        """Train the model using the selected directories."""
        if not self.class_a_dir or not self.class_b_dir:
            messagebox.showerror("Error", "Please select both Class A and Class B directories first.")
            return
            
        # Prepare command and arguments
        script_path = os.path.join(os.path.dirname(__file__), "retrain_model.py")
        args = [self.class_a_dir, self.class_b_dir]
        
        self.output_text.insert(tk.END, "\n=== Starting Model Training ===\n", "success")
        self.output_text.insert(tk.END, "This may take several minutes. Progress will be shown below.\n", "warning")
        self.output_text.see(tk.END)
        
        # Reset progress bar
        self.progress_bar['value'] = 0
        self.progress_var.set("Starting training...")
        
        threading.Thread(target=run_script, args=(script_path, args, self.output_text, self.progress_bar, self.progress_var), daemon=True).start()

    def predict_unknown(self):
        if not self.unknown_dir:
            messagebox.showerror("Error", "Please select the unknown spectra directory first.")
            return
            
        script_path = "test_unknown.py"
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"{script_path} not found in the current directory.")
            return
        
        self.output_text.insert(tk.END, "\n=== Starting Unknown Prediction ===\n", "success")
        self.output_text.see(tk.END)
        
        # Reset progress bar
        self.progress_bar['value'] = 0
        self.progress_var.set("Starting prediction...")
        
        # Pass the unknown directory path as an argument to the script
        args = [self.unknown_dir]
        threading.Thread(target=run_script, args=(script_path, args, self.output_text, self.progress_bar, self.progress_var), daemon=True).start()

    def visualize_results(self):
        """Run the visualization script to create a spatial plot of classification results."""
        if not os.path.exists("unknown_spectra_results.csv"):
            messagebox.showerror("Error", "No classification results found. Please run the prediction step first.")
            return
            
        script_path = "visualize_spatial_results.py"
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"{script_path} not found in the current directory.")
            return
        
        self.output_text.insert(tk.END, "\n=== Creating Spatial Distribution Visualization ===\n", "success")
        self.output_text.see(tk.END)
        
        # Reset progress bar
        self.progress_bar['value'] = 0
        self.progress_var.set("Creating visualization...")
        
        # Run the visualization script
        threading.Thread(target=run_script, args=(script_path, None, self.output_text, self.progress_bar, self.progress_var), daemon=True).start()


if __name__ == "__main__":
    root = tk.Tk()
    gui = RamanGUIManager(root)
    root.mainloop()