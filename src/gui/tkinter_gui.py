import tkinter as tk
from tkinter import filedialog, messagebox, Menu
from PIL import Image, ImageTk
from src import image_processor
import cv2

# Initialize the main application window
root = tk.Tk()
root.title("tweeks")
root.state('zoomed')

image_frame = tk.Frame(root)
image_frame.pack(expand=True, fill=tk.BOTH)


# This canvas will display the image
canvas = tk.Canvas(image_frame)
canvas.pack(fill=tk.BOTH, expand=True)

original_image = None
displayed_image = None  # PIL image to be displayed
photo_image = None  # Tkinter-compatible image

def display_image():
    global displayed_image, photo_image
    if original_image is not None:
        # Get the current zoom level from the slider
        zoom_level = zoom_slider.get() / 100.0
        # Calculate the new size of the image
        size = round(original_image.width * zoom_level), round(original_image.height * zoom_level)
        resized = original_image.resize(size, Image.Resampling.LANCZOS)
        photo_image = ImageTk.PhotoImage(resized)

        # Calculate the center position for the zoom
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        center_x = canvas_width / 2
        center_y = canvas_height / 2

        # Delete the previous image from the canvas
        canvas.delete("all")

        # Create the new image on the canvas, centered
        canvas.create_image(center_x, center_y, image=photo_image, anchor='center')

def gui_load_image():
    global original_image
    file_path = filedialog.askopenfilename(initialdir='./tests/test_images/load')
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_image = Image.fromarray(img)
        display_image()

def gui_save_image():
    global original_image
    initial_directory = './tests/test_images/save'
    file_path = filedialog.asksaveasfilename(initialdir=initial_directory)
    if file_path:
        if not image_processor.save_image(original_image, file_path):
            messagebox.showerror("Save Error", "Failed to save the image.")

def create_menu():
    menubar = Menu(root)
    root.config(menu=menubar)

    # File menu
    file_menu = Menu(menubar, tearoff=0)
    menubar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open...", command=gui_load_image)
    file_menu.add_command(label="Save", command=gui_save_image)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)

    # Add more menus or menu items as needed

create_menu()

# Zoom slider
zoom_slider = tk.Scale(root, from_=5, to=200, orient=tk.HORIZONTAL, command=lambda x: display_image(), showvalue=0)
zoom_slider.set(100)  # Set the initial zoom to 100%
zoom_slider.pack(side=tk.BOTTOM, fill=tk.X)

# Start the Tkinter event loop
root.mainloop()
