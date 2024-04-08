import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, Label
from PIL import Image, ImageTk
import cv2
from src import image_processor  # Adjust import path as needed

def main():
    global img_label, current_image, zoom_slider

    root = tb.Window(themename="darkly")
    root.title("tweekz")
    root.state('zoomed')
    current_image = None

    menu_bar = tb.Menu(root)
    root.config(menu=menu_bar)

    file_menu = tb.Menu(menu_bar, tearoff=False)
    menu_bar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="Open", command=lambda: open_image(root))
    file_menu.add_command(label="Save", command=save_image)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)

    tools_menu = tb.Menu(menu_bar, tearoff=False)
    menu_bar.add_cascade(label="Tools", menu=tools_menu)
    tools_menu.add_command(label="Blackout", command=blackout_image)
    tools_menu.add_command(label="Reset", command=restore_image)  # New option for restoring the image

    img_label = Label(root)
    img_label.pack()

    root.mainloop()

def zoom_image(zoom_level):
    global current_image
    if current_image is not None:
        # Convert the zoom_level to a scale factor (zoom_level is a string, so we convert it to float)
        scale_factor = float(zoom_level) / 100

        # Call a function to resize and display the image based on the scale_factor
        resize_and_display_image(current_image, scale_factor)

def open_image(root):
    global current_image
    file_path = filedialog.askopenfilename()
    if file_path:
        current_image = image_processor.load_image(file_path)
        display_image(current_image)

def save_image():
    global current_image
    if current_image is not None:
        file_path = filedialog.asksaveasfilename(defaultextension=".jpg")
        if file_path:
            success = image_processor.save_image(current_image, file_path)
            if success:
                messagebox.showinfo("Save Image", "Image saved successfully!")
            else:
                messagebox.showerror("Save Image", "Failed to save the image.")
    else:
        messagebox.showerror("Save Image", "No image to save.")

def blackout_image():
    global current_image
    if current_image is not None:
        current_image = image_processor.blackout_image(current_image)
        display_image(current_image)

def restore_image():
    global current_image
    if current_image is not None:
        restored_image = image_processor.restore_image()
        if restored_image is not None:
            current_image = restored_image
            display_image(current_image)
        else:
            messagebox.showerror("Restore Image", "No image to restore.")
    else:
        messagebox.showerror("Restore Image", "No image loaded.")

def display_image(np_image):
    global img_label
    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_image)
    photo = ImageTk.PhotoImage(image)
    img_label.configure(image=photo)
    img_label.image = photo


def resize_and_display_image(np_image, scale_factor):
    global img_label
    # Calculate the new size
    new_width = int(np_image.shape[1] * scale_factor)
    new_height = int(np_image.shape[0] * scale_factor)

    # Resize the image
    resized_image = cv2.resize(np_image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Continue with the conversion to RGB and display as before
    rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_image)
    photo = ImageTk.PhotoImage(image)
    img_label.configure(image=photo)
    img_label.image = photo  # Keep a reference

if __name__ == "__main__":
    main()
