import json
import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, Label, Frame, Menu
from PIL import Image, ImageTk
import cv2
from image_controller import ImageProcessor
import threading

class ImageEditorApp:
    def __init__(self, root, config):
        self.root = root
        self.config = config
        self.image_processor = ImageProcessor()
        self.img_container = Frame(self.root)  # Container for the image label
        self.img_container.pack(expand=True, fill="both")  # Center the container frame
        self.img_label = Label(self.img_container)
        self.img_label.place(relx=0.5, rely=0.5, anchor=CENTER)  # Center the label in the container
        self.setup_ui()

    def setup_ui(self):
        self.root.title(self.config["window_title"])
        self.root.state('zoomed')

        menu_bar = Menu(self.root)
        self.root.config(menu=menu_bar)

        file_menu = Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="File", menu=file_menu)
        for option in self.config["menu_options"]["file"]:
            if option == "Exit":
                file_menu.add_command(label=option, command=self.root.quit)
            else:
                # Assuming open_image and save_image methods for simplicity
                file_menu.add_command(label=option, command=getattr(self, option.lower() + "_image"))

        tools_menu = Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)
        for option in self.config["menu_options"]["tools"]:
            tools_menu.add_command(label=option, command=getattr(self, option.lower() + "_image"))


    def open_image(self):
            file_path = filedialog.askopenfilename()
            if file_path:
                try:
                    np_image = self.image_processor.load_image(file_path)
                    self.root.after(0, self.display_image, np_image)
                except Exception as e:
                    messagebox.showerror("Open Image", "Failed to open the image.\n" + str(e))

    def save_image(self):
        if self.image_processor.current_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".jpg")
            if file_path:
                success = self.image_processor.save_image(file_path)
                if success:
                    messagebox.showinfo("Save Image", "Image saved successfully!")
                else:
                    messagebox.showerror("Save Image", "Failed to save the image.")
        else:
            messagebox.showerror("Save Image", "No image to save.")

    def blackout_image(self):
        try:
            np_image = self.image_processor.blackout_image()
            self.root.after(0, self.display_image, np_image)
        except Exception as e:
            messagebox.showerror("Blackout Image", "Failed to blackout the image.\n" + str(e))

    def restore_image(self):
        try:
            np_image = self.image_processor.restore_image()
            self.root.after(0, self.display_image, np_image)
        except Exception as e:
            messagebox.showerror("Restore Image", "Failed to restore the image.\n" + str(e))

    def display_image(self, np_image):
        # Convert the NumPy image to a PIL image
        pil_image = Image.fromarray(cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB))

        # Get screen size and image size
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        img_width, img_height = pil_image.size

        # Calculate the scaling factor to fit the image within the screen size
        scale_width = screen_width / img_width
        scale_height = screen_height / img_height
        scale_factor = min(scale_width, scale_height, 1)  # Ensure scale factor is not more than 1

        # Resize the image if necessary
        if scale_factor < 1:
            new_width = int(img_width * scale_factor)
            new_height = int(img_height * scale_factor)
            pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)

        # Display the image
        photo = ImageTk.PhotoImage(pil_image)
        self.img_label.configure(image=photo)
        self.img_label.image = photo  # Keep a reference


def main():
    # Load configuration from JSON file
    config_file = open('config.json')
    config = json.load(config_file)

    root = tb.Window(themename=config["themename"])
    app = ImageEditorApp(root, config)
    root.mainloop()


if __name__ == "__main__":
    main()
