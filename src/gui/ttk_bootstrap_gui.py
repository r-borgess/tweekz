import ttkbootstrap as tb
from ttkbootstrap.constants import *
from tkinter import filedialog, messagebox, Label
from PIL import Image, ImageTk
import cv2
from image_controller import ImageProcessor

class ImageEditorApp:
    def __init__(self, root):
        self.root = root
        self.image_processor = ImageProcessor()  # Initialize the image processor
        self.img_label = Label(self.root)
        self.img_label.pack()
        self.setup_ui()

    def setup_ui(self):
        self.root.title("tweekz")
        self.root.state('zoomed')

        menu_bar = tb.Menu(self.root)
        self.root.config(menu=menu_bar)

        file_menu = tb.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open", command=self.open_image)
        file_menu.add_command(label="Save", command=self.save_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)

        tools_menu = tb.Menu(menu_bar, tearoff=False)
        menu_bar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Blackout", command=self.blackout_image)
        tools_menu.add_command(label="Reset", command=self.restore_image)

    def open_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            np_image = self.image_processor.load_image(file_path)
            self.display_image(np_image)

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
        np_image = self.image_processor.blackout_image()
        self.display_image(np_image)

    def restore_image(self):
        np_image = self.image_processor.restore_image()
        self.display_image(np_image)

    def display_image(self, np_image):
        rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_image)
        photo = ImageTk.PhotoImage(image)
        self.img_label.configure(image=photo)
        self.img_label.image = photo  # Keep a reference to avoid garbage collection

def main():
    root = tb.Window(themename="darkly")
    app = ImageEditorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
