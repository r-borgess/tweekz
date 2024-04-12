from ttkbootstrap.constants import *
import ttkbootstrap as tkb
from tkinter import filedialog, messagebox, Label, Frame, Menu, Toplevel, Entry, Button
from PIL import Image, ImageTk
import cv2
from image_controller import ImageProcessor

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

    def gamma_transform_image(self):
        self.create_gamma_popup()

    def create_gamma_popup(self):
        popup = Toplevel(self.root)
        popup.title("Gamma Transformation")
        popup.geometry("200x100")

        Label(popup, text="Gamma value:").pack(side="top", fill="x", pady=10)

        gamma_value_entry = Entry(popup)
        gamma_value_entry.pack(side="top", fill="x", padx=60)

        apply_button = Button(popup, text="Apply", command=lambda: self.apply_gamma_and_close_popup(gamma_value_entry.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def apply_gamma_and_close_popup(self, gamma_value, popup):
        try:
            gamma_value = float(gamma_value)
            np_image = self.image_processor.gamma_transform_image(gamma_value)
            self.root.after(0, self.display_image, np_image)
            popup.destroy()
        except ValueError:
            messagebox.showerror("Gamma Transform", "Invalid gamma value. Please enter a valid number.")
        except Exception as e:
            messagebox.showerror("Gamma Transform", "Failed to transform the image.\n" + str(e))
            popup.destroy()

    def contrast_stretch_image(self):
        self.create_contrast_popup()

    def create_contrast_popup(self):
        popup = Toplevel(self.root)
        popup.title("Contrast Stretching")
        popup.geometry("200x300")

        Label(popup, text="r1 value:").pack(side="top", fill="x", pady=10)
        r1_entry = Entry(popup)
        r1_entry.pack(side="top", fill="x", padx=60)
        Label(popup, text="s1 value:").pack(side="top", fill="x", pady=10)
        s1_entry = Entry(popup)
        s1_entry.pack(side="top", fill="x", padx=60)
        Label(popup, text="r2 value:").pack(side="top", fill="x", pady=10)
        r2_entry = Entry(popup)
        r2_entry.pack(side="top", fill="x", padx=60)
        Label(popup, text="s2 value:").pack(side="top", fill="x", pady=10)
        s2_entry = Entry(popup)
        s2_entry.pack(side="top", fill="x", padx=60)

        apply_button = Button(popup, text="Apply", command=lambda: self.apply_contrast_and_close_popup(r1_entry.get(), s1_entry.get(), r2_entry.get(), s2_entry.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def apply_contrast_and_close_popup(self, r1_value, s1_value, r2_value, s2_value, popup):
        try:
            r1_value = float(r1_value)
            s1_value = float(s1_value)
            r2_value = float(r2_value)
            s2_value = float(s2_value)
            np_image = self.image_processor.contrast_stretch_image(r1_value, s1_value, r2_value, s2_value)
            self.root.after(0, self.display_image, np_image)
            popup.destroy()
        except ValueError:
            messagebox.showerror("Contrast Stretch", "Invalid values. Please enter a valid number.")
        except Exception as e:
            messagebox.showerror("Contrast Stretch", "Failed to transform the image.\n" + str(e))
            popup.destroy()

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

