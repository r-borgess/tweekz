from ttkbootstrap.constants import *
import ttkbootstrap as tkb
from tkinter import filedialog, messagebox, Label, Frame, Menu, Toplevel, Entry, Button, StringVar, OptionMenu, Text
from tkinter.ttk import Combobox
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import cv2
import numpy as np
from image_controller import ImageProcessor
import logging

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
        self.notch_points = []
        self.notch_reject_active = False
        self.is_fft_image = False
        self.fft = None
        self.fft_phase_angle = None

    def setup_logging():
        logging.basicConfig(level=logging.DEBUG, filename='app.log', filemode='w',
                            format='%(name)s - %(levelname)s - %(message)s')

    def handle_error(self, error_message, exception=None):
        if exception:
            logging.error(f"{error_message}: {str(exception)}")
        else:
            logging.error(error_message)
        messagebox.showerror("Error", error_message)

    def setup_ui(self):
        self.root.title(self.config["window_title"])
        self.root.state('zoomed')
        menu_bar = Menu(self.root)
        self.root.config(menu=menu_bar)
        self.create_menus(menu_bar, self.config['menu_options'])

        self.img_label.bind("<Button-1>", self.on_image_click)
        self.root.bind("<Return>", self.on_enter_pressed)

    def create_menus(self, menu_bar, menu_config):
        for menu_name, items in menu_config.items():
            if isinstance(items, list):
                menu = Menu(menu_bar, tearoff=False)
                menu_bar.add_cascade(label=menu_name.replace('_', ' '), menu=menu)
                for item in items:
                    if item == "Exit":
                        menu.add_command(label=item.replace('_', ' '), command=self.root.quit)
                    elif item == "Notch Reject":
                        menu.add_command(label=item.replace('_', ' '), command=self.notch_reject_image, state=DISABLED if not self.is_fft_image else NORMAL)
                    else:
                        method_name = item.lower().replace(' ', '_') + '_image'
                        if hasattr(self, method_name):
                            menu.add_command(label=item.replace('_', ' '), command=getattr(self, method_name))
                        else:
                            print(f"Warning: No method found for {item}")
            elif isinstance(items, dict):
                parent_menu = Menu(menu_bar, tearoff=False)
                menu_bar.add_cascade(label=menu_name.replace('_', ' '), menu=parent_menu)
                self.create_menus(parent_menu, items)

    def open_image(self):
            file_path = filedialog.askopenfilename()
            if file_path:
                try:
                    np_image = self.image_processor.load_image(file_path)
                    self.root.after(0, self.display_image, np_image)
                except Exception as e:
                    self.handle_error("Failed to open the image", e)

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
            self.handle_error("Failed to blackout the image", e)
    
    def gaussian_image(self):
        try:
            np_image, hist = self.image_processor.apply_gaussian_noise(mean=0, std=100, fixed_size=(640,640))
            self.root.after(0, self.show_noise_histogram, hist)
            self.root.after(0, self.display_image, np_image)
        except Exception as e:
            self.handle_error("Failed to apply/generate noise", e)
    
    def salt_and_pepper_image(self):
        try:
            np_image, hist = self.image_processor.apply_salt_and_pepper_noise(salt_prob=0.05, pepper_prob=0.05, fixed_size=(640,640))
            self.root.after(0, self.show_noise_histogram, hist)
            self.root.after(0, self.display_image, np_image)
        except Exception as e:
            self.handle_error("Failed to apply/generate noise", e) 

    def show_noise_histogram(self, hist):
        popup = Toplevel(self.root)
        popup.title("Noise histogram")
        popup.geometry("800x600")

        # Plot histograms
        fig = Figure(figsize=(6, 4), dpi=100)
        ax1 = fig.add_subplot(121)

        ax1.bar(range(len(hist)), hist.ravel(), color='gray')
        ax1.set_title('Noise Histogram')
        ax1.set_xlim([0, 256])

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)

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
            self.handle_error("Failed to transform the image", e)
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
            self.handle_error("Failed to transform the image", e)
            popup.destroy()

    def bit_plane_extract_image(self):
        self.create_bit_plane_popup()

    def create_bit_plane_popup(self):
        popup = Toplevel(self.root)
        popup.title("Bit plane extraction")
        popup.geometry("200x100")

        Label(popup, text="Desired plane:").pack(side="top", fill="x", pady=10)

        bit_plane_entry = Entry(popup)
        bit_plane_entry.pack(side="top", fill="x", padx=60)

        apply_button = Button(popup, text="Apply", command=lambda: self.extract_bit_plane_and_close_popup(bit_plane_entry.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def extract_bit_plane_and_close_popup(self, bit_plane, popup):
        try:
            bit_plane = int(bit_plane)
            np_image = self.image_processor.extract_bit_plane(bit_plane)
            self.root.after(0, self.display_image, np_image)
            popup.destroy()
        except ValueError:
            messagebox.showerror("Bit plane extraction", "Invalid plane value. Please enter a valid number.")
        except Exception as e:
            self.handle_error("Failed to extract bit plane from the image", e)
            popup.destroy()

    def equalize_histogram_image(self):
        try:
            equalized_image, original_hist, equalized_hist = self.image_processor.equalize_histogram()
            self.root.after(0, self.show_histogram_results, original_hist, equalized_hist)
            np_image = equalized_image
            self.root.after(0, self.display_image, np_image)
        except Exception as e:
            self.handle_error("Failed to equalize the image", e)

    def show_histogram_results(self, original_hist, equalized_hist):
        popup = Toplevel(self.root)
        popup.title("Histogram Equalization Results")
        popup.geometry("800x600")

        # Plot histograms
        fig = Figure(figsize=(6, 4), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.bar(range(len(original_hist)), original_hist.ravel(), color='gray')
        ax1.set_title('Original Histogram')
        ax1.set_xlim([0, 256])

        ax2.bar(range(len(equalized_hist)), equalized_hist.ravel(), color='gray')
        ax2.set_title('Equalized Histogram')
        ax2.set_xlim([0, 256])

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)
    
    def restore_image(self):
        try:
            np_image = self.image_processor.restore_image()
            self.root.after(0, self.display_image, np_image)
        except Exception as e:
            self.handle_error("Failed to restore the image", e)

    def intensity_slice_image(self):
        self.create_intensity_slicing_popup()

    def create_intensity_slicing_popup(self):
        popup = Toplevel(self.root)
        popup.title("Intensity Slicing")
        popup.geometry("300x200")

        Label(popup, text="Number of Ranges:").pack(side="top", fill="x", pady=10)

        num_ranges_entry = Entry(popup)
        num_ranges_entry.pack(side="top", fill="x", padx=50)

        Label(popup, text="Color Map:").pack(side="top", fill="x", pady=5)

        # Combo box for color map selection
        cmap_var = StringVar()
        cmap_combo = Combobox(popup, textvariable=cmap_var, state='readonly')
        cmap_combo['values'] = ('viridis', 'plasma', 'inferno', 'magma', 'cividis', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter', 'jet', 'turbo')
        cmap_combo.current(0)  # set the selected item
        cmap_combo.pack(side="top", fill="x", padx=50)

        apply_button = Button(popup, text="Apply", command=lambda: self.apply_intensity_slicing_and_close_popup(num_ranges_entry.get(), cmap_var.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def apply_intensity_slicing_and_close_popup(self, num_ranges, cmap_name, popup):
        try:
            num_ranges = int(num_ranges)
            if num_ranges < 1 or num_ranges > 255:
                raise ValueError("Number of ranges must be between 1 and 255.")

            np_image = self.image_processor.intensity_slicing_pseudocolor(num_ranges, cmap_name)
            self.root.after(0, self.display_image, np_image)  # Assumes a method to display images on the main GUI
            popup.destroy()
        except ValueError as e:
            messagebox.showerror("Intensity Slicing", f"Invalid input: {str(e)}")
        except Exception as e:
            self.handle_error("Failed to apply intensity slicing to the image", e)
            popup.destroy()

    def average_filter_image(self):
        self.create_average_filter_popup()

    def create_average_filter_popup(self):
        popup = Toplevel(self.root)
        popup.title("Average Filter")
        popup.geometry("200x100")

        Label(popup, text="Kernel Size:").pack(side="top", fill="x", pady=10)

        kernel_size_entry = Entry(popup)
        kernel_size_entry.pack(side="top", fill="x", padx=60)

        apply_button = Button(popup, text="Apply", command=lambda: self.apply_average_filter_and_close_popup(kernel_size_entry.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def apply_average_filter_and_close_popup(self, kernel_size, popup):
        try:
            kernel_size = int(kernel_size)
            np_image = self.image_processor.apply_average_filter(kernel_size)
            self.root.after(0, self.display_image, np_image)
            popup.destroy()
        except ValueError:
            messagebox.showerror("Average Filter", "Invalid kernel size value. Please enter a valid number.")
        except Exception as e:
            self.handle_error("Failed to filter the image", e)
            popup.destroy()

    def min_filter_image(self):
        self.create_min_filter_popup()

    def create_min_filter_popup(self):
        popup = Toplevel(self.root)
        popup.title("Min Filter")
        popup.geometry("200x100")

        Label(popup, text="Kernel Size:").pack(side="top", fill="x", pady=10)

        kernel_size_entry = Entry(popup)
        kernel_size_entry.pack(side="top", fill="x", padx=60)

        apply_button = Button(popup, text="Apply", command=lambda: self.apply_min_filter_and_close_popup(kernel_size_entry.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def apply_min_filter_and_close_popup(self, kernel_size, popup):
        try:
            kernel_size = int(kernel_size)
            np_image = self.image_processor.apply_min_filter(kernel_size)
            self.root.after(0, self.display_image, np_image)
            popup.destroy()
        except ValueError:
            messagebox.showerror("Min Filter", "Invalid kernel size value. Please enter a valid number.")
        except Exception as e:
            self.handle_error("Failed to filter the image", e)
            popup.destroy()

    def max_filter_image(self):
        self.create_max_filter_popup()

    def create_max_filter_popup(self):
        popup = Toplevel(self.root)
        popup.title("Max Filter")
        popup.geometry("200x100")

        Label(popup, text="Kernel Size:").pack(side="top", fill="x", pady=10)

        kernel_size_entry = Entry(popup)
        kernel_size_entry.pack(side="top", fill="x", padx=60)

        apply_button = Button(popup, text="Apply", command=lambda: self.apply_max_filter_and_close_popup(kernel_size_entry.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def apply_max_filter_and_close_popup(self, kernel_size, popup):
        try:
            kernel_size = int(kernel_size)
            np_image = self.image_processor.apply_max_filter(kernel_size)
            self.root.after(0, self.display_image, np_image)
            popup.destroy()
        except ValueError:
            messagebox.showerror("Max Filter", "Invalid kernel size value. Please enter a valid number.")
        except Exception as e:
            self.handle_error("Failed to filter the image", e)
            popup.destroy()

    def median_filter_image(self):
        self.create_median_filter_popup()

    def create_median_filter_popup(self):
        popup = Toplevel(self.root)
        popup.title("Median Filter")
        popup.geometry("200x100")

        Label(popup, text="Kernel Size:").pack(side="top", fill="x", pady=10)

        kernel_size_entry = Entry(popup)
        kernel_size_entry.pack(side="top", fill="x", padx=60)

        apply_button = Button(popup, text="Apply", command=lambda: self.apply_median_filter_and_close_popup(kernel_size_entry.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def apply_median_filter_and_close_popup(self, kernel_size, popup):
        try:
            kernel_size = int(kernel_size)
            np_image = self.image_processor.apply_median_filter(kernel_size)
            self.root.after(0, self.display_image, np_image)
            popup.destroy()
        except ValueError:
            messagebox.showerror("Median Filter", "Invalid kernel size value. Please enter a valid number.")
        except Exception as e:
            self.handle_error("Failed to filter the image", e)
            popup.destroy()

    def laplacian_filter_image(self):
        try:
            sharpened_image, raw_laplacian, adjusted_laplacian = self.image_processor.apply_laplacian_filter()
            self.root.after(0, self.show_laplacian_results, raw_laplacian, adjusted_laplacian)
            np_image = sharpened_image
            self.root.after(0, self.display_image, np_image)
        except Exception as e:
            self.handle_error("Failed to filter the image", e)

    def show_laplacian_results(self, raw_laplacian, adjusted_laplacian):
        popup = Toplevel(self.root)
        popup.title("Laplacian Filter Results")
        popup.geometry("800x600")

        # Plot results
        fig = Figure(figsize=(6, 4), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        # Displaying the raw Laplacian
        ax1.imshow(raw_laplacian, cmap='gray')
        ax1.set_title('Raw Laplacian')
        ax1.axis('off')  # Turn off axis numbers and ticks

        # Displaying the adjusted Laplacian
        ax2.imshow(adjusted_laplacian, cmap='gray')
        ax2.set_title('Adjusted Laplacian')
        ax2.axis('off')

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(side="bottom", fill="both", expand=True)

    def ft_image(self):
        try:
            magnitude_spectrum, phase_angle, f = self.image_processor.compute_fft_spectrum_and_phase()
            self.root.after(0, self.show_fft_results, magnitude_spectrum, phase_angle)
            np_image = magnitude_spectrum
            self.root.after(0, self.display_image, np_image)
            self.is_fft_image = True
            self.fft = f
            self.fft_phase_angle = phase_angle
        except Exception as e:
            self.handle_error("Failed to transform", e)

    def ift_image(self, magnitude_spectrum, phase_angle):
        try:
            # Compute the inverse Fourier Transform
            ift_image = self.image_processor.compute_inverse_fft(magnitude_spectrum, phase_angle)
            np_image = ift_image
            self.root.after(0, self.display_image, np_image)
        except Exception as e:
            self.handle_error("Failed to compute IFT", e)

    def show_fft_results(self, magnitude_spectrum, phase_angle):
        popup = Toplevel(self.root)
        popup.title("Fourier Transform Results")
        popup.geometry("800x600")

        # Plot results
        fig = Figure(figsize=(6, 4), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        ax1.imshow(magnitude_spectrum, cmap='gray')
        ax1.set_title('Fourier spectrum')
        ax1.axis('off')

        ax2.imshow(phase_angle, cmap='gray')
        ax2.set_title('Phase angle')
        ax2.axis('off')

        canvas = FigureCanvasTkAgg(fig, master=popup)
        canvas.draw()
        canvas.get_tk_widget().pack(side="top", fill="both", expand=True)

        # Add a button to compute the inverse transform
        btn_ift = Button(popup, text="Compute IFT", command=lambda: self.ift_image(magnitude_spectrum, phase_angle))
        btn_ift.pack(side="bottom", pady=20)

    def high_pass_image(self):
        self.create_high_pass_popup()

    def create_high_pass_popup(self):
        popup = Toplevel(self.root)
        popup.title("High pass filter")
        popup.geometry("200x100")

        Label(popup, text="radius size:").pack(side="top", fill="x", pady=10)

        radius_size_entry = Entry(popup)
        radius_size_entry.pack(side="top", fill="x", padx=60)

        apply_button = Button(popup, text="Apply", command=lambda: self.apply_high_pass_and_close_popup(radius_size_entry.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def apply_high_pass_and_close_popup(self, radius_size, popup):
        try:
            radius_size = float(radius_size)
            np_image = self.image_processor.apply_high_pass(radius_size)
            self.root.after(0, self.display_image, np_image)
            popup.destroy()
        except ValueError:
            messagebox.showerror("High pass filter", "Invalid radius value. Please enter a valid number.")
        except Exception as e:
            self.handle_error("Failed to filter the image", e)
            popup.destroy()

    def low_pass_image(self):
        self.create_low_pass_popup()

    def create_low_pass_popup(self):
        popup = Toplevel(self.root)
        popup.title("Low pass filter")
        popup.geometry("200x100")

        Label(popup, text="radius size:").pack(side="top", fill="x", pady=10)

        radius_size_entry = Entry(popup)
        radius_size_entry.pack(side="top", fill="x", padx=60)

        apply_button = Button(popup, text="Apply", command=lambda: self.apply_low_pass_and_close_popup(radius_size_entry.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def apply_low_pass_and_close_popup(self, radius_size, popup):
        try:
            radius_size = float(radius_size)
            np_image = self.image_processor.apply_low_pass(radius_size)
            self.root.after(0, self.display_image, np_image)
            popup.destroy()
        except ValueError:
            messagebox.showerror("Low pass filter", "Invalid radius value. Please enter a valid number.")
        except Exception as e:
            self.handle_error("Failed to filter the image", e)
            popup.destroy()

    def notch_reject_image(self):
        if not self.is_fft_image:
            messagebox.showinfo("Notch Reject Filter", "This operation can only be applied to an FFT image.")
            return
        self.notch_reject_active = True
        messagebox.showinfo("Notch Reject Filter", "Click on the desired points in the image and press Enter to apply the filter.")
        self.notch_points = []

    def on_image_click(self, event):
        if self.notch_reject_active:
            x, y = event.x, event.y
            self.show_radius_popup(x, y)

    def show_radius_popup(self, x, y):
        popup = Toplevel(self.root)
        popup.title("Notch Reject Radius")
        popup.geometry("200x100")

        Label(popup, text="Radius size:").pack(side="top", fill="x", pady=10)
        radius_entry = Entry(popup)
        radius_entry.pack(side="top", fill="x", padx=60)
        apply_button = Button(popup, text="Apply", command=lambda: self.add_notch_point(x, y, radius_entry.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def add_notch_point(self, x, y, radius, popup):
        try:
            radius = float(radius)
            self.notch_points.append((x, y, radius))
            self.update_display_with_notch_points()
            popup.destroy()
        except ValueError:
            messagebox.showerror("Notch Reject", "Invalid radius value. Please enter a valid number.")

    def update_display_with_notch_points(self):
        # Ensure we start from the original image
        np_image = self.image_processor.current_image.copy()
        for (x, y, radius) in self.notch_points:
            cv2.circle(np_image, (x, y), int(radius), (0, 255, 0), -1)  # Fill the circle with -1 thickness
        self.display_image(np_image)

    def on_enter_pressed(self, event):
        if self.notch_reject_active and self.notch_points:
            try:
                # Apply the notch reject filter to the FFT image
                filtered_fft_image = self.image_processor.apply_notch_reject( self.fft, self.notch_points)
                # Convert the filtered FFT image back to the spatial domain
                #spatial_image = self.image_processor.inverse_fft(filtered_fft_image, self.fft_phase_angle)
                self.display_image(filtered_fft_image)
                self.notch_points = []
                self.notch_reject_active = False
                self.is_fft_image = False
            except Exception as e:
                self.handle_error("Failed to apply notch reject filter", e)

    def geometric_mean_image(self):
        self.create_geometric_mean_filter_popup()

    def create_geometric_mean_filter_popup(self):
        popup = Toplevel(self.root)
        popup.title("Geometric Mean Filter")
        popup.geometry("200x100")

        Label(popup, text="Kernel Size:").pack(side="top", fill="x", pady=10)

        kernel_size_entry = Entry(popup)
        kernel_size_entry.pack(side="top", fill="x", padx=60)

        apply_button = Button(popup, text="Apply", command=lambda: self.apply_geometric_mean_filter_and_close_popup(kernel_size_entry.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def apply_geometric_mean_filter_and_close_popup(self, kernel_size, popup):
        try:
            kernel_size = int(kernel_size)
            np_image = self.image_processor.apply_geometric_mean_filter(kernel_size)
            self.root.after(0, self.display_image, np_image)
            popup.destroy()
        except ValueError:
            messagebox.showerror("Geometric Mean Filter", "Invalid kernel size value. Please enter a valid number.")
        except Exception as e:
            self.handle_error("Failed to filter the image", e)
            popup.destroy()

    def alpha_trimmed_mean_image(self):
        self.create_alpha_trimmed_mean_filter_popup()

    def create_alpha_trimmed_mean_filter_popup(self):
        popup = Toplevel(self.root)
        popup.title("Alpha Trimmed Mean Filter")
        popup.geometry("200x200")

        Label(popup, text="Kernel Size:").pack(side="top", fill="x", pady=10)

        kernel_size_entry = Entry(popup)
        kernel_size_entry.pack(side="top", fill="x", padx=60)

        Label(popup, text="d:").pack(side="top", fill="x", pady=10)

        d_value_entry = Entry(popup)
        d_value_entry.pack(side="top", fill="x", padx=60)

        apply_button = Button(popup, text="Apply", command=lambda: self.apply_alpha_trimmed_mean_and_close_popup(kernel_size_entry.get(), d_value_entry.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def apply_alpha_trimmed_mean_and_close_popup(self, kernel_size, d_value,popup):
        try:
            kernel_size = int(kernel_size)
            np_image = self.image_processor.apply_alpha_trimmed_mean_filter(kernel_size, int(d_value))
            self.root.after(0, self.display_image, np_image)
            popup.destroy()
        except ValueError:
            messagebox.showerror("Alpha Trimmed Mean Filter", "Invalid kernel size value. Please enter a valid number.")
        except Exception as e:
            self.handle_error("Failed to filter the image", e)
            popup.destroy()

    def erosion_image(self):
        self.create_erosion_popup()

    def create_erosion_popup(self):
        popup = Toplevel(self.root)
        popup.title("Erosion")
        popup.geometry("200x300")

        Label(popup, text="Kernel Size:").pack(side="top", fill="x", pady=10)

        kernel_size_entry = Entry(popup)
        kernel_size_entry.pack(side="top", fill="x", padx=60)

        Label(popup, text="Iterations:").pack(side="top", fill="x", pady=10)

        iterations_entry = Entry(popup)
        iterations_entry.pack(side="top", fill="x", padx=60)
        
        Label(popup, text="Kernel Type:").pack(side="top", fill="x", pady=10)

        kernel_type_var = StringVar(popup)
        kernel_type_var.set("rect")  # default value

        kernel_type_options = ["rect", "ellipse", "cross"]
        kernel_type_menu = OptionMenu(popup, kernel_type_var, *kernel_type_options)
        kernel_type_menu.pack(side="top", fill="x", padx=60)

        apply_button = Button(popup, text="Apply", command=lambda: self.apply_erosion_and_close_popup(kernel_size_entry.get(), iterations_entry.get(), kernel_type_var.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def apply_erosion_and_close_popup(self, kernel_size, iterations, kernel_type, popup):
        try:
            kernel_size = int(kernel_size)
            iterations = int(iterations)
            kernel_type = str(kernel_type)
            np_image = self.image_processor.apply_erosion(kernel_size, iterations, kernel_type)
            self.root.after(0, self.display_image, np_image)
            popup.destroy()
        except ValueError:
            messagebox.showerror("Erosion", "Invalid values. Please enter a valid number.")
        except Exception as e:
            self.handle_error("Failed to transform the image", e)
            popup.destroy()

    def dilation_image(self):
        self.create_dilation_popup()

    def create_dilation_popup(self):
        popup = Toplevel(self.root)
        popup.title("Dilation")
        popup.geometry("200x300")

        Label(popup, text="Kernel Size:").pack(side="top", fill="x", pady=10)

        kernel_size_entry = Entry(popup)
        kernel_size_entry.pack(side="top", fill="x", padx=60)

        Label(popup, text="Iterations:").pack(side="top", fill="x", pady=10)

        iterations_entry = Entry(popup)
        iterations_entry.pack(side="top", fill="x", padx=60)
        
        Label(popup, text="Kernel Type:").pack(side="top", fill="x", pady=10)

        kernel_type_var = StringVar(popup)
        kernel_type_var.set("rect")  # default value

        kernel_type_options = ["rect", "ellipse", "cross"]
        kernel_type_menu = OptionMenu(popup, kernel_type_var, *kernel_type_options)
        kernel_type_menu.pack(side="top", fill="x", padx=60)

        apply_button = Button(popup, text="Apply", command=lambda: self.apply_dilation_and_close_popup(kernel_size_entry.get(), iterations_entry.get(), kernel_type_var.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def apply_dilation_and_close_popup(self, kernel_size, iterations, kernel_type, popup):
        try:
            kernel_size = int(kernel_size)
            iterations = int(iterations)
            kernel_type = str(kernel_type)
            np_image = self.image_processor.apply_dilation(kernel_size, iterations, kernel_type)
            self.root.after(0, self.display_image, np_image)
            popup.destroy()
        except ValueError:
            messagebox.showerror("Dilation", "Invalid values. Please enter a valid number.")
        except Exception as e:
            self.handle_error("Failed to transform the image", e)
            popup.destroy()

    def opening_image(self):
        self.create_opening_popup()

    def create_opening_popup(self):
        popup = Toplevel(self.root)
        popup.title("Opening")
        popup.geometry("200x300")

        Label(popup, text="Kernel Size:").pack(side="top", fill="x", pady=10)

        kernel_size_entry = Entry(popup)
        kernel_size_entry.pack(side="top", fill="x", padx=60)

        Label(popup, text="Iterations:").pack(side="top", fill="x", pady=10)

        iterations_entry = Entry(popup)
        iterations_entry.pack(side="top", fill="x", padx=60)
        
        Label(popup, text="Kernel Type:").pack(side="top", fill="x", pady=10)

        kernel_type_var = StringVar(popup)
        kernel_type_var.set("rect")  # default value

        kernel_type_options = ["rect", "ellipse", "cross"]
        kernel_type_menu = OptionMenu(popup, kernel_type_var, *kernel_type_options)
        kernel_type_menu.pack(side="top", fill="x", padx=60)

        apply_button = Button(popup, text="Apply", command=lambda: self.apply_opening_and_close_popup(kernel_size_entry.get(), iterations_entry.get(), kernel_type_var.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def apply_opening_and_close_popup(self, kernel_size, iterations, kernel_type, popup):
        try:
            kernel_size = int(kernel_size)
            iterations = int(iterations)
            kernel_type = str(kernel_type)
            np_image = self.image_processor.apply_opening(kernel_size, iterations, kernel_type)
            self.root.after(0, self.display_image, np_image)
            popup.destroy()
        except ValueError:
            messagebox.showerror("Opening", "Invalid values. Please enter a valid number.")
        except Exception as e:
            self.handle_error("Failed to transform the image", e)
            popup.destroy()

    def closing_image(self):
        self.create_opening_popup()

    def create_closing_popup(self):
        popup = Toplevel(self.root)
        popup.title("Closing")
        popup.geometry("200x300")

        Label(popup, text="Kernel Size:").pack(side="top", fill="x", pady=10)

        kernel_size_entry = Entry(popup)
        kernel_size_entry.pack(side="top", fill="x", padx=60)

        Label(popup, text="Iterations:").pack(side="top", fill="x", pady=10)

        iterations_entry = Entry(popup)
        iterations_entry.pack(side="top", fill="x", padx=60)
        
        Label(popup, text="Kernel Type:").pack(side="top", fill="x", pady=10)

        kernel_type_var = StringVar(popup)
        kernel_type_var.set("rect")  # default value

        kernel_type_options = ["rect", "ellipse", "cross"]
        kernel_type_menu = OptionMenu(popup, kernel_type_var, *kernel_type_options)
        kernel_type_menu.pack(side="top", fill="x", padx=60)

        apply_button = Button(popup, text="Apply", command=lambda: self.apply_closing_and_close_popup(kernel_size_entry.get(), iterations_entry.get(), kernel_type_var.get(), popup))
        apply_button.pack(side="bottom", pady=10)

    def apply_closing_and_close_popup(self, kernel_size, iterations, kernel_type, popup):
        try:
            kernel_size = int(kernel_size)
            iterations = int(iterations)
            kernel_type = str(kernel_type)
            np_image = self.image_processor.apply_closing(kernel_size, iterations, kernel_type)
            self.root.after(0, self.display_image, np_image)
            popup.destroy()
        except ValueError:
            messagebox.showerror("Closing", "Invalid values. Please enter a valid number.")
        except Exception as e:
            self.handle_error("Failed to transform the image", e)
            popup.destroy()

    def huffman_code_image(self):
        try:
            results = self.image_processor.apply_huffman_coding()
            self.show_huffman_results(results)
        except Exception as e:
            self.handle_error("Failed to process the image for Huffman coding", e)
    
    def show_huffman_results(self, results):
        popup = Toplevel(self.root)
        popup.title("Huffman Coding Results")
        popup.geometry("800x600")

        huffman_codes = results['Huffman Codes']
        entropy = results['Entropy']
        compression_ratio = results['Compression Ratio']
        relative_redundancy = results['Relative Redundancy']

        results_text = f"Entropy: {entropy:.4f}\nCompression Ratio: {compression_ratio:.4f}\nRelative Redundancy: {relative_redundancy:.4f}\n\nHuffman Codes:\n"
        results_text += "\n".join([f"Intensity: {intensity}, Probability: {probability:.4f}, Code: {code}" for intensity, probability, code in huffman_codes])

        text_widget = Text(popup, wrap="word")
        text_widget.insert("1.0", results_text)
        text_widget.config(state="disabled")
        text_widget.pack(fill="both", expand=True)

    def handle_error(self, message, error):
        messagebox.showerror("Error", f"{message}\n\n{error}")

    def display_image(self, np_image):
        # Convert the NumPy image to a PIL image
        if np_image.dtype != np.uint8:
            np_image = ((np_image - np_image.min()) / (np_image.max() - np_image.min()) * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(cv2.cvtColor(np_image.astype(np.uint8), cv2.COLOR_BGR2RGB))

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

