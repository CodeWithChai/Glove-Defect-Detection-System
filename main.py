import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import ImageTk,Image

def detect_mould(img_path):
    image_original = cv2.imread(img_path)
    # we make a copy of this picture but this copy is all black, we'll draw annotations on it later 
    overlay = np.zeros_like(image_original)
    
    # we change the picture to a different color setup that helps us see colors better
    img_hsv = cv2.cvtColor(image_original, cv2.COLOR_BGR2HSV)
    # colors that are close to the color of typical mould 
    lower_hue = np.array([20, 50, 20])
    upper_hue = np.array([80, 255, 255])

    # making a mask, that only lets through the colors we care about
    mask = cv2.inRange(img_hsv, lower_hue, upper_hue)
    
    # cleaning up the mask a bit, closing gaps and opening clumps, making it clearer 
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # we look for shapes in our cleaned-up mask
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No mould detected")
        return overlay
    
    for contour in contours:
        # we check of the mould detected to see how big it is
        area = cv2.contourArea(contour)
        if 200 < area < 3000:  # we're only interested in shapes that are just the right size
            # for each shape that's the right size, we draw a green box around it on our black copy of the picture
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(overlay, "Mould Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # we mix our original picture with our black copy that now has green boxes and writing on it
    annotated_image = cv2.addWeighted(image_original, 1, overlay, 0.4, 0)
    
    return annotated_image


def detect_punctures(img_path):
    image_original = cv2.imread(img_path)
    # creating an overlay with the same dimensions and type as the original image but all zeros
    # this will be used to draw the annotations
    overlay = np.zeros_like(image_original)

    # converting the image to grayscale for processing
    img = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
    # adjust the brightness and contrast
    # this is to account for low quality images in our dataset
    img = cv2.convertScaleAbs(img, alpha=1.5, beta=6)

    # thresholding to get a binary image
    value_lower = 25
    value_higher = 255
    thresh = cv2.inRange(img, value_lower, value_higher)

    # removing small noises and filling holes
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

    # increase the size of the holes using dilation
    dilation = cv2.dilate(closing, kernel, iterations=5)

    # find contours in the image
    # this will be used to draw the bounding boxes
    contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 < area < 3050:
            # find and draw the bounding box
            # this will be used to highlight the detected punctures
            bbox = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)

            # Place the text
            message = "Puncture Detected here"
            text_size, _ = cv2.getTextSize(message, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_width, text_height = text_size


            cv2.putText(
                overlay,
                message,
                (int(bbox[0] + bbox[2] / 2 - text_width / 2),
                 (bbox[1] + bbox[3] + text_height + 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

    # add the overlay to the original image 
    # this will be used to display the annotated image
    annotated_image = cv2.add(image_original, overlay)
    return annotated_image

def detect_paint(image_path):
    image_original = cv2.imread(image_path)
    blurred_image = cv2.GaussianBlur(image_original, (7, 7), 0)


    hsv_image = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

    # paint masks
    # lower_red = np.array([170, 70, 50])
    # upper_red = np.array([hue - the actual color, how rich (when mixed with white), how bright])
    lower_red = np.array([0, 120, 180])
    upper_red = np.array([10, 255, 255])

    # Yellow color range
    lower_yellow = np.array([20, 150, 200])
    upper_yellow = np.array([30, 255, 255])

    # Green color range
    lower_green = np.array([40, 50, 100])
    upper_green = np.array([80, 255, 255])

    # Create masks
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    # Combine masks for all colors into one mask for the new image
    new_mask_combined = cv2.bitwise_or(mask_red, mask_yellow)
    new_mask_combined = cv2.bitwise_or(new_mask_combined, mask_green)

    # morphological operations to remove noise and fill holes
    kernel = np.ones((5, 5), np.uint8)
    new_mask_combined = cv2.morphologyEx(new_mask_combined, cv2.MORPH_OPEN, kernel)
    new_mask_combined = cv2.morphologyEx(new_mask_combined, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(new_mask_combined, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Draw bounding boxes on the original image
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image_original, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return image_original

class LeatherGloveDetection(tk.Toplevel):
    def __init__(self):
        super().__init__()
        self.title("Leather Glove Defect Detection")
        self.geometry("1000x600") 
        
        self.selected_image_path = None 
        
        # Elements
        self.create_widgets()

    def create_widgets(self):
        print('current working ---> \n\n',os.getcwd())
        self.select_button = tk.Button(self, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)
        
        self.process_button = tk.Button(self, text="Detect Paint", command=self.get_detect_paint)
        self.process_button.pack(pady=10)

        self.process_button = tk.Button(self, text="Detect Mould", command=self.get_detect_mould)
        self.process_button.pack(pady=10)

        self.process_button = tk.Button(self, text="Detect Puncture", command=self.get_detect_puncture)
        self.process_button.pack(pady=10)
        
        self.quit_button = tk.Button(self, text="Quit", command=self.destroy)
        self.quit_button.pack(pady=10)

        # Frame for displaying the selected image
        self.image_frame = tk.Frame(self, width=500, height=500)
        self.image_frame.pack(pady=20)
        self.image_frame.pack_propagate(False)  # Prevents the frame from resizing to fit the image

        self.image_label = tk.Label(self.image_frame, text="No Image Selected")
        self.image_label.pack(expand=True)

    def select_image(self):
        file_path = filedialog.askopenfilename(initialdir='defect-detection-opencv-python/images', title="Select an Image",
                                               filetypes=(("jpeg files", "*.jpg"), ("png files", "*.png")))
        if file_path:
            self.selected_image_path = file_path
            self.display_image(file_path)

    def display_image(self, img_path):
        img = Image.open(img_path)
        img = ImageTk.PhotoImage(img)
        self.image_label.configure(image=img, text="")
        self.image_label.image = img  

    def get_detect_paint(self):
        if self.selected_image_path:
            processed_img = detect_paint(self.selected_image_path)
            cv2.imshow("Processed Image", processed_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            messagebox.showerror("Error", "Please select an image first")

    def get_detect_mould(self):
        if self.selected_image_path:
            processed_img = detect_mould(self.selected_image_path)
            cv2.imshow("Processed Image", processed_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            messagebox.showerror("Error", "Please select an image first")


    def get_detect_puncture(self):
        if self.selected_image_path:
            processed_img = detect_punctures(self.selected_image_path)
            cv2.imshow("Processed Image", processed_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            messagebox.showerror("Error", "Please select an image first")

# # Create and run the application
# app = LeatherGloveDetection()
# app.mainloop()

