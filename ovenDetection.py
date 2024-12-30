import tkinter as tk                                             # Library for creating GUI applications
from tkinter import filedialog                                   # Module for file dialog window
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # Matplotlib's backend for embedding plots in tkinter
import matplotlib.pyplot as plt                                  # Plotting library
import cv2                                                       # OpenCV library for computer vision tasks
import numpy as np                                               # Numerical computing library

class OvenDetection(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.title("Oven Glove Defect Detection")
        self.geometry("1000x800")
        
        # Load Image Button
        load_button = tk.Button(self, text="Load New Image", command=self.load_image)
        load_button.pack()
        
        # Create figure and subplots
        self.fig = plt.Figure(figsize=(10, 6))
        self.ax1 = self.fig.add_subplot(231)   # Subplot for original image
        self.ax2 = self.fig.add_subplot(232)   # Subplot for scratch defect  
        self.ax3 = self.fig.add_subplot(233)   # Subplot for burn defect  
        self.ax4 = self.fig.add_subplot(234)   # Subplot for hole defect 
        self.ax5 = self.fig.add_subplot(235)   # Subplot for stain defect
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def load_image(self):
        filename = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.bmp")])
        if filename:
            image = cv2.imread(filename)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.ax1.imshow(image_rgb)
            self.ax1.set_title("Original Image")

            # Create copies of the original image to be used for each defect detection
            img1 = image_rgb.copy()
            img2 = image.copy()
            img3 = image_rgb.copy()
            img4 = image.copy()
            
            # Implement the detection algorithms on each image
            scratchImg = self.scratch_detect(img1)
            burnImg = cv2.cvtColor(self.burn_detect(img2), cv2.COLOR_BGR2RGB)
            holeImg = self.hole_detect(img3)
            stainImg = cv2.cvtColor(self.stain_detect(img4), cv2.COLOR_BGR2RGB)

            # Clear the plots before displaying the images
            self.ax2.clear()
            self.ax3.clear()
            self.ax4.clear()
            self.ax5.clear()

            # Display images on each plot
            self.ax2.imshow(scratchImg)
            self.ax3.imshow(burnImg)
            self.ax4.imshow(holeImg)
            self.ax5.imshow(stainImg)

            # Set title for each plot
            self.ax2.set_title("Scratch Detection")
            self.ax3.set_title("Burn Detection")
            self.ax4.set_title("Hole Detection")
            self.ax5.set_title("Stain Detection")

            self.canvas.draw()


    def scratch_detect(self, image):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Perform edge detection using Canny
        edges = cv2.Canny(blurred_image, 30, 150)

        # Find contours in the edge-detected image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through each contour and calculate its area
        for contour in contours:
            area = cv2.contourArea(contour)
            if 25 < area < 75:             # Filter contours based on contour area to detect scratches
                # Get the bounding box of the contour
                x, y, w, h = cv2.boundingRect(contour)
                # Draw the bounding box
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Label the contour as 'scratch'
                cv2.putText(image, 'scratch', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image

    def hole_detect(self, image):
        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to reduce noise
        blur = cv2.GaussianBlur(gray_image, (5, 5), 0)

        # Perform non-local means denoising to further reduce noise
        dst = cv2.fastNlMeansDenoising(blur, None, 10, 7, 21)

        # Perform edge detection using Canny
        edges = cv2.Canny(dst, 30, 150)

        # Threshold the edge-detected image to get a binary image
        _, binary_image = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Define a kernel for dilation
        kernel = np.ones((5, 5), np.uint8)

        # Dilate the binary image to enhance the holes
        dilation = cv2.dilate(binary_image, kernel, iterations=1)   

        # Find contours in the dilated image
        contours, _ = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # Iterate through contours and draw rectangles around detected holes
        for contour in contours:
            # Find the bounding rectangle for each contour
            x, y, w, h = cv2.boundingRect(contour)
            if 60 <= w <= 75 and 60 <= h <= 67:   # Filter contours based on size to detect holes
                # Draw a rectangle around the detected hole on the original image
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Add text indicating the detected object as 'hole'
                cv2.putText(image, "hole", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image
    
    def burn_detect(self,image):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define lower and upper bounds for burnt color
        lower_burnt = np.array([0, 50, 50])
        upper_burnt = np.array([15, 255, 255])
            
        # Create a mask using the defined bounds
        mask = cv2.inRange(hsv_image, lower_burnt, upper_burnt)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes around detected burnt marks with specified sizes
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if 1000 <= area <= 20000:  # Specify the desired area range for the rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                # Add the word 'burn' with area on the rectangle
                cv2.putText(image, 'burn', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return image
    
    def stain_detect(self,image):
        # Convert the image to HSV color space
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define lower and upper bounds for burnt color
        lower_burnt = np.array([0, 50, 50])
        upper_burnt = np.array([30, 255, 255])
        
        # Create a mask using the defined bounds
        mask = cv2.inRange(hsv_image, lower_burnt, upper_burnt)
        
        # Apply morphological operations to clean up the mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw bounding boxes around detected dirty stains with specified sizes
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if area > 200:  # Specify the desired area range for the rectangle
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
               
                cv2.putText(image, 'stain', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return image   
