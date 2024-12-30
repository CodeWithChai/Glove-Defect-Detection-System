import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk  # Import Image class from PIL module
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tkinter import ttk


class SurgicalDetectionPage(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Surgical Glove Detection Page")
        self.geometry("600x400")  # Adjusted height to accommodate the back button

        # Title header 1
        title_label = tk.Label(self, text="This is Surgical Glove Detection Page", font=('Times New Roman', 16, 'bold', 'underline'))
        title_label.pack(pady=10)

        # Header 2
        subtitle_label = tk.Label(self, text="Please import your surgical glove image to detect the defect", font=('Times New Roman', 14))
        subtitle_label.pack(pady=10)

        # Button to import images
        import_button = ttk.Button(self, text="Import Image", command=self.import_image, style='Custom.TButton')
        import_button.pack(pady=10)

        # Back button
        back_button = ttk.Button(self, text="Back", command=self.back_to_main, style='Custom.TButton')
        back_button.pack(pady=10)

        # Label to display imported image
        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)

    def import_image(self):
        # Open file dialog to select image
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

        # Check if a file was selected
        if file_path:
            # Open the image using PIL
            image = Image.open(file_path)

            # Resize image to fit the label with anti-aliasing
            image = image.resize((300, 150), Image.LANCZOS)

            # Convert Image object to Tkinter PhotoImage object
            photo = ImageTk.PhotoImage(image)

            # Update the image label with the selected image
            self.image_label.config(image=photo)
            self.image_label.image = photo  # Keep a reference to avoid garbage collection

            # Pass the imported image to the detection function
            detection_result = cv2.imread(file_path)

            # Determine the type of defect detected
            defect_type = determine_defect_type(detection_result)

            # Call the appropriate function based on the detected defect type
            if defect_type == "Stain":
                detection_result = detect_stains(detection_result)
            elif defect_type == "Hole":
                detection_result = detect_holes(detection_result)
            elif defect_type == "Tear":
                detection_result = detect_tears(detection_result)

            # Display the result
            detection_page = DetectionResultPage(self, detection_result, defect_type)

    def back_to_main(self):
        self.destroy()  # Close the detection page
        self.master.deiconify()  # Show the main GUI again

class DetectionResultPage(tk.Toplevel):
    def __init__(self, master, result_image, defect_type):
        super().__init__(master)
        self.title("Detection Result")
        self.geometry("600x550")  # Increased height to accommodate the text box

        # Frame to contain all widgets
        main_frame = tk.Frame(self, bg='#f0f0f0', bd=2, relief='groove')
        main_frame.pack(expand=True, fill='both', padx=20, pady=20)

        # Resize the result image to fit within the window
        resized_image = self.resize_image(result_image, width=580, height=400)

        # Convert resized image to PhotoImage
        result_photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)))

        # Label to display detection result
        self.result_label = tk.Label(main_frame, image=result_photo, bd=2, relief='solid')
        self.result_label.image = result_photo
        self.result_label.pack(pady=10)

        # Text box to display detected defect type
        self.defect_type_label = tk.Entry(main_frame, font=('Helvetica', 14), justify='center', bd=2, relief='solid')
        self.defect_type_label.insert(tk.END, defect_type)
        self.defect_type_label.config(state='readonly')  # Make the text box read-only
        self.defect_type_label.pack(pady=10)

        # Back button
        back_button = ttk.Button(main_frame, text="Back", command=self.destroy, style='Custom.TButton')
        back_button.pack(pady=10)

    def resize_image(self, image, width, height):
        """
        Resize the image to fit within the specified width and height.
        """
        # Get the original image dimensions
        original_width, original_height = image.shape[1], image.shape[0]

        # Calculate the aspect ratio
        aspect_ratio = original_width / original_height

        # Calculate the new dimensions while maintaining the aspect ratio
        if aspect_ratio > width / height:
            new_width = width
            new_height = int(width / aspect_ratio)
        else:
            new_height = height
            new_width = int(height * aspect_ratio)

        # Resize the image
        resized_image = cv2.resize(image, (new_width, new_height))

        return resized_image


def detect_stains(image):
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for stain color in HSV
    lower_stain = np.array([0, 200, 200])  # Adjust as needed
    upper_stain = np.array([50, 255, 255])  # Adjust as needed

    # Create mask for stain color
    mask_stain = cv2.inRange(hsv, lower_stain, upper_stain)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask_stain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Count the number of detected stains
    num_stains = 0

    # Check if any stains are detected
    if contours:
        # Find the minimum bounding rectangle that encloses all contours
        x, y, w, h = cv2.boundingRect(np.concatenate(contours))

        # Draw a single rectangle around the minimum bounding rectangle
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, "Stain", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Set the number of detected stains to 1
        num_stains = 1

    # Print the number of detected stains
    cv2.putText(image, f"Defects Detected: {num_stains}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return image



def decrease_size(size, image):
    scale = size
    height = int(image.shape[0] * scale / 100)
    width = int(image.shape[1] * scale / 100)
    dim = (width, height)
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return image


def detect_holes(image):
    # Convert the image to HSV color space
    imageHsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Defining kernel size
    kernel = np.ones((7, 7), np.uint8)

    # Set lower bounds and upper bounds on each colour
    # Pink Colour
    minHSV_Glove = np.array([90, 120, 80])
    maxHSV_Glove = np.array([130, 255, 255])

    # Threshold the HSV image to get only color that wish to display
    mask = cv2.inRange(imageHsv, minHSV_Glove, maxHSV_Glove)

    # Removing noise that are in the mask
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask1 = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Result Image
    resultImage = cv2.bitwise_and(image, image, mask=mask1)
    result_image = detect_hole_defect(resultImage)
    update_defects_count(0)

    return result_image

def update_defects_count(count):
    print(f"Defects Detected: {count}")

def detect_hole_defect(image):
    image = decrease_size(100, image)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to remove noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection to detect edges
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize counter for detected defects
    defect_count = 0

    # Loop over the contours
    for contour in contours:
        # Calculate contour area
        area = cv2.contourArea(contour)
        # If contour area is smaller than a threshold, consider it as a hole
        if 10 < area < 100:  # Adjust the threshold as needed
            # Get the bounding box coordinates of the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Draw a rectangle around the detected hole
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Hole", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Increment defect counter
            defect_count += 1

    # Display the number of detected defects
    cv2.putText(image, f"Defects Detected: {defect_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return image


def detect_tears(image):
    # convert image to hsv colour space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # reduce dimension
    new_image = hsv.reshape(-1, 3)

    # KMeans Clustering
    k_means = KMeans(n_clusters=2, n_init=30)
    k_means.fit(new_image)

    # get the clustering result
    result = k_means.cluster_centers_[k_means.labels_]
    result = result.reshape(image.shape).astype(np.uint8)

    # Convert the image to grayscale
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

    # Thresholding and Binarization
    ret, th = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)

    # Reduce noise
    blur = cv2.medianBlur(th, 9)

    # Reduce small noise or hole
    morphology = cv2.morphologyEx(blur, cv2.MORPH_OPEN, np.ones((7, 7)))

    # Find tear with contour
    contours, hierarchy = cv2.findContours(morphology, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    tearCounter = 0
    for contour in contours:
        # draw out all the contour
        if cv2.contourArea(contour) >= 20000 or cv2.contourArea(contour) <= 20:
            continue
        perimeter = cv2.arcLength(contour, True)
        vertices = cv2.approxPolyDP(contour, perimeter * 0.02, True)
        x, y, w, h = cv2.boundingRect(vertices)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Put text on top of the rectangle
        cv2.putText(image, 'Tear', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        holeCounter = holeCounter + 1

    # Put text to show number of hole detected
    cv2.putText(image, 'Defects detected: ' + str(tearCounter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 0, 255),
                2, cv2.LINE_AA)

    # show result
    return image

def determine_defect_type(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_stain = np.array([0, 200, 200])  # Adjust as needed
    upper_stain = np.array([50, 255, 255])  # Adjust as needed
    mask_stain = cv2.inRange(hsv, lower_stain, upper_stain)
    contours, _ = cv2.findContours(mask_stain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return "Stain"

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if 80 < area < 87:  # Adjust the threshold as needed
            return "Hole"

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    new_image = image_hsv.reshape(-1, 3)
    k_means = KMeans(n_clusters=2, n_init=30)
    k_means.fit(new_image)
    labels = k_means.labels_
    unique_labels = np.unique(labels)
    if len(unique_labels) == 2:  # Assuming tears result in two clusters
        return "Hole"

    return None

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()

