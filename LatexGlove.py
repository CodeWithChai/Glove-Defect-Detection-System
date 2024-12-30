import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import ImageTk, Image

# Function for preprocessing the image
def preprocess_image(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return binary

# Function for detecting holes in gloves
def detect_holes(glove_img):
    try:
        # Preprocess the image
        binary = preprocess_image(glove_img)

        # Find contours in the binary image
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours based on area
        min_contour_area = 100 
        large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        # Find the contour representing the hole 
        hole_contour = None
        for contour in large_contours:
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
            if len(approx) >= 5: 
                hole_contour = contour
                break

        # If no hole is detected
        if hole_contour is None:
            root = tk.Tk()
            root.withdraw()  
            messagebox.showinfo("No Hole Detected", "No hole was detected in the image.")
            root.destroy() 

        else:
            # Contour representing the hole on the original image
            cv2.drawContours(glove_img, [hole_contour], -1, (0, 255, 0), 2)
            
            display_image_gui(glove_img, "Detected Hole")

    except Exception as e:
        print(f"An error occurred: {e}")

    return glove_img

# Function for detecting stains in gloves
def detect_stains(img_path):
    try:
        # Load the image
        img = cv2.imread(img_path)

        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Define color ranges for pink and white
        lower_pink = np.array([130, 60, 110])
        upper_pink = np.array([180, 180, 180])
        lower_white = np.array([160, 160, 160])
        upper_white = np.array([255, 255, 255])

        # Mask pink and white colors
        mask_pink = cv2.inRange(img_rgb, lower_pink, upper_pink)
        mask_white = cv2.inRange(img_rgb, lower_white, upper_white)

        # Combine masks
        mask = cv2.bitwise_or(mask_pink, mask_white)

        # Apply mask to original image
        masked_img = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

        # Convert masked image to grayscale
        gray = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)

        # Apply thresholding
        ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get largest contour (assumed to be the glove)
        max_contour = max(contours, key=cv2.contourArea)

        # Create mask for the glove
        glove_mask = np.zeros_like(gray)
        cv2.drawContours(glove_mask, [max_contour], -1, (255), thickness=cv2.FILLED)

        # Apply mask to original image to get the glove only
        glove_img = cv2.bitwise_and(img_rgb, img_rgb, mask=glove_mask)

        # Convert glove image to grayscale
        gray_glove = cv2.cvtColor(glove_img, cv2.COLOR_RGB2GRAY)

        # Apply Gaussian blur to smooth the image
        blurred = cv2.GaussianBlur(gray_glove, (5, 5), 0)

        # Define lower and upper bounds for stain color in grayscale
        lower_stain = 30  # Adjust as needed
        upper_stain = 100  # Adjust as needed

        # Create a mask for stain color
        mask_stain = cv2.inRange(blurred, lower_stain, upper_stain)

        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        mask_stain = cv2.morphologyEx(mask_stain, cv2.MORPH_OPEN, kernel)
        mask_stain = cv2.morphologyEx(mask_stain, cv2.MORPH_CLOSE, kernel)

        # Find contours in the stain mask
        contours, _ = cv2.findContours(mask_stain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around the detected stains
        stained_img = glove_img.copy()
        stains_detected = False
        for contour in contours:
            # Filter out small contours
            area = cv2.contourArea(contour)
            if area > 100:  
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(stained_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                stains_detected = True

        if stains_detected:
            return stained_img
        else:
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


# Function for detecting defects in gloves
def detect_defects(defect_var):
    defect_type = defect_var.get()
    glove_img = None

    if defect_type == "Holes":
        glove_img = select_image()
        if glove_img is not None:
            detect_holes(glove_img)
    elif defect_type == "Stains":
        glove_img_path = select_image_path()
        if glove_img_path is not None:
            stained_img = detect_stains(glove_img_path)
            if stained_img is not None:
                display_image_gui(stained_img, "Detected Stains")
            else:
                messagebox.showinfo("No Stains Detected", "No stains were detected in the gloves.")
    elif defect_type == "Missing Finger":
        glove_img = select_image()
        if glove_img is not None:
            result_img, missing_fingers = detect_missing_fingers(glove_img)
            if result_img is not None:
                missing_fingers_text = f"Number of missing fingers: {missing_fingers}"
                display_image_gui(result_img, "Detected Missing Fingers", missing_fingers_text)

    return glove_img


# Function to select and load an image
def select_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            # Load the image
            img = cv2.imread(file_path)

            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            return img_rgb

        except Exception as e:
            print(f"An error occurred: {e}")
            return None


# Function to select and return image path
def select_image_path():
    file_path = filedialog.askopenfilename()
    return file_path


# Function to display an image in a GUI 
def display_image_gui(image, title, missing_fingers_text=None):
    window = tk.Toplevel()
    window.title(title)

    image_pil = Image.fromarray(image)  
    image_tk = ImageTk.PhotoImage(image_pil)  

    label = tk.Label(window, image=image_tk)
    label.image = image_tk  
    label.pack()

    if missing_fingers_text:
        missing_label = tk.Label(window, text=missing_fingers_text, font=("Arial", 12), fg="white", bg="black")
        missing_label.pack(pady=5)


# Function to update defect detection when radio button changes
def defect_selection_changed(defect_var):
    detect_defects(defect_var)


# Function for detecting missing fingers within gloves
def detect_missing_fingers(glove_img):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(glove_img, cv2.COLOR_RGB2GRAY)

        # Apply thresholding
        ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get largest contour (assumed to be the glove)
        max_contour = max(contours, key=cv2.contourArea)

        # Create mask for the glove
        glove_mask = np.zeros_like(gray)
        cv2.drawContours(glove_mask, [max_contour], -1, (255), thickness=cv2.FILLED)

        # Apply mask to original image to get the glove only
        glove_only_img = cv2.bitwise_and(glove_img, glove_img, mask=glove_mask)

        # Compute convex hull and convexity defects
        hull = cv2.convexHull(max_contour, returnPoints=False)
        defects = cv2.convexityDefects(max_contour, hull)

        # Initialize variables
        missing_fingers = 0
        finger_bboxes = []

        # Loop over defects and detect missing fingers
        if defects is not None:
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(max_contour[s][0])
                end = tuple(max_contour[e][0])
                far = tuple(max_contour[f][0])

                # Check if the defect is a missing finger
                if d > 16000:
                    missing_fingers += 1
                    finger_bboxes.append(cv2.boundingRect(np.array([start, end, far])))

        # Draw bounding boxes for missing fingers on the glove
        for bbox in finger_bboxes:
            x, y, w, h = bbox
            cv2.rectangle(glove_only_img, (x, y), (x + w, y + h), (0, 0, 255), 2)


        if missing_fingers == 0:
            root = tk.Tk()
            root.withdraw()  
            messagebox.showinfo("No Missing Fingers Detected", "No missing fingers were detected in the image.")
            root.destroy() 

        return glove_only_img, missing_fingers

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, 0




def latex_detection_ui():
    # Create main GUI 
    root = tk.Tk()
    root.title("Glove Defect Detection")

    # Title 
    title_label = tk.Label(root, text="Welcome to Latex GDD Page", font=("Arial", 16))
    title_label.pack(pady=30)

    # Radio buttons for defect type selection
    defect_var = tk.StringVar()
    defect_var.set("Holes") 

    defect_frame = tk.Frame(root)
    defect_frame.pack()

    defect_label = tk.Label(defect_frame, text="Select defect type:")
    defect_label.grid(row=0, column=0, padx=5)

    holes_radio = tk.Radiobutton(defect_frame, text="Holes", variable=defect_var, value="Holes",
                                command=defect_selection_changed(defect_var))
    holes_radio.grid(row=0, column=1, padx=5)

    stains_radio = tk.Radiobutton(defect_frame, text="Stains", variable=defect_var, value="Stains",
                                command=defect_selection_changed(defect_var))
    stains_radio.grid(row=0, column=2, padx=5)

    missing_finger_radio = tk.Radiobutton(defect_frame, text="Missing Finger", variable=defect_var,
                                        value="Missing Finger", command=defect_selection_changed(defect_var))
    missing_finger_radio.grid(row=0, column=3, padx=5)
