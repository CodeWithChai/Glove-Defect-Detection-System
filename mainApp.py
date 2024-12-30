import tkinter as tk
from ovenDetection import OvenDetection
from leatherGlove.main import LeatherGloveDetection
from Latex.LatexGlove import latex_detection_ui
from surgicalDetection import SurgicalDetectionPage


class MainApplication(tk.Tk):
    def __init__(self):
        #Initialize the main application
        super().__init__()
        self.title("Glove Defect Detection System")
        self.geometry("800x500")
        
        # Button for latex glove detection page
        detect_button = tk.Button(self, text="Latex Glove Defect Detection", command=self.latex_detection_page, font=("Arial", 12))
        detect_button.pack(pady=40)
        
        # Button for surgical glove detection page
        detect_button2 = tk.Button(self, text="Surgical Glove Defect Detection", command=self.surgical_detection_page, font=("Arial", 12))
        detect_button2.pack(pady=40)
        
        # Button for oven glove detection page
        detect_button3 = tk.Button(self, text="Oven Glove Defect Detection", command=self.oven_detection_page, font=("Arial", 12))
        detect_button3.pack(pady=40)
        
        # Button for leather glove detection page
        detect_button4 = tk.Button(self, text="Leather Glove Defect Detection", command=self.leather_detection_page, font=("Arial", 12))
        detect_button4.pack(pady=40)

    def latex_detection_page(self):
        latex_detection_ui()
        # glove_img = select_image()
        # if glove_img is not None:
        #     detect_defects(glove_img)
    
    def surgical_detection_page(self):
        SurgicalDetectionPage(self)

    def oven_detection_page(self):
        OvenDetection(self)
    
    def leather_detection_page(self):
        LeatherGloveDetection()

if __name__ == "__main__":
    # Create an instance of the MainApplication class
    app = MainApplication()
    # Start the Tkinter event loop to run application
    app.mainloop()
