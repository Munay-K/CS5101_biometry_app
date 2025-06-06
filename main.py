import os
import cv2
import numpy as np
from gui import BiometricAppGUI
import tkinter as tk

if __name__ == "__main__":
    root = tk.Tk()
    app = BiometricAppGUI(root)
    root.mainloop()
