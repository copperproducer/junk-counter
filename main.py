import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update the path as per your installation
import tkinter as tk
from tkinter import messagebox, font, simpledialog
import pyautogui
import cv2
import pytesseract
import numpy as np
import json
import keyboard
import pyttsx3
import winsound
import threading
import time

def take_screenshot_and_visualize_slots():
    # Capture the screen
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)  # Convert PIL Image to numpy array
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    # Get the scaled slot positions
    scaled_positions = get_scaled_positions()

    # Draw rectangles around the slots
    for (top_left, bottom_right) in scaled_positions:
        x1, y1 = top_left
        x2, y2 = bottom_right
        cv2.rectangle(screenshot_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw with green

    # Show the screenshot with the visualized slots
    cv2.imshow('Screenshot with Slots', screenshot_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def capture_screen_to_cv2():
    #wait for the info to be displayed
    time.sleep(0.1)
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)  # Convert the PIL Image to a numpy array
    # Scale down to 1080p
    screenshot_np = cv2.resize(screenshot_np, (1600,900), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV


def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get dimensions of the image
    height, width = gray.shape[:2]

    # Define the region of interest (ROI) for the trading area
    roi_top = int(height * 0.5)
    roi_bottom = int(height * 0.85)

    # Extract the ROI
    roi = gray[roi_top:roi_bottom, :]

    # Apply thresholding to the ROI
    _, thresh = cv2.threshold(roi, 150, 255, cv2.THRESH_BINARY)
    return thresh

# Create a threading lock
tts_lock = threading.Lock()

# Create a flag to track whether the announcement has been made
announcement_made = False

def play_confirmation_sound():
    threading.Thread(target=winsound.Beep, args=(400, 100)).start()

def read_total_platinum(total_platinum):
    threading.Thread(target=_read_total_platinum_thread, args=(total_platinum,)).start()

def _read_total_platinum_thread(total_platinum):
    global announcement_made  # Use global to modify the flag
    # Acquire the lock to ensure exclusive access
    if tts_lock.acquire(blocking=False):
        try:
            if not announcement_made:  # Check if announcement has been made
                time.sleep(.25)  # Wait for 1 second before reading
                announcement_made = True  # Set flag to True
                engine = pyttsx3.init()
                engine.say(f"{total_platinum}")
                engine.runAndWait()
        finally:
            # Release the lock
            tts_lock.release()



def find_ducat_value(image):
    processed_image = preprocess_image(image)
    custom_oem_psm_config = r'--oem 3 --psm 6'
    text = pytesseract.image_to_string(processed_image, config=custom_oem_psm_config)
    return extract_ducat_value(text)

def extract_ducat_value(text):
    import re
    match = re.search(r'(\d+)\s*Ducats?', text)
    if match:
        return int(match.group(1))
    return None


class PrimeJunkApp:
    def __init__(self, master):
        self.master = master
        master.title("Prime Junk Counter")
        self.font_style = font.Font(family="Arial", size=10)
        self.default_prices = {100: 9, 65: 4, 45: 3, 25: 1, 15:1}  # Default prices
        self.prices = self.default_prices.copy()
        self.load_settings()  # Load settings from file

        # Load slot positions from settings or set default values
        self.load_slot_positions()

        # Price adjustment section
        self.price_labels = {}
        self.price_entries = {}
        for ducats, platinum in self.prices.items():
            frame = tk.Frame(master)
            frame.pack(pady=2)
            label = tk.Label(frame, text=f"{ducats} Ducats:", font=self.font_style)
            label.pack(side=tk.LEFT)
            entry = tk.Entry(frame, width=5)
            entry.insert(0, str(platinum))
            entry.pack(side=tk.LEFT)
            self.price_labels[ducats] = label
            self.price_entries[ducats] = entry

        # Save, Restore Defaults, and Load settings buttons
        self.save_button = tk.Button(master, text="Save Settings", command=self.save_settings, font=self.font_style)
        self.save_button.pack(pady=5)
        self.restore_button = tk.Button(master, text="Restore Defaults", command=self.restore_defaults, font=self.font_style)
        self.restore_button.pack(pady=5)
        master.configure(bg='#333333')  # Dark background color

        # Font and Button Styling
        self.font_style_button = font.Font(family="Arial", size=12, weight="bold")
        self.font_style_label = font.Font(family="Arial", size=10)
        self.font_style_platinum = font.Font(family="Arial", size=24, weight="bold")

        # Slot Position Settings
        self.slot_settings_button = tk.Button(master, text="Adjust Slot Positions", command=self.adjust_slot_positions,
                                              font=self.font_style_button, bg="#444444", fg="#FFFFFF")
        self.slot_settings_button.pack(pady=10, padx=20, fill=tk.X)

        # Visualize Slots Button
        self.visualize_slots_button = tk.Button(master, text="Visualize Slots", command=take_screenshot_and_visualize_slots,
                                                font=self.font_style_button, bg="#444444", fg="#FFFFFF")
        self.visualize_slots_button.pack(pady=10, padx=20, fill=tk.X)

        # Scanned Slots Visual Indicators
        self.slot_indicators = []
        slots_frame = tk.Frame(master, bg='#333333')
        slots_frame.pack(pady=10)
        for i in range(6):
            label = tk.Label(slots_frame, text=f"Slot {i + 1}", bg="gray", width=12, height=2)
            label.pack(side=tk.LEFT, padx=5)
            self.slot_indicators.append(label)

        # Initialize the Scanned Slots Label
        self.scanned_slots_label = tk.Label(master, text="Scanned Slots: None", font=self.font_style_label,
                                            bg='#333333', fg='white')
        self.scanned_slots_label.pack(pady=10)

        # Enhanced visual styling
        master.configure(bg='#333333')  # Set a dark background color for the window

        # Modern font styles
        self.font_style_button = font.Font(family="Arial", size=12, weight="bold")
        self.font_style_label = font.Font(family="Arial", size=10)
        self.font_style_platinum = font.Font(family="Arial", size=24, weight="bold")

        # Button Styling
        button_bg = "#444444"
        button_fg = "#FFFFFF"

        # Activate/Deactivate button
        self.toggle_button = tk.Button(master, text="Activate", command=self.toggle,
                                       font=self.font_style_button, bg=button_bg, fg=button_fg)
        self.toggle_button.pack(pady=10, padx=20, fill=tk.X)

        # Polling Rate Controls
        self.polling_rate_label = tk.Label(master, text="Polling Rate (times per second):",
                                           font=self.font_style_label, bg='#333333', fg='white')
        self.polling_rate_label.pack()
        self.polling_rate = tk.Scale(master, from_=1, to=360, orient="horizontal", resolution=1,
                                     font=self.font_style_label, troughcolor="#555555", sliderlength=20, width=15)
        self.polling_rate.pack(fill=tk.X, padx=20, pady=5)

        # Platinum Counter
        self.total_platinum = 0
        self.total_platinum_label = tk.Label(master, text=f"Total Platinum: {self.total_platinum}",
                                             font=self.font_style_platinum, bg='#333333', fg='#FFD700')
        self.total_platinum_label.pack(pady=10)

        # Reset button
        self.reset_button = tk.Button(master, text="Reset", command=self.reset_total,
                                      font=self.font_style_button, bg=button_bg, fg=button_fg)
        self.reset_button.pack(pady=10, padx=20, fill=tk.X)

        # Bind the right Ctrl key to the reset_total method
        keyboard.on_press_key("right ctrl", self.reset_total)



        self.active = False
        self.scanned_slots = set()  # Track scanned slots to avoid multiple counts
    def toggle(self):
        self.active = not self.active
        self.toggle_button.config(text="Deactivate" if self.active else "Activate")
        if self.active:
            self.periodic_scan()

    def update_slot_indicator(self, index, scanned):
        """ Update the visual indicator for the given slot index """
        if scanned:
            self.slot_indicators[index].config(bg='gold')
        else:
            self.slot_indicators[index].config(bg='gray')

    def periodic_scan(self):
        if self.active:
            mouse_over, slot_index = is_mouse_over_slot()
            if mouse_over and slot_index not in self.scanned_slots:
                screen_image = capture_screen_to_cv2()
                ducat_value = find_ducat_value(screen_image)
                if ducat_value is not None:
                    self.scanned_slots.add(slot_index)  # Mark this slot as scanned
                    self.update_slot_indicator(slot_index - 1, True)  # Update visual indicator
                    platinum_value = self.ducat_to_platinum(ducat_value)
                    self.total_platinum += platinum_value
                    self.update_total_platinum_display()
                    play_confirmation_sound()  # Play confirmation sound
            if len(self.scanned_slots) == 6:  # Check if all slots have been scanned
                read_total_platinum(self.total_platinum)  # Read total platinum using text-to-speech
            self.master.after(1000 // self.polling_rate.get(), self.periodic_scan)

    def reset_total(self, event=None):  # Add event=None parameter to accept event argument from keyboard event
        global announcement_made  # Use global to modify the flag
        self.total_platinum = 0
        self.scanned_slots.clear()  # Reset scanned slots
        for indicator in self.slot_indicators:  # Reset slot indicators to gray
            indicator.config(bg='gray')
        self.update_total_platinum_display()
        tts_lock.acquire()
        tts_lock.release()
        announcement_made = False

    def ducat_to_platinum(self, ducats):


        for ducat_value, platinum_value in self.prices.items():

            ducat_value = int(ducat_value)
            if ducats >= ducat_value:
                return platinum_value
        return 0

    def update_total_platinum_display(self):
        """ Update display of total platinum and scanned slots. """
        scanned_slots_text = ", ".join(map(str, self.scanned_slots)) if self.scanned_slots else "None"
        self.scanned_slots_label.config(text=f"Scanned Slots: {scanned_slots_text}")
        self.total_platinum_label.config(text=f"Total Platinum: {self.total_platinum}")

    def mute_sound(self):
        # Mute the sound
        winsound.Beep(0, 0)  # Just resetting the sound parameters to 0 will mute it

    def mute_text_to_speech(self):
        # Mute text-to-speech
        engine = pyttsx3.init()
        engine.setProperty('volume', 0)  # Set volume to 0 to mute

    def unmute_sound(self):
        # Unmute the sound
        pass  # No need to unmute for system sound

    def unmute_text_to_speech(self):
        # Unmute text-to-speech
        engine = pyttsx3.init()
        engine.setProperty('volume', 1)  # Set volume back to 1 for normal volume

    def save_settings(self):
        for ducats in self.prices:
            try:
                self.prices[ducats] = int(self.price_entries[ducats].get())
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter a valid number for platinum prices.")
                return
        with open('settings.json', 'w') as f:
            json.dump(self.prices, f)
        messagebox.showinfo("Settings", "Settings saved successfully!")

    def load_settings(self):
        try:
            with open('settings.json', 'r') as f:
                self.prices = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.prices = self.default_prices.copy()

    def restore_defaults(self):
        self.prices = self.default_prices.copy()
        for ducats, platinum in self.prices.items():
            self.price_entries[ducats].delete(0, tk.END)
            self.price_entries[ducats].insert(0, str(platinum))
        messagebox.showinfo("Settings", "Restored to default settings!")

    def adjust_slot_positions(self):
        # Use simple dialogs to get new slot position values
        start_x = simpledialog.askinteger("Input", "Enter start X position:", parent=self.master)
        start_y = simpledialog.askinteger("Input", "Enter start Y position:", parent=self.master)
        slot_width = simpledialog.askinteger("Input", "Enter slot width:", parent=self.master)
        slot_height = simpledialog.askinteger("Input", "Enter slot height:", parent=self.master)
        gap_between_slots = simpledialog.askinteger("Input", "Enter gap between slots:", parent=self.master)
        if all(v is not None for v in [start_x, start_y, slot_width, slot_height, gap_between_slots]):
            update_slot_positions(start_x, start_y, slot_width, slot_height, gap_between_slots)
            messagebox.showinfo("Info", "Slot positions updated!")

    def get_default_slot_positions(self):
        """
        Define default slot positions.
        """
        start_x = 218
        start_y = 670  # Adjust this value if necessary to move up or down
        slot_width = 220
        slot_height = 220
        gap_between_slots = 33

        # Calculate positions for each slot
        default_slot_positions = []
        for i in range(6):  # Assuming 6 slots
            top_left_x = start_x + i * (slot_width + gap_between_slots)
            top_left_y = start_y
            bottom_right_x = top_left_x + slot_width
            bottom_right_y = top_left_y + slot_height
            default_slot_positions.append(((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)))

        return default_slot_positions

    def load_slot_positions(self):
        try:
            with open('slot_positions.json', 'r') as f:
                self.slot_positions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.slot_positions = self.get_default_slot_positions()
            self.save_slot_positions()  # Save default positions

    def save_slot_positions(self):
        with open('slot_positions.json', 'w') as f:
            json.dump(self.slot_positions, f)

    def adjust_slot_positions(self):
        # Use simple dialogs to get new slot position values
        start_x = simpledialog.askinteger("Input", "Enter start X position:", parent=self.master)
        start_y = simpledialog.askinteger("Input", "Enter start Y position:", parent=self.master)
        slot_width = simpledialog.askinteger("Input", "Enter slot width:", parent=self.master)
        slot_height = simpledialog.askinteger("Input", "Enter slot height:", parent=self.master)
        gap_between_slots = simpledialog.askinteger("Input", "Enter gap between slots:", parent=self.master)
        if all(v is not None for v in [start_x, start_y, slot_width, slot_height, gap_between_slots]):
            self.update_slot_positions(start_x, start_y, slot_width, slot_height, gap_between_slots)
            messagebox.showinfo("Info", "Slot positions updated!")

    def update_slot_positions(self, start_x, start_y, slot_width, slot_height, gap_between_slots):
        """
        Update the slot positions based on user input and save them to settings.
        """
        # Calculate new positions for each slot based on the input values
        self.slot_positions = []
        for i in range(6):  # Assuming 6 slots
            top_left_x = start_x + i * (slot_width + gap_between_slots)
            top_left_y = start_y
            bottom_right_x = top_left_x + slot_width
            bottom_right_y = top_left_y + slot_height
            self.slot_positions.append(((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)))

        # Save the updated positions to settings
        self.save_slot_positions()

    def get_default_slot_positions(self):
        """
        Define default slot positions.
        """
        start_x = 218
        start_y = 670  # Adjust this value if necessary to move up or down
        slot_width = 220
        slot_height = 220
        gap_between_slots = 33

        # Calculate positions for each slot
        default_slot_positions = []
        for i in range(6):  # Assuming 6 slots
            top_left_x = start_x + i * (slot_width + gap_between_slots)
            top_left_y = start_y
            bottom_right_x = top_left_x + slot_width
            bottom_right_y = top_left_y + slot_height
            default_slot_positions.append(((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)))

        return default_slot_positions

    def update_total_platinum_display(self):
        scanned_slots_text = ", ".join(map(str, self.scanned_slots)) if self.scanned_slots else "None"
        self.scanned_slots_label.config(text=f"Scanned Slots: {scanned_slots_text}")
        self.total_platinum_label.config(text=f"Total Platinum: {self.total_platinum}")

def is_mouse_over_slot():
    mouse_x, mouse_y = pyautogui.position()
    slots = get_scaled_positions()
    for index, (top_left, bottom_right) in enumerate(slots):
        if top_left[0] <= mouse_x <= bottom_right[0] and top_left[1] <= mouse_y <= bottom_right[1]:
            return True, index + 1
    return False, None

def get_slot_positions():
    # Define top-left corner of the first slot
    start_x = 218
    start_y = 670  # Adjust this value if necessary to move up or down
    slot_width = 220
    slot_height = 220
    gap_between_slots = 33

    # Calculate positions for each slot
    slot_positions = []
    for i in range(6):  # Assuming 6 slots
        top_left_x = start_x + i * (slot_width + gap_between_slots)
        top_left_y = start_y
        bottom_right_x = top_left_x + slot_width
        bottom_right_y = top_left_y + slot_height
        slot_positions.append(((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)))

    return slot_positions

def get_scaled_positions():
    screen_width, screen_height = pyautogui.size()
    base_resolution = (1920, 1080)
    slot_positions = get_slot_positions()
    scaled_positions = [
        ((int(x1 * screen_width / base_resolution[0]), int(y1 * screen_height / base_resolution[1])),
         (int(x2 * screen_width / base_resolution[0]), int(y2 * screen_height / base_resolution[1])))
        for (x1, y1), (x2, y2) in slot_positions
    ]
    return scaled_positions

def update_slot_positions(start_x, start_y, slot_width, slot_height, gap_between_slots):
    """
    Update the global slot positions based on user input.
    """
    global slot_positions  # Assuming slot_positions is a global variable used throughout your app

    # Calculate new positions for each slot based on the input values
    slot_positions = []
    for i in range(6):  # Assuming 6 slots
        top_left_x = start_x + i * (slot_width + gap_between_slots)
        top_left_y = start_y
        bottom_right_x = top_left_x + slot_width
        bottom_right_y = top_left_y + slot_height
        slot_positions.append(((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)))

    # Optionally, update any GUI components or internal states that depend on these positions
    print("Slot positions updated:", slot_positions)

root = tk.Tk()
app = PrimeJunkApp(root)
root.mainloop()




# def find_ducat_icon(screen_image, ducat_icon):
#     height, width, _ = screen_image.shape
#     # Increase the cropped area to ensure we don't miss the text
#     cropped_screen = screen_image[height//2 - 100:height + 100, 0:width]
#
#     sift = cv2.SIFT_create()
#     keypoints1, descriptors1 = sift.detectAndCompute(ducat_icon, None)
#     keypoints2, descriptors2 = sift.detectAndCompute(cropped_screen, None)
#
#     FLANN_INDEX_KDTREE = 1
#     index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
#     search_params = dict(checks=50)
#
#     flann = cv2.FlannBasedMatcher(index_params, search_params)
#     matches = flann.knnMatch(descriptors1, descriptors2, k=2)
#
#     good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
#     MIN_MATCH_COUNT = 10
#     if len(good_matches) > MIN_MATCH_COUNT:
#         src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#         dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
#         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
#         h, w = ducat_icon.shape[:2]
#         pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
#         dst = cv2.perspectiveTransform(pts, M)
#         rect = cv2.boundingRect(dst)
#         x, y, w, h = rect
#         # Expand the ROI slightly to capture adjacent text
#         roi = cropped_screen[y-100:y+h+100, x-100:x+w+100]
#
#         # Convert ROI to grayscale and enhance features
#         roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
#         roi_enhanced = cv2.equalizeHist(roi_gray)
#
#         # OCR with Tesseract
#         custom_config = r'--oem 3 --psm 6'
#         text = pytesseract.image_to_string(roi_enhanced, config=custom_config)
#         ducat_amount = extract_ducat_value(text)
#
#         return screen_image, (int(x + w/2), int(y + h/2 + height//2)), ducat_amount
#     else:
#         print("Ducat icon not found. Matches found:", len(good_matches))
#         return screen_image, None, None




#
#
#
#     # Capture the screen
#
# for i in range(10):
#     screen_image = capture_screen_to_cv2()
#
#     # Load the ducat icon image
#     ducat_icon = cv2.imread('ducat_icon.png')
#
#     # Find the ducat icon using the SIFT-based function
#     result_image, found_location, match_count = find_ducat_icon(screen_image, ducat_icon)
#
#     if found_location:
#         print("Ducat icon found at position:", found_location, "with", match_count, "good matches.")
#         #cv2.imshow('Result with Debugging Visuals', result_image)
#         cv2.waitKey(0)  # Wait for a key press to close the window
#         cv2.destroyAllWindows()
#     else:
#         print("Ducat icon not found. Matches found:", match_count)
#
#

