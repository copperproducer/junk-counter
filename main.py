import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update the path as per your installation
import keyboard
import pyttsx3
import pytesseract
import winsound
import pyperclip
import pandas as pd
import tkinter as tk
from tkinter import messagebox, font, simpledialog
import pyautogui
import cv2
import numpy as np
import json
import threading
import time
import easyocr
#is cuda available
import torch
print(f'CUDA available: {torch.cuda.is_available()}')




def take_screenshot_and_visualize_slots():
    # Capture the screen
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)  # Convert PIL Image to numpy array
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    # Get the scaled slot positions
    scaled_positions = get_scaled_positions()

    #scale up scaled_positions by the necessary amount for the user's screen resolution
    base_resolution = pyautogui.size()
    target_resolution = (1920, 1080)  # New target resolution

    scaled_positions = [
        ((int(x1 * base_resolution[0] / target_resolution[0]), int(y1 * base_resolution[1] / target_resolution[1])),
         (int(x2 * base_resolution[0] / target_resolution[0]), int(y2 * base_resolution[1] / target_resolution[1]))
        )
        for (x1, y1), (x2, y2) in scaled_positions
    ]

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
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)  # Convert the PIL Image to a numpy array
    # Scale down to 1080p
    screenshot_np = cv2.resize(screenshot_np, (1920,1080), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV


def load_ducat_values():
    try:
        return pd.read_csv('ducat_values.csv', index_col='Part Name')
    except FileNotFoundError:
        return pd.DataFrame(columns=['Part Name', 'Value']).set_index('Part Name')



def preprocess_image(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get dimensions of the image
    height, width = gray.shape[:2]

    # Define the region of interest (ROI) for the trading area
    roi_top = int(height * 0.45)
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


import csv

def read_ducat_values():
    """Read ducat values from the CSV file."""
    try:
        with open('ducat_values.csv', mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return {row['Part Name'].lower(): int(row['Value']) for row in reader}
    except FileNotFoundError:
        return {}


def take_screenshot():
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)
    return screenshot_bgr


from cv2 import fastNlMeansDenoisingColored


# Initialize the OCR reader with English language support
reader = easyocr.Reader(['en'])
def extract_and_format_part_name(part_name_image):
    """Use EasyOCR to extract and format the part name from an image."""
    results = reader.readtext(part_name_image, paragraph=True)  # Use paragraph mode for better context understanding
    part_names = [text[1].lower() for text in results]  # Extract the text and convert to lowercase
    formatted_text = ' '.join(part_names)  # Join part names into a single string
    return formatted_text









from fuzzywuzzy import process

def find_closest_match(part_name, known_parts):
    """Find the closest match for a part name in known parts using fuzzy matching."""
    if known_parts:
        # Use fuzzy matching to find the closest match with a similarity score
        result = process.extractOne(part_name, known_parts.keys(), score_cutoff=95)  # 80 is the similarity threshold
        if result:
            best_match, score = result
            return best_match
    return None




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
        self.last_left_time = 0  # Track the last time the mouse left a slot
        self.last_scanned_time = 0  # Track the last time a slot was scanned
        self.last_mouse_slot = None  # No slot is selected initially
        self.last_enter_time = 0  # Initialize the time the mouse entered a slot
        self.last_preemptive_scan_time = 0  # Track the last preemptive scan time



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
        self.polling_rate_label = tk.Label(master, text="Polling Rate (mouse location checks per second):",
                                           font=self.font_style_label, bg='#333333', fg='white')
        self.polling_rate_label.pack()
        self.polling_rate = tk.Scale(master, from_=1, to=360, orient="horizontal", resolution=1,
                                     font=self.font_style_label, troughcolor="#555555", sliderlength=20, width=15)
        self.polling_rate.set(360)  # Set default value

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

        self.clipboard_button = tk.Button(master, text="Copy Trade Message",
                                          command=self.copy_trade_message_to_clipboard,
                                          font=self.font_style_button, bg="#444444", fg="#FFFFFF")
        self.clipboard_button.pack(pady=10, padx=20, fill=tk.X)

        # Bind the right Ctrl key to the reset_total method
        keyboard.on_press_key("right ctrl", self.reset_total)



        self.active = False
        self.scanned_slots = set()  # Track scanned slots to avoid multiple counts

        # Preemptive Scan Interval Controls
        self.preemptive_scan_interval_label = tk.Label(master, text="Automatic Scan Interval (seconds):",
                                                       font=self.font_style_label, bg='#333333', fg='white')
        self.preemptive_scan_interval_label.pack()
        self.preemptive_scan_interval = tk.Scale(master, from_=0.1, to=10, orient="horizontal", resolution=0.1,
                                                 font=self.font_style_label, troughcolor="#555555", sliderlength=20,
                                                 width=15)
        self.preemptive_scan_interval.set(1)  # Set default value
        self.preemptive_scan_interval.pack(fill=tk.X, padx=20, pady=5)




        self.allow_preemptive_scan = False  # Add this line

    def set_color_filter(self, color):
        self.color_filter = color
        messagebox.showinfo("Color Set", f"Color set to {color}")



    def toggle(self):
        self.active = not self.active
        self.toggle_button.config(text="Deactivate" if self.active else "Activate")
        self.allow_preemptive_scan = self.active  # Update this flag based on activity
        if self.active:
            self.periodic_scan()
        else:
            self.scanned_slots.clear()  # Optionally clear slots on deactivation


    def update_slot_indicator(self, index, scanned):
        """ Update the visual indicator for the given slot index """
        if scanned:
            self.slot_indicators[index].config(bg='gold')
        else:
            self.slot_indicators[index].config(bg='gray')

    # Update the periodic_scan method
    def periodic_scan(self):
        if self.active:
            current_time = time.time()
            # Ensure only one preemptive scan runs, and only if allowed
            if self.allow_preemptive_scan and (
                    current_time - self.last_preemptive_scan_time > self.preemptive_scan_interval.get()):
                self.preemptive_scan()
                self.last_preemptive_scan_time = current_time

            # Check if the mouse is over a slot
            mouse_over, slot_index = is_mouse_over_slot()
            current_time = time.time()  # Capture the current time

            # Handling mouse and slot interaction
            if mouse_over:
                if self.last_mouse_slot != slot_index:
                    self.last_mouse_slot = slot_index
                    self.last_enter_time = current_time
                # Scanning logic
                elif slot_index == self.last_mouse_slot and (current_time - self.last_enter_time > 0.1):
                    if slot_index not in self.scanned_slots:
                        self.scan_slot(slot_index)

            # Continue the periodic scan
            self.master.after(1000 // self.polling_rate.get(), self.periodic_scan)

    def capture_slot_image(self, slot_index):
        """Capture the image of the bottom 40% of the specified slot."""
        slot_positions = get_scaled_positions()  # Corrected this line
        top_left, bottom_right = slot_positions[slot_index]
        x1, y1, x2, y2 = *top_left, *bottom_right
        height = y2 - y1
        roi_top = y1 + int(height * 0.6)
        roi_bottom = y2 + int(height * 0.6)
        screen_image = capture_screen_to_cv2()



        return screen_image[roi_top:roi_bottom, x1:x2]

    def scan_slot(self, slot_index):
        """ Function to handle the scanning of a specific slot. """
        try:
            screen_image = capture_screen_to_cv2()
            ducat_value = find_ducat_value(screen_image)
            if ducat_value is not None:
                self.update_platinum_and_slots(slot_index, ducat_value)
            else:
                # If ducat value is None, it's possible that the preemptive scan did not work correctly
                # Attempt to scan again
                part_name_image = self.capture_slot_image(slot_index)  # Include self here
                if part_name_image.size > 0:  # Check if the image is empty
                    part_name_text = extract_and_format_part_name(part_name_image)
                    closest_match = find_closest_match(part_name_text, read_ducat_values())
                    if closest_match:
                        self.update_platinum_and_slots(slot_index, read_ducat_values()[closest_match])
                    else:
                        # If still no match, keep slot in pending state
                        print(f"Failed to recognize slot {slot_index}. Please hover over again.")
                else:
                    print(f"Slot {slot_index} image capture failed, possibly due to a size issue.")
        except Exception as e:
            # Log the exception for debugging
            print(f"Error scanning slot {slot_index}: {e}")

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

    def copy_trade_message_to_clipboard(self):
        # Fetch prices from entries and sort by ducat value
        price_data = {int(ducats): int(self.price_entries[ducats].get()) for ducats in self.prices}
        sorted_prices = sorted(price_data.items(), key=lambda x: -x[0])  # Sort by ducat value in descending order

        # Combine 15 and 25 ducat items if they have the same price
        if price_data.get(15) == price_data.get(25):
            price_data[25] = price_data[15]  # Combine under a single key
            sorted_prices = [(k, v) for k, v in sorted_prices if k != 15]  # Remove the 15 entry from sorted list
            sorted_prices = [(25, v) if k == 25 else (k, v) for k, v in sorted_prices]  # Replace 25 with "25/15"

        # Build the trade message
        message_parts = []
        for ducats, platinum in sorted_prices:
            if ducats == 25 and price_data.get(15) == price_data[25]:
                message_parts.append(f"25/15:ducats:={platinum}:platinum:")
            else:
                message_parts.append(f"{ducats}:ducats:={platinum}:platinum:")
        trade_message = f"WTB Prime Junk |{'|'.join(message_parts)}| Full Trades Only"

        # Copy to clipboard
        pyperclip.copy(trade_message)
        #play confirmation sound
        play_confirmation_sound()

    def preemptive_scan(self):
        """Modified preemptive scan using EasyOCR."""
        ducat_values_df = load_ducat_values()
        full_screen_image = capture_screen_to_cv2()
        known_parts = read_ducat_values()  # Load known part values
        for i, (top_left, bottom_right) in enumerate(get_scaled_positions(), start=1):
            if i not in self.scanned_slots:
                slot_image = full_screen_image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
                bottom_40_start = int(slot_image.shape[0] * 0.6)
                part_name_image = slot_image[bottom_40_start:, :]
                part_name_text = extract_and_format_part_name(part_name_image)
                closest_match = find_closest_match(part_name_text, known_parts)
                print(f"Slot {i}: {part_name_text} -> {closest_match}")
                if closest_match:
                    self.update_platinum_and_slots(i, known_parts[closest_match])

    def update_platinum_and_slots(self, slot_index, ducat_value):
        if slot_index not in self.scanned_slots:
            self.scanned_slots.add(slot_index)
            self.update_slot_indicator(slot_index - 1, True)
            platinum_value = self.ducat_to_platinum(ducat_value)
            self.total_platinum += platinum_value
            self.update_total_platinum_display()
            play_confirmation_sound()

            # Check if all slots have been scanned and read total platinum aloud
            if len(self.scanned_slots) == 6:  # Assuming there are 6 slots
                read_total_platinum(self.total_platinum)


def is_mouse_over_slot():
    mouse_x, mouse_y = pyautogui.position()
    slots = get_scaled_positions()
    for index, (top_left, bottom_right) in enumerate(slots):
        if top_left[0] <= mouse_x <= bottom_right[0] and top_left[1] <= mouse_y <= bottom_right[1]:
            return True, index + 1
    return False, None

def get_slot_positions():
    #get the actual screen resolution
    base_resolution = pyautogui.size()
    target_resolution = (1920, 1080)  # New target resolution

    scale_x = target_resolution[0] / base_resolution[0]
    scale_y = target_resolution[1] / base_resolution[1]

    # Define the positions based on a 4K resolution
    start_x = 218 * scale_x
    start_y = 675 * scale_y
    slot_width = 220 * scale_x
    slot_height = 220 * scale_y
    gap_between_slots = 34 * scale_x

    slot_positions = []
    for i in range(6):  # Assuming 6 slots
        top_left_x = start_x + i * (slot_width + gap_between_slots)
        top_left_y = start_y
        bottom_right_x = top_left_x + slot_width
        bottom_right_y = top_left_y + slot_height
        slot_positions.append(((int(top_left_x), int(top_left_y)), (int(bottom_right_x), int(bottom_right_y))))

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


