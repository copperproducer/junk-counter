import pytesseract
import keyboard
import pyttsx3
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
import torch
import csv
import re
from fuzzywuzzy import process


# **Vertical Offset Factor**
# Adjust this value to move the slots up or down.
# Positive values move the slots down, negative values move them up.
vertical_offset_factor = 0.7  # Default is 0.0 (no vertical adjustment)

# **Slot Height Reduction Factor**
# Adjust this value to change the height of the slots as a percentage of the original height.
slot_height_reduction_factor = 0.6  # Default is 0.6 (60% of original height)
debug_mode = False
def take_screenshot_and_visualize_slots():
    """
    Capture the whole screen and visualize the slots.
    """
    # Capture the full screen
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)
    screenshot_bgr = cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)

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
    screenshot = pyautogui.screenshot()
    screenshot_np = np.array(screenshot)  # Convert the PIL Image to a numpy array
    return cv2.cvtColor(screenshot_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

def load_ducat_values():
    try:
        return pd.read_csv('ducat_values.csv', index_col='Part Name')
    except FileNotFoundError:
        return pd.DataFrame(columns=['Part Name', 'Value']).set_index('Part Name')


def preprocess_image(image):
    # Convert the image to the LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Define the LAB range
    lower_bound = np.array([170, 105, 135])
    upper_bound = np.array([190, 152, 190])

    # Create a mask for pixels within the specified LAB range
    mask = cv2.inRange(lab, lower_bound, upper_bound)

    # Calculate the percentage of non-zero pixels in the mask
    total_pixels = mask.size
    non_zero_pixels = np.count_nonzero(mask)
    percentage_non_zero = (non_zero_pixels / total_pixels) * 100

    # Check if the percentage of non-zero pixels is greater than 1%
    has_pixels = percentage_non_zero > 1

    # Convert the mask to a black-and-white image
    _, black_and_white_image = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

    return black_and_white_image, has_pixels




def read_ducat_values():
    """Read ducat values from the CSV file."""
    try:
        with open('ducat_values.csv', mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            return {row['Part Name'].lower(): int(row['Value']) for row in reader}
    except FileNotFoundError:
        return {}

import re

def extract_and_format_part_name(part_name_image):
    """Use EasyOCR to extract and format the part name from an image."""
    if part_name_image is None or part_name_image.size == 0:
        return ""

    # Extract text using EasyOCR
    results = reader.readtext(part_name_image, paragraph=True)
    part_names = [text[1].lower() for text in results]
    # Join the recognized text into a single string

    formatted_text = ' '.join(part_names)

    # Remove numbers and special characters using regex
    formatted_text = re.sub(r'[^a-zA-Z\s]', '', formatted_text)

    # Optionally, remove specific words
    words_to_exclude = {'trade', 'read', 'ready', 'not'}  # Example words to exclude
    filtered_words = [word for word in formatted_text.split() if word not in words_to_exclude]

    # Rejoin the filtered words into the final text
    formatted_text = ' '.join(filtered_words)

    return formatted_text


def find_closest_match(part_name, known_parts):
    """Find the closest match for a part name in known parts using fuzzy matching."""
    if known_parts:
        # Use fuzzy matching to find the closest match with a similarity score
        result = process.extractOne(part_name, known_parts.keys(), score_cutoff=80)
        if result:
            best_match, score = result
            return best_match
    return None

def read_total_platinum(total_platinum):
    threading.Thread(target=_read_total_platinum_thread, args=(total_platinum,)).start()

def _read_total_platinum_thread(total_platinum):
    acquired = tts_lock.acquire(blocking=False)
    if acquired:
        try:
            engine = pyttsx3.init()
            engine.say(f"{total_platinum}")
            engine.runAndWait()
        finally:
            tts_lock.release()
    else:
        pass  # If the lock is already held, skip reading

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
        master.configure(bg='#333333')  # Set the dark theme background
        self.font_style = font.Font(family="Arial", size=8)
        self.default_prices = {100: 9, 65: 4, 45: 3, 25: 1, 15: 1}
        self.prices = self.default_prices.copy()
        self.load_settings()

        # Load slot positions from settings or set default values
        self.load_slot_positions()
        self.scanned_slots = set()

        # Price adjustment section
        self.price_labels = {}
        self.price_entries = {}

        # Create a frame to hold the price adjustments
        price_frame = tk.Frame(master, bg='#333333')
        price_frame.pack(pady=5)

        # Arrange the price adjustments in a grid, 3 columns per row
        prices = list(self.prices.items())
        for idx, (ducats, platinum) in enumerate(prices):
            row = idx // 3  # Integer division
            col = idx % 3   # Modulo operation
            frame = tk.Frame(price_frame, bg='#333333')
            frame.grid(row=row, column=col, padx=10, pady=5, sticky='w')

            label = tk.Label(frame, text=f"{ducats} Ducats:", font=self.font_style, bg='#333333', fg='white')
            label.pack(side=tk.LEFT)
            entry = tk.Entry(frame, width=5, bg="#444444", fg="white", insertbackground="white")
            entry.insert(0, str(platinum))
            entry.pack(side=tk.LEFT)
            self.price_labels[ducats] = label
            self.price_entries[ducats] = entry

        # Save and Restore Defaults buttons side by side
        buttons_frame = tk.Frame(master, bg='#333333')
        buttons_frame.pack(pady=5, padx=10, fill=tk.X)

        self.save_button = tk.Button(buttons_frame, text="Save Settings", command=self.save_settings,
                                     font=self.font_style, bg="#444444", fg="#FFFFFF")
        self.save_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.restore_button = tk.Button(buttons_frame, text="Restore Defaults", command=self.restore_defaults,
                                        font=self.font_style, bg="#444444", fg="#FFFFFF")
        self.restore_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Font and Button Styling
        self.font_style_button = font.Font(family="Arial", size=10, weight="bold")
        self.font_style_label = font.Font(family="Arial", size=8)
        self.font_style_platinum = font.Font(family="Arial", size=18, weight="bold")

        # Slot Position Settings buttons side by side
        slot_buttons_frame = tk.Frame(master, bg='#333333')
        slot_buttons_frame.pack(pady=5, padx=10, fill=tk.X)

        self.slot_settings_button = tk.Button(slot_buttons_frame, text="Adjust Slot Positions",
                                              command=self.adjust_slot_positions,
                                              font=self.font_style_button, bg="#444444", fg="#FFFFFF")
        self.slot_settings_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.visualize_slots_button = tk.Button(slot_buttons_frame, text="Visualize Slots",
                                                command=take_screenshot_and_visualize_slots,
                                                font=self.font_style_button, bg="#444444", fg="#FFFFFF")
        self.visualize_slots_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Scanned Slots Visual Indicators
        self.slot_indicators = []
        slots_frame = tk.Frame(master, bg='#333333')
        slots_frame.pack(pady=5)
        for i in range(6):
            label = tk.Label(slots_frame, text=f"Slot {i + 1}", bg="gray", fg='white', width=10, height=1)
            label.pack(side=tk.LEFT, padx=2)
            self.slot_indicators.append(label)

        # Scanned Slots Label
        self.scanned_slots_label = tk.Label(master, text="Scanned Slots: None", font=self.font_style_label,
                                            bg='#333333', fg='white')
        self.scanned_slots_label.pack(pady=5)

        # Activate and Reset buttons side by side
        action_buttons_frame = tk.Frame(master, bg='#333333')
        action_buttons_frame.pack(pady=5, padx=10, fill=tk.X)

        self.toggle_button = tk.Button(action_buttons_frame, text="Activate", command=self.toggle,
                                       font=self.font_style_button, bg="#444444", fg="#FFFFFF")
        self.toggle_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        self.reset_button = tk.Button(action_buttons_frame, text="Reset", command=self.reset_total,
                                      font=self.font_style_button, bg="#444444", fg="#FFFFFF")
        self.reset_button.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # Polling Rate Controls
        self.polling_rate_label = tk.Label(master, text="Polling Rate (checks per second):",
                                           font=self.font_style_label, bg='#333333', fg='white')
        self.polling_rate_label.pack()
        self.polling_rate = tk.Scale(master, from_=1, to=50, orient="horizontal", resolution=1,
                                     font=self.font_style_label, troughcolor="#555555", sliderlength=15, width=10,
                                     bg='#333333', fg='white')
        self.polling_rate.set(10)
        self.polling_rate.pack(fill=tk.X, padx=10, pady=2)

        # Platinum Counter
        self.total_platinum = 0
        self.total_platinum_label = tk.Label(master, text=f"Total Platinum: {self.total_platinum}",
                                             font=self.font_style_platinum, bg='#333333', fg='#FFD700')
        self.total_platinum_label.pack(pady=5)

        # Copy Trade Message Button
        self.clipboard_button = tk.Button(master, text="Copy Trade Message",
                                          command=self.copy_trade_message_to_clipboard,
                                          font=self.font_style_button, bg="#444444", fg="#FFFFFF")
        self.clipboard_button.pack(pady=5, padx=10, fill=tk.X)

        # Bind the right Ctrl key to the reset_total method
        keyboard.on_press_key("right ctrl", self.reset_total)

        self.active = False
        self.allow_automatic_scan = False

        if debug_mode:
            # Add a button to debug and display the processed image
            self.debug_button = tk.Button(master, text="Debug Image", command=self.debug_image_display,
                                          font=self.font_style_button, bg="#444444", fg="#FFFFFF")
            self.debug_button.pack(pady=5, padx=10, fill=tk.X)

    def toggle(self):
        self.active = not self.active
        self.toggle_button.config(text="Deactivate" if self.active else "Activate")
        self.allow_automatic_scan = self.active
        if self.active:
            self.periodic_scan()
        else:
            self.scanned_slots.clear()  # Optionally clear slots on deactivation

    def update_slot_indicator(self, index, scanned):
        """Update the visual indicator for the given slot index only if it hasn't been updated before."""
        current_color = self.slot_indicators[index].cget("bg")
        new_color = 'gold' if scanned else 'gray'
        if current_color != new_color:  # Check if an update is necessary
            self.slot_indicators[index].config(bg=new_color)

    def debug_image_display(self):
        """Function to display the debug image on the main thread."""
        image = capture_screen_to_cv2()  # Capture the image
        processed_image = preprocess_image(image)  # Preprocess the image

        # Schedule the imshow on the main thread
        self.master.after(0, lambda: self.show_debug_image(processed_image))

    def show_debug_image(self, image):
        """Show the debug image using OpenCV."""
        cv2.imshow('Debug Image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    def periodic_scan(self):
        if self.active:
            if self.allow_automatic_scan and len(self.scanned_slots) < 6:  # Only scan if slots are not fully scanned
                self.automatic_scan()
            else:
                # Deactivate automatic scanning if all slots are scanned
                self.allow_automatic_scan = False

            # Continue polling at the set rate
            self.master.after(1000 // self.polling_rate.get(), self.periodic_scan)

    def capture_slot_image(slot_index):
        """Capture the image of the specified slot only."""
        slot_positions = get_scaled_positions()
        # Get the position for the specified slot
        top_left, bottom_right = slot_positions[slot_index - 1]
        x1, y1 = top_left
        x2, y2 = bottom_right

        # Calculate the width and height of the slot
        width = x2 - x1
        height = y2 - y1

        # Capture only the slot region
        slot_image = pyautogui.screenshot(region=(x1, y1, width, height))
        slot_image_np = np.array(slot_image)  # Convert the PIL Image to a numpy array
        return cv2.cvtColor(slot_image_np, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

    def reset_total(self, event=None):
        self.total_platinum = 0
        self.scanned_slots.clear()
        for indicator in self.slot_indicators:
            indicator.config(bg='gray')
        self.update_total_platinum_display()
        self.allow_automatic_scan = True
        # No need to release the lock here

    def ducat_to_platinum(self, ducats):
        for ducat_value, platinum_value in sorted(self.prices.items(), key=lambda x: -x[0]):
            ducat_value = int(ducat_value)
            if ducats >= ducat_value:
                return platinum_value
        return 0

    def update_total_platinum_display(self):
        """Update the display of total platinum and scanned slots only if changes occurred."""
        previous_total = getattr(self, '_previous_total_platinum', None)
        if previous_total != self.total_platinum:  # Update only if the total platinum has changed
            self._previous_total_platinum = self.total_platinum
            self.total_platinum_label.config(text=f"Total Platinum: {self.total_platinum}")

        scanned_slots_text = ", ".join(map(str, sorted(self.scanned_slots))) if self.scanned_slots else "None"
        previous_scanned_slots = getattr(self, '_previous_scanned_slots_text', None)
        if previous_scanned_slots != scanned_slots_text:  # Update only if scanned slots have changed
            self._previous_scanned_slots_text = scanned_slots_text
            self.scanned_slots_label.config(text=f"Scanned Slots: {scanned_slots_text}")

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
                self.prices = {int(k): v for k, v in self.prices.items()}  # Convert keys to integers
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
            self.update_slot_positions(start_x, start_y, slot_width, slot_height, gap_between_slots)
            messagebox.showinfo("Info", "Slot positions updated!")

    def load_slot_positions(self):
        try:
            with open('slot_positions.json', 'r') as f:
                self.slot_positions = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.slot_positions = get_slot_positions()
            self.save_slot_positions()  # Save default positions

    def save_slot_positions(self):
        with open('slot_positions.json', 'w') as f:
            json.dump(self.slot_positions, f)

    def update_slot_positions(self, start_x, start_y, slot_width, slot_height, gap_between_slots):
        """
        Update the slot positions based on user input and save them to settings.
        """
        global vertical_offset_factor, slot_height_reduction_factor
        # You can adjust vertical_offset_factor and slot_height_reduction_factor here if needed

        # Calculate new positions for each slot based on the input values
        self.slot_positions = []
        for i in range(6):  # Assuming 6 slots
            top_left_x = start_x + i * (slot_width + gap_between_slots)

            # Apply vertical offset
            vertical_offset = int(slot_height * vertical_offset_factor)
            top_left_y = start_y + vertical_offset

            # Reduce the slot height
            adjusted_slot_height = int(slot_height * slot_height_reduction_factor)
            bottom_right_x = top_left_x + slot_width
            bottom_right_y = top_left_y + adjusted_slot_height

            self.slot_positions.append(((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)))
        self.save_slot_positions()

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
        trade_message = f"WTB Prime Junk | {' | '.join(message_parts)} | Full Trades Only"

        # Copy to clipboard
        pyperclip.copy(trade_message)

    def automatic_scan(self):
        """Preemptive scan using EasyOCR."""
        full_screen_image = capture_screen_to_cv2()
        known_parts = read_ducat_values()  # Load known part values

        for i, (top_left, bottom_right) in enumerate(get_scaled_positions(), start=1):
            if i not in self.scanned_slots:
                x1, y1 = top_left
                x2, y2 = bottom_right
                slot_image = full_screen_image[y1:y2, x1:x2]

                # Preprocess the image and check for relevant pixels
                processed_image, has_pixels = preprocess_image(slot_image)

                if not has_pixels:
                    continue  # Skip OCR if no relevant pixels are detected

                part_name_image = slot_image  # Use the entire slot image for scanning
                part_name_text = extract_and_format_part_name(part_name_image)
                closest_match = find_closest_match(part_name_text, known_parts)
                print(f"Slot {i}: {part_name_text} -> {closest_match}")

                if closest_match:
                    ducat_value = known_parts[closest_match]
                    self.update_platinum_and_slots(i, ducat_value)

    def update_platinum_and_slots(self, slot_index, ducat_value):
        if slot_index not in self.scanned_slots:
            self.scanned_slots.add(slot_index)
            self.update_slot_indicator(slot_index - 1, True)
            platinum_value = self.ducat_to_platinum(ducat_value)
            previous_total = self.total_platinum
            self.total_platinum += platinum_value

            # Only update if the total platinum has actually changed
            if self.total_platinum != previous_total:
                self.update_total_platinum_display()

            # Read the total platinum aloud only when all slots have been scanned
            if len(self.scanned_slots) == 6:  # Assuming there are 6 slots
                read_total_platinum(self.total_platinum)


def get_slot_positions():
    """
    Define adjusted slot positions at the base resolution (e.g., 1920x1080).
    """
    global vertical_offset_factor, slot_height_reduction_factor

    start_x = 218
    start_y = 675
    slot_width = 220
    slot_height = 220
    gap_between_slots = 34

    slot_positions = []
    for i in range(6):
        top_left_x = start_x + i * (slot_width + gap_between_slots)

        # Apply vertical offset
        vertical_offset = int(slot_height * vertical_offset_factor)
        top_left_y = start_y + vertical_offset

        # Reduce the slot height
        adjusted_slot_height = int(slot_height * slot_height_reduction_factor)
        bottom_right_x = top_left_x + slot_width
        bottom_right_y = top_left_y + adjusted_slot_height

        slot_positions.append(((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)))

    return slot_positions

def get_scaled_positions():
    """
    Scale slot positions based on the actual screen resolution.
    """
    screen_width, screen_height = pyautogui.size()
    base_resolution = (1920, 1080)
    scale_x = screen_width / base_resolution[0]
    scale_y = screen_height / base_resolution[1]
    slot_positions = get_slot_positions()
    scaled_positions = [
        ((int(x1 * scale_x), int(y1 * scale_y)),
         (int(x2 * scale_x), int(y2 * scale_y)))
        for (x1, y1), (x2, y2) in slot_positions
    ]
    return scaled_positions

if __name__ == "__main__":
    # Initialize global variables
    tts_lock = threading.Lock()

    # Check if CUDA is available
    print(f'CUDA available: {torch.cuda.is_available()}')

    # Initialize EasyOCR reader with English language support
    reader = easyocr.Reader(['en'])

    root = tk.Tk()
    app = PrimeJunkApp(root)
    root.mainloop()
