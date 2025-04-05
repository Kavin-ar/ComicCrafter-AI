import os
import numpy as np
import cv2
from PIL import Image, ImageFont
from typing import List, Dict, Any, Optional, Tuple
import textwrap
import traceback
from PIL import ImageDraw


class ImprovedPanelLayout:
    """
    Handles the layout and creation of the final comic page image from individual panels,
    using OpenCV for drawing dialogues and bubbles (at fixed positions).
    """

    def __init__(self, output_dir: str = "./outputs"):
        # Layout parameters
        self.grid_rows = 2
        self.grid_cols = 3
        self.panel_width = 1024
        self.panel_height = 768
        self.padding = 25
        self.title_height = 100
        self.background_color = (255, 255, 255) # White (BGR for OpenCV later)
        self.border_color = (0, 0, 0) # Black
        self.border_width = 3
        self.number_background_color = (255, 255, 255, 190) # Keep for PIL number bg
        self.number_text_color = (0, 0, 0) # Keep for PIL number text

        # --- Font loading (Use PIL for title, OpenCV has limited built-in fonts) ---
        self.font_search_paths = [".", "/usr/share/fonts/truetype/msttcorefonts/", "/usr/share/fonts/truetype/dejavu/", "/Library/Fonts/", "C:/Windows/Fonts/"]
        self.title_font = self._load_pil_font(["arialbd.ttf", "DejaVuSans-Bold.ttf"], 60) # PIL font for title
        self.panel_num_font = self._load_pil_font(["arial.ttf", "DejaVuSans.ttf"], 36) # PIL font for panel numbers

        # --- OpenCV Font Settings ---
        self.cv_font_face = cv2.FONT_HERSHEY_SIMPLEX
        self.cv_font_scale = 0.8 # Adjust as needed
        self.cv_font_thickness = 2
        self.cv_text_color = (0, 0, 0) # Black (BGR)
        self.cv_bubble_color = (255, 255, 255) # White (BGR)
        self.cv_bubble_outline_color = (0, 0, 0) # Black (BGR)
        self.cv_bubble_padding = 10

        self._calculate_dimensions()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        if self.title_font is None: print("CRITICAL WARNING: Could not load any PIL title font!")
        if self.panel_num_font is None: print("CRITICAL WARNING: Could not load any PIL panel number font!")

    def _find_font_file(self, font_filenames: List[str]) -> Optional[str]:
        for filename in font_filenames:
            if os.path.exists(filename): return filename
            for path_dir in self.font_search_paths:
                 full_path = os.path.join(path_dir, filename)
                 if os.path.exists(full_path): print(f"Font found: {full_path}"); return full_path
        print(f"Font files not found: {font_filenames}"); return None

    def _load_pil_font(self, font_filenames: List[str], font_size: int) -> Optional[ImageFont.FreeTypeFont]:
        font_path = self._find_font_file(font_filenames)
        if font_path:
            try: return ImageFont.truetype(font_path, font_size)
            except Exception as e: print(f"Warning: Error loading PIL font '{font_path}'. Err: {e}")
        print(f"Warning: Could not load PIL fonts: {font_filenames}. Trying default.");
        try: return ImageFont.load_default() # Basic fallback
        except Exception as e_def: print(f"CRITICAL ERROR: Could not load default PIL font. Error: {e_def}"); return None

    def _calculate_dimensions(self):
        self.comic_width = self.grid_cols * (self.panel_width + self.padding) + self.padding
        total_panel_height = self.grid_rows * (self.panel_height + self.padding) + self.padding
        self.comic_height = total_panel_height + self.title_height
        print(f"Calculated comic dimensions: {self.comic_width}x{self.comic_height}")

    def update_layout(self, grid_rows: int, grid_cols: int, panel_width: int, panel_height: int):
         print(f"Updating layout: R={grid_rows}, C={grid_cols}, Panel={panel_width}x{panel_height}")
         self.grid_rows = max(1, grid_rows); self.grid_cols = max(1, grid_cols); self.panel_width = panel_width; self.panel_height = panel_height;
         self.padding = max(5, self.padding); self.title_height = max(30, self.title_height); self._calculate_dimensions()

    def _draw_dialogue_bubble_cv(self, image: np.ndarray, dialogue: str, panel_x: int, panel_y: int):
        """Draws dialogue bubble and text using OpenCV at a FIXED position."""
        if not dialogue or not isinstance(dialogue, str): return # Skip if no dialogue

        # --- Configuration for fixed position (Top Center) ---
        bubble_max_width_ratio = 0.7 # Max width relative to panel width
        target_y_offset = 30 # Fixed distance from top panel border
        line_height_factor = 1.5 # Multiplier for line spacing based on font size

        # --- Text Wrapping (Manual for OpenCV) ---
        max_bubble_width_px = int(self.panel_width * bubble_max_width_ratio)
        wrapped_lines = []
        words = dialogue.split(' ')
        current_line = ""
        for word in words:
            test_line = f"{current_line} {word}".strip()
            (text_width, text_height), _ = cv2.getTextSize(test_line, self.cv_font_face, self.cv_font_scale, self.cv_font_thickness)
            if text_width <= max_bubble_width_px:
                current_line = test_line
            else:
                # Word doesn't fit, push current line and start new one
                if current_line: wrapped_lines.append(current_line)
                # Handle very long words that might exceed max width alone
                if cv2.getTextSize(word, self.cv_font_face, self.cv_font_scale, self.cv_font_thickness)[0][0] > max_bubble_width_px:
                     # Simple truncation for very long words (can be improved)
                     wrapped_lines.append(word[:int(len(word)*0.8)] + "...")
                     current_line = ""
                else:
                     current_line = word
        if current_line: wrapped_lines.append(current_line) # Add last line

        if not wrapped_lines: return # No lines to draw

        # --- Calculate Bubble Dimensions ---
        max_line_width = 0
        total_text_height = 0
        (base_width, base_height), baseline = cv2.getTextSize("Tg", self.cv_font_face, self.cv_font_scale, self.cv_font_thickness) # Approx height
        line_spacing = int(base_height * line_height_factor)

        for line in wrapped_lines:
            (w, h), _ = cv2.getTextSize(line, self.cv_font_face, self.cv_font_scale, self.cv_font_thickness)
            max_line_width = max(max_line_width, w)
        total_text_height = len(wrapped_lines) * line_spacing - (line_spacing - base_height) # Adjust for last line

        bubble_width = max_line_width + 2 * self.cv_bubble_padding
        bubble_height = total_text_height + 2 * self.cv_bubble_padding

        # --- Calculate Bubble Position (Fixed Top-Center) ---
        bubble_x = (self.panel_width - bubble_width) // 2
        bubble_y = target_y_offset
        bubble_x += panel_x # Adjust relative to panel origin
        bubble_y += panel_y # Adjust relative to panel origin

        # Bubble tail (simple triangle pointing down from middle-bottom) - Fixed position
        tail_base_x = bubble_x + bubble_width // 2
        tail_base_y = bubble_y + bubble_height
        tail_tip_x = tail_base_x
        tail_tip_y = tail_base_y + 20 # Length of tail
        tail_width = 15 # Width of tail base

        # --- Draw Bubble Background & Outline ---
        cv2.rectangle(image, (bubble_x, bubble_y), (bubble_x + bubble_width, bubble_y + bubble_height), self.cv_bubble_color, -1) # Filled background
        cv2.rectangle(image, (bubble_x, bubble_y), (bubble_x + bubble_width, bubble_y + bubble_height), self.cv_bubble_outline_color, self.cv_font_thickness) # Outline

        # --- Draw Bubble Tail ---
        pts = np.array([[tail_base_x - tail_width // 2, tail_base_y],
                        [tail_base_x + tail_width // 2, tail_base_y],
                        [tail_tip_x, tail_tip_y]], np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.drawContours(image, [pts], 0, self.cv_bubble_color, -1) # Fill tail
        cv2.line(image, (tail_base_x - tail_width//2, tail_base_y), (tail_tip_x, tail_tip_y), self.cv_bubble_outline_color, self.cv_font_thickness)
        cv2.line(image, (tail_base_x + tail_width//2, tail_base_y), (tail_tip_x, tail_tip_y), self.cv_bubble_outline_color, self.cv_font_thickness)


        # --- Draw Text Lines ---
        current_y = bubble_y + self.cv_bubble_padding + base_height # Start y for first line (approx baseline)
        for line in wrapped_lines:
            # Center each line horizontally within the bubble
            (w, h), _ = cv2.getTextSize(line, self.cv_font_face, self.cv_font_scale, self.cv_font_thickness)
            line_x = bubble_x + (bubble_width - w) // 2
            cv2.putText(image, line, (line_x, current_y), self.cv_font_face, self.cv_font_scale, self.cv_text_color, self.cv_font_thickness, cv2.LINE_AA)
            current_y += line_spacing


    def create_comic_layout(self,
                           panel_images: List[np.ndarray],
                           dialogues: List[Optional[str]], # Add dialogues parameter
                           title: Optional[str] = "Comic Story") -> Optional[np.ndarray]:
        """
        Creates comic layout using PIL for base and title, then OpenCV for panels & dialogue bubbles.
        """
        if not panel_images: print("Error: No panel images provided."); return None
        if not all(isinstance(img, np.ndarray) for img in panel_images): print("Error: panel_images list contains non-numpy elements."); return None
        if len(panel_images) != len(dialogues): print(f"Warning: Mismatch between images ({len(panel_images)}) and dialogues ({len(dialogues)}). Some dialogues may be ignored.");

        ref_h, ref_w, ref_c = panel_images[0].shape
        if ref_c != 3: print(f"Error: Panel image channel count ({ref_c}) != 3."); return None
        if ref_h != self.panel_height or ref_w != self.panel_width:
             print(f"Warning: Panel img dims ({ref_w}x{ref_h}) != layout ({self.panel_width}x{self.panel_height}). Adjusting layout.");
             self.panel_width = ref_w; self.panel_height = ref_h; self._calculate_dimensions()

        # --- Create Base Canvas with PIL (Easier for Title Font Handling) ---
        try:
            comic_pil = Image.new('RGB', (self.comic_width, self.comic_height), color=self.background_color)
            draw_pil = ImageDraw.Draw(comic_pil)
        except Exception as e: print(f"Error creating PIL canvas: {e}"); return None

        # --- Add Title using PIL ---
        display_title = title or "Generated Comic"
        if self.title_font:
            try:
                 bbox = draw_pil.textbbox((0, 0), display_title, font=self.title_font)
                 tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
                 tx = (self.comic_width - tw) // 2; ty = max(5, (self.title_height - th) // 2)
                 draw_pil.text((tx, ty), display_title, font=self.title_font, fill=(0, 0, 0))
            except Exception as e_title: print(f"Error drawing PIL title: {e_title}. Skipping.")
        else: print("Warning: Title font not loaded. Skipping title.")

        # --- Convert PIL canvas to OpenCV format (NumPy array BGR) ---
        try:
            # Convert PIL RGB to NumPy array
            comic_cv = np.array(comic_pil)
            # Convert RGB to BGR for OpenCV functions
            comic_cv = cv2.cvtColor(comic_cv, cv2.COLOR_RGB2BGR)
        except Exception as e_conv:
             print(f"Error converting PIL canvas to OpenCV format: {e_conv}")
             return None

        # --- Add Panels and Dialogue using OpenCV ---
        num_panels_to_draw = min(len(panel_images), self.grid_rows * self.grid_cols)
        print(f"Layout grid: {self.grid_rows}x{self.grid_cols}. Drawing {num_panels_to_draw} panels with OpenCV.")

        for i in range(num_panels_to_draw):
            panel_array_rgb = panel_images[i] # Assume RGB from generator
            dialogue = dialogues[i] if i < len(dialogues) else None

            # Convert panel to BGR for OpenCV processing
            try:
                 panel_array_bgr = cv2.cvtColor(panel_array_rgb, cv2.COLOR_RGB2BGR)
            except Exception as e_panel_conv:
                 print(f"Error converting panel {i+1} to BGR: {e_panel_conv}. Skipping.")
                 continue

            row = i // self.grid_cols; col = i % self.grid_cols
            x = col * (self.panel_width + self.padding) + self.padding
            y = row * (self.panel_height + self.padding) + self.padding + self.title_height

            # Draw border (OpenCV rectangle)
            if self.border_width > 0:
                 cv2.rectangle(comic_cv, (x - self.border_width, y - self.border_width),
                               (x + self.panel_width + self.border_width, y + self.panel_height + self.border_width),
                               self.border_color, self.border_width)

            # Paste panel (roi assignment)
            try:
                 # Ensure panel fits exactly, resize if necessary (shouldn't be needed ideally)
                 if panel_array_bgr.shape[0] != self.panel_height or panel_array_bgr.shape[1] != self.panel_width:
                      print(f"Warning: Resizing panel {i+1} in OpenCV paste.")
                      panel_array_bgr = cv2.resize(panel_array_bgr, (self.panel_width, self.panel_height), interpolation=cv2.INTER_LANCZOS4)

                 comic_cv[y:y + self.panel_height, x:x + self.panel_width] = panel_array_bgr
            except Exception as paste_err:
                 print(f"Error pasting panel {i+1} with OpenCV: {paste_err}")
                 continue # Skip dialogue/number if paste fails

            # --- Draw Dialogue Bubble (using OpenCV) ---
            # IMPORTANT: This draws at a FIXED position due to lack of object detection
            try:
                 self._draw_dialogue_bubble_cv(comic_cv, dialogue, x, y)
            except Exception as e_bubble:
                 print(f"Error drawing CV dialogue bubble for panel {i+1}: {e_bubble}")


            # --- Add Panel Number (using PIL on the final image - easier font handling) ---

        # --- Convert final OpenCV BGR image back to RGB PIL Image for numbering & return ---
        try:
            final_comic_bgr = comic_cv
            final_comic_rgb = cv2.cvtColor(final_comic_bgr, cv2.COLOR_BGR2RGB)
            final_comic_pil = Image.fromarray(final_comic_rgb)
            draw_final = ImageDraw.Draw(final_comic_pil) # Draw object for final PIL image

            # --- Add Panel Numbers using PIL (easier fonts/background) ---
            for i in range(num_panels_to_draw):
                 if self.panel_num_font:
                      panel_num_text = str(i + 1)
                      row = i // self.grid_cols; col = i % self.grid_cols
                      x = col * (self.panel_width + self.padding) + self.padding
                      y = row * (self.panel_height + self.padding) + self.padding + self.title_height
                      num_padding = 8
                      try:
                           num_bbox = draw_final.textbbox((0,0), panel_num_text, font=self.panel_num_font)
                           num_w = num_bbox[2]-num_bbox[0]; num_h = num_bbox[3]-num_bbox[1]
                           rect_x0=x+num_padding//2; rect_y0=y+num_padding//2
                           rect_x1=rect_x0+num_w+num_padding; rect_y1=rect_y0+num_h+num_padding
                           # Need alpha channel for transparency - create temp layer
                           num_bg_layer = Image.new('RGBA', final_comic_pil.size, (0,0,0,0))
                           draw_num_bg = ImageDraw.Draw(num_bg_layer)
                           draw_num_bg.rectangle([(rect_x0,rect_y0),(rect_x1,rect_y1)], fill=self.number_background_color)
                           final_comic_pil.paste(num_bg_layer, (0,0), num_bg_layer) # Paste with alpha

                           # Redraw draw object on updated image
                           draw_final = ImageDraw.Draw(final_comic_pil)
                           text_x = rect_x0+num_padding//2; text_y = rect_y0+num_padding//2 - num_bbox[1] # Adjust y based on bbox top
                           draw_final.text((text_x, text_y), panel_num_text, font=self.panel_num_font, fill=self.number_text_color)
                      except Exception as e_num: print(f"Error drawing panel number {i+1} with PIL: {e_num}")
                 else: print(f"Skipping panel number {i+1} - PIL font failed.")

            # Convert final PIL image back to NumPy array for Gradio output
            final_comic_array = np.array(final_comic_pil)
            print("Comic layout created successfully using OpenCV for bubbles.")
            return final_comic_array
        except Exception as e:
            print(f"Error during final conversion or numbering: {e}")
            return None


    def save_comic(self, comic_array: np.ndarray, filename: str) -> Optional[str]:
        """ Saves the comic layout numpy array (RGB) to an image file."""
        if not isinstance(comic_array, np.ndarray): print("Error: Invalid data to save_comic."); return None
        if '.' not in filename: filename += ".png"
        file_path = os.path.join(self.output_dir, filename)
        try:
            # Assuming comic_array is RGB from the final conversion step
            comic_img = Image.fromarray(comic_array)
            comic_img.save(file_path)
            print(f"Comic layout saved to: {file_path}")
            return file_path
        except Exception as e: print(f"Error saving comic image to {file_path}: {e}\n{traceback.format_exc()}"); return None