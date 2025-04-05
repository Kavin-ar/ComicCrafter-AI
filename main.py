# main.py
import os
import json
import numpy as np
import gradio as gr
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import time
import uuid
from PIL import Image
import traceback
import re
from PIL import ImageDraw

from comic_crafter_ai import ImprovedComicCrafterAI
from panel_layout import ImprovedPanelLayout

# --- Constants (Ensure these match comic_crafter_ai.py) ---
PROMPT_FAILURE_PLACEHOLDER_COLOR = (200, 150, 150)
GENERATOR_UNAVAILABLE_PLACEHOLDER_COLOR = (150, 150, 180)
NSFW_PLACEHOLDER_COLOR = (255, 150, 150)
GENERAL_ERROR_PLACEHOLDER_COLOR = (220, 200, 200)
OTHER_FAILURE_PLACEHOLDER_COLOR = (200, 200, 220)
LLM_LOAD_ERROR_PLACEHOLDER_COLOR = (180, 180, 220)
# --- End Constants ---

OUTPUT_DIR = Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

config_path = "config.json"
comic_crafter = None
panel_layout = None
initialization_errors = []

# --- Art Styles ---
# Define available art styles (key: display name, value: prompt suffix)
ART_STYLES = {
    "Comic Book (Default)": ", vibrant comic book art style illustration, detailed lines, dynamic composition, cinematic lighting",
    "Anime / Manga": ", anime art style, vibrant colors, manga illustration",
    "Fantasy Art": ", digital fantasy art, epic composition, detailed painting, cinematic lighting",
    "Sci-Fi Concept Art": ", sci-fi concept art, futuristic design, detailed digital painting",
    "Pixel Art": ", pixel art style, 16-bit, retro game art",
    "Minimalist Line Art": ", minimalist line art, simple, clean lines, black and white",
    "Watercolor": ", watercolor painting style, soft edges, blended colors",
    "No Specific Style": "" # Allows using only the panel description
}

# --- Initialization ---
try:
    print("Initializing ComicCrafterAI..."); start_init = time.time()
    comic_crafter = ImprovedComicCrafterAI(config_path); end_init = time.time()
    print(f"ComicCrafterAI initialized ({end_init - start_init:.2f}s).")
    if comic_crafter is None or not comic_crafter.is_llm_loaded: initialization_errors.append("WARN: LLM failed initial load.")
except Exception as e: error_msg = f"FATAL ERROR initializing ComicCrafterAI: {e}\n{traceback.format_exc()}"; print(error_msg); initialization_errors.append(error_msg); comic_crafter = None
try:
    print("Initializing PanelLayout (with OpenCV)...")
    panel_layout = ImprovedPanelLayout(output_dir=str(OUTPUT_DIR)) # Now uses OpenCV internally
    print("PanelLayout initialized.")
except ImportError: initialization_errors.append("ERROR: Failed to import OpenCV (cv2). Panel Layout disabled. Install with 'pip install opencv-python'"); panel_layout = None
except Exception as e: error_msg = f"ERROR initializing PanelLayout: {e}"; print(error_msg); initialization_errors.append(error_msg); panel_layout = None


# --- Placeholder Check (Same as before) ---
def is_placeholder_image(img_array: Optional[np.ndarray]) -> Tuple[bool, str]:
    if not isinstance(img_array, np.ndarray) or img_array.ndim != 3 or img_array.shape[2] != 3:
      return False, "Invalid data"
    try:
        h, w, _ = img_array.shape; points = [(h // 2, w // 2), (h // 4, w // 4), (3 * h // 4, 3 * w // 4)]
        first_clr = None; all_match = True
        for r, c in points:
          clr = tuple(img_array[r, c, :]);
          if first_clr is None: first_clr = clr
          elif clr != first_clr: all_match = False; break
        if not all_match or first_clr is None: return False, "Not solid color"
        ph_colors = {tuple(PROMPT_FAILURE_PLACEHOLDER_COLOR), tuple(GENERATOR_UNAVAILABLE_PLACEHOLDER_COLOR), tuple(NSFW_PLACEHOLDER_COLOR), tuple(GENERAL_ERROR_PLACEHOLDER_COLOR), tuple(OTHER_FAILURE_PLACEHOLDER_COLOR), tuple(LLM_LOAD_ERROR_PLACEHOLDER_COLOR)}
        if first_clr == tuple(PROMPT_FAILURE_PLACEHOLDER_COLOR): return True, "Prompt Failed"
        if first_clr == tuple(GENERATOR_UNAVAILABLE_PLACEHOLDER_COLOR): return True, "Generator Unavailable"
        if first_clr == tuple(NSFW_PLACEHOLDER_COLOR): return True, "NSFW Detected"
        if first_clr == tuple(GENERAL_ERROR_PLACEHOLDER_COLOR): return True, "General Error"
        if first_clr == tuple(OTHER_FAILURE_PLACEHOLDER_COLOR): return True, "Other Failure"
        if first_clr == tuple(LLM_LOAD_ERROR_PLACEHOLDER_COLOR): return True, "LLM Load Error"
        return False, "Solid color (unknown)"
    except Exception: return False, "Check error"


# --- Main Generation Function (Adjusted to return PIL Image) ---
def generate_comic(
    prompt: str,
    art_style_key: str,
    num_panels: int = 6,
    panel_width: int = 1024, panel_height: int = 768,
    num_steps: int = 30, guidance_scale: float = 7.0,
    create_layout: bool = True,
    progress=gr.Progress(track_tqdm=True)
) -> Tuple[Optional[Image.Image], Optional[str], str]: # Return type changed to Optional[Image.Image]
    if comic_crafter is None: return None, None, "FATAL ERROR: Comic Crafter AI not initialized."
    if not comic_crafter.is_llm_loaded:
      print("LLM not loaded. Reloading..."); comic_crafter.ensure_llm_loaded();
      if not comic_crafter.is_llm_loaded: return None, None, "ERROR: LLM failed to load."
    layout_available = panel_layout is not None
    if create_layout and not layout_available: print("Warn: Panel Layout unavailable."); create_layout = False

    # Get the art style suffix from the selected key
    selected_art_style = ART_STYLES.get(art_style_key, ART_STYLES["Comic Book (Default)"])
    print(f"Selected Art Style Suffix: '{selected_art_style}'")

    progress(0, desc="Initializing..."); start_time = time.time(); unique_id = uuid.uuid4().hex[:8]; timestamp = time.strftime("%Y%m%d-%H%M%S"); output_prefix = f"{timestamp}_{unique_id}"
    # Initialize comic_output_image as None
    comic_output_image: Optional[Image.Image] = None
    story_path, story_display_text = None, "Generation started..."
    # Progress stages adjusted for dialogue
    p_story_end=0.20; p_prompt_start=p_story_end; p_prompt_end=0.50; p_dialogue_start=p_prompt_end; p_dialogue_end=0.65; p_image_start=p_dialogue_end; p_image_end=0.90; p_save_start=p_image_end; p_layout_start=0.95; p_finish=1.0
    try:
        progress(0.05, desc="Generating story elements..."); story_info = comic_crafter.generate_story_only(prompt)
        if story_info is None or story_info.get("storyline", "").startswith("[F:"):
          error_msg = story_info.get("storyline", "[Story Gen F: Unknown]") if story_info else "[Story Gen F: None]";
          print(f"Crit Fail: Story failed: {error_msg}");
          story_display_text = f"## Story Gen Failed\n**Reason:** {error_msg}";
          if comic_crafter: comic_crafter._clear_gpu_memory(); return None, None, story_display_text
        progress(p_story_end, desc="Story elements generated."); story_info['dialogues'] = []
        progress(p_prompt_start, desc="Generating panel prompts...")
        def panel_prog_cb(desc=""):
          prog=p_prompt_start; m=re.search(r'(\d+)/(\d+)',desc);
          if m:
            try:
              c,t=int(m.group(1)),int(m.group(2)); prog=p_prompt_start+(p_prompt_end-p_prompt_start)*(c/t) if t>0 else p_prompt_start;
            except ValueError: pass;
          progress(max(p_prompt_start, min(prog,p_prompt_end)), desc=desc)
        image_prompts = comic_crafter.populate_panel_prompts(story_info, prompt, num_panels, progress_callback=panel_prog_cb)
        if image_prompts is None: print("Crit Err: populate_panel_prompts returned None."); image_prompts = [f"F_PANEL_{i+1}: Crit LLM Err" for i in range(num_panels)]
        story_info['image_prompts'] = image_prompts; num_prompt_failed = sum(1 for p in image_prompts if isinstance(p,str) and p.startswith("F_")); print(f"Prompt gen finished. Total: {len(image_prompts)}, Failed: {num_prompt_failed}"); progress(p_prompt_end, desc=f"Prompts finished ({num_prompt_failed} failed).")
        progress(p_dialogue_start, desc="Generating dialogue...")
        dialogues = []
        if num_panels > 0:
             story_ctx = f"Title: {story_info.get('title','')}. Story: {story_info.get('storyline','')}"; ctx_snip = story_ctx[:1500] + ("..." if len(story_ctx)>1500 else "")
             for i in range(num_panels):
                 cur_prompt = image_prompts[i]; prog = p_dialogue_start + (p_dialogue_end-p_dialogue_start)*((i+1)/num_panels); progress(min(prog, p_dialogue_end), desc=f"Gen dialogue {i+1}/{num_panels}...")
                 dialogue = None;
                 if isinstance(cur_prompt, str) and not cur_prompt.startswith("F_"): dialogue = comic_crafter.generate_dialogue_for_panel(cur_prompt, story_info.get('characters',[]), ctx_snip)
                 else: print(f"Skip dialogue for failed panel {i+1}")
                 dialogues.append(dialogue)
        story_info['dialogues'] = dialogues; num_dial_failed = sum(1 for d in dialogues if isinstance(d,str) and d.startswith("[Dial Gen F")); num_no_dial = sum(1 for d in dialogues if d is None); print(f"Dialogue gen finished. Fails: {num_dial_failed}, None: {num_no_dial}"); progress(p_dialogue_end, desc="Dialogue finished.")

        # --- MODIFIED: Simplified Story Display Text ---
        story_display_text = f"# {story_info.get('title', '[Title N/A]')}\n\n"
        for k, t in [("storyline", "Storyline"), ("moral", "Moral")]:
            cont = story_info.get(k)
            disp = f"[{t} N/A]"
            if cont and isinstance(cont, str):
                disp = cont if not cont.startswith("[F:") else f"**{cont}**"
            story_display_text += f"## {t}\n{disp}\n\n"

        progress(p_image_start, desc="Starting image generation...")
        panel_images: List[Optional[np.ndarray]] = []
        num_prompts = len(image_prompts)
        if num_prompts > 0:
             for i in range(num_prompts):
                 panel_prompt = image_prompts[i]; prog = p_image_start + (p_image_end - p_image_start)*((i+1)/num_prompts); desc = f"Gen image {i+1}/{num_prompts}"
                 if isinstance(panel_prompt, str) and panel_prompt.startswith("F_"): desc += " (Placeholder)";
                 progress(min(prog, p_image_end), desc=desc)
                 # Pass the selected art style to the image generator
                 panel_img_arr = comic_crafter.generate_image(
                     panel_prompt, num_steps, guidance_scale, panel_width, panel_height, art_style=selected_art_style
                 );
                 panel_images.append(panel_img_arr)
        else: print("No prompts for image gen.")
        num_imgs = len(panel_images); num_phs = sum(1 for img in panel_images if is_placeholder_image(img)[0]); print(f"Image gen finished. Imgs: {num_imgs}, Placeholders: {num_phs}"); progress(p_image_end, desc="Images finished. Saving...")

        progress(p_save_start, desc="Saving story details..."); story_fname = f"{output_prefix}_story_details.md"; story_path_obj = OUTPUT_DIR / story_fname
        # Save the *simplified* story details to the markdown file as well
        try:
          with open(story_path_obj, 'w', encoding='utf-8') as f:
            f.write(story_display_text); story_path = str(story_path_obj);
            print(f"Story saved: {story_path}")
        except Exception as e:
          print(f"Error saving story MD: {e}"); story_path = None

        comic_path = None; valid_layout_images = []; corresponding_dialogues = []
        for idx, img in enumerate(panel_images):
             if img is not None and not is_placeholder_image(img)[0]: valid_layout_images.append(img); corresponding_dialogues.append(story_info['dialogues'][idx] if idx < len(story_info['dialogues']) else None)
        if create_layout and layout_available and valid_layout_images:
             progress(p_layout_start, desc="Creating layout image..."); num_valid = len(valid_layout_images); print(f"Layout with {num_valid} valid images.")
             if num_valid <= 0: rows, cols = 0, 0
             elif num_valid == 1: rows, cols = 1, 1
             elif num_valid == 2: rows, cols = 1, 2
             elif num_valid == 3: rows, cols = 1, 3
             elif num_valid == 4: rows, cols = 2, 2
             elif num_valid <= 6: rows, cols = 2, (num_valid+1)//2
             elif num_valid <= 9: rows, cols = 3, (num_valid+2)//3
             else: cols=4; rows=(num_valid+3)//4
             if rows > 0 and cols > 0:
                 actual_h, actual_w, _ = valid_layout_images[0].shape; panel_layout.update_layout(rows, cols, actual_w, actual_h);
                 layout_title = story_info.get('title', 'Generated Comic');
                 if layout_title.startswith('['): layout_title = 'Generated Comic'
                 # Pass dialogues to OpenCV layout function
                 comic_array = panel_layout.create_comic_layout(panel_images=valid_layout_images, dialogues=corresponding_dialogues, title=layout_title)
                 if comic_array is not None:
                    # --- CHANGE: Load image from array into PIL instead of saving/reloading ---
                    try:
                        comic_output_image = Image.fromarray(comic_array)
                        print("Comic layout generated as PIL Image.")
                        # Optional: Still save a copy to disk
                        try:
                            comic_fname = f"{output_prefix}_comic_layout.png"
                            save_path = panel_layout.save_comic(comic_array, comic_fname)
                            if save_path: print(f"Layout also saved to: {save_path}")
                            else: print("Failed to save layout copy.")
                        except Exception as save_err: print(f"Error saving layout copy: {save_err}")
                    except Exception as pil_err:
                        print(f"Error converting layout array to PIL Image: {pil_err}")
                        comic_output_image = None
                 else: print("Failed create layout array.")
             else: print("Skip layout: No valid images or zero rows/cols.")
        elif create_layout: print(f"Skip layout: {'Component unavailable' if not layout_available else 'No valid images'}.")
        progress(p_finish, desc="Finished!"); end_time = time.time(); print(f"Total time: {end_time - start_time:.2f}s")
        print("Run finished. Cleanup...")
        if comic_crafter: comic_crafter._clear_gpu_memory()
        # --- CHANGE: Return the PIL image object ---
        return comic_output_image, story_path, story_display_text
    except Exception as e:
      error_message = f"FATAL Error during comic gen: {e}\n\n{traceback.format_exc()}"; print(error_message); print("Error occurred. Cleanup...");
      if comic_crafter: comic_crafter._clear_gpu_memory(); return None, None, f"## Gen Failed\n\n**Error:**\n```\n{error_message}\n```"

# --- UI Function (Updated Image component type) ---
def create_ui():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky), title="Comic Crafter AI") as interface:
        gr.Markdown("# Comic Crafter AI\nGenerate stories & visuals!")

        gr.Markdown(
            """
            **How to Use:**
            1.  **Enter Your Story Idea:** Type a short concept or theme for your comic in the text box (e.g., "A cat detective solves the mystery of the missing yarn", "Two robots explore a jungle planet").
            2.  **Choose an Art Style:** Select a visual style for the comic panels from the dropdown.
            3.  **(Optional) Adjust Advanced Settings:** Expand the section below to change the number of panels, image size, quality steps, or how strictly the images follow the text prompts.
            4.  **Generate:** Click the "Generate Comic" button. Generation can take several minutes depending on settings and hardware.
            5.  **View Results:**
                * The final comic page layout (if enabled) will appear in the "Comic Page Layout" tab.
                * The generated story (Title, Storyline, Moral) will be in the "Story & Details" tab.
                * Download the story details file from the "Downloads" tab. The comic image can be downloaded directly from its display.

            **Example Prompts:**
            * A superhero squirrel saves the city park from a litterbug monster.
            * A lonely astronaut finds a friendly alien on Mars.
            * Medieval knights discover a time-traveling phone booth.
            * A group of vegetables starts a rock band.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt_input = gr.Textbox(label="Enter Your Story Idea", placeholder="e.g., A brave knight confronts a friendly dragon.", lines=4)

                # Art Style Dropdown
                art_style_input = gr.Dropdown(
                    label="Select Art Style",
                    choices=list(ART_STYLES.keys()),
                    value="Comic Book (Default)", # Default selection
                    info="Choose the visual style for the comic panels."
                )

                cfg_defaults = comic_crafter.default_settings if comic_crafter else {}
                default_panels=6; default_width=cfg_defaults.get("image_width", 1024); default_height=cfg_defaults.get("image_height", 768); default_steps=cfg_defaults.get("num_inference_steps", 30); default_guidance=cfg_defaults.get("guidance_scale", 7.0)

                with gr.Accordion("Advanced Settings", open=False):
                     gr.Markdown("Adjust generation parameters (optional):")
                     with gr.Row():
                         num_panels = gr.Slider(label="Number of Panels", minimum=1, maximum=9, value=default_panels, step=1, info="How many distinct images/panels for the comic?")
                     with gr.Row():
                         panel_width = gr.Slider(label="Panel Width (px)", minimum=512, maximum=1024, value=default_width, step=64, info="Width of each generated image panel.")
                         panel_height = gr.Slider(label="Panel Height (px)", minimum=512, maximum=1024, value=default_height, step=64, info="Height of each generated image panel.")
                     with gr.Row():
                         num_steps = gr.Slider(label="Image Quality Steps", minimum=20, maximum=50, value=default_steps, step=1, info="More steps = better image quality, but slower generation.")
                         guidance_scale = gr.Slider(label="Prompt Adherence (Guidance)", minimum=1.0, maximum=12.0, value=default_guidance, step=0.5, info="Higher value = image follows prompt more strictly.")
                     create_layout = gr.Checkbox(label="Generate Final Comic Page Image", value=True, info="Combine panels into a single layout image (using OpenCV)?")

                generate_btn = gr.Button("Generate Comic", variant="primary")

            with gr.Column(scale=3):
                with gr.Tabs():
                    with gr.TabItem("Comic Page Layout"):
                        # --- CHANGE: Set type to "pil" ---
                        comic_output = gr.Image(label="Generated Comic Page", type="pil", show_label=True, show_download_button=True, interactive=False, height=600)
                    with gr.TabItem("Story & Details"):
                        # Output only Title, Storyline, Moral
                        story_output = gr.Markdown(label="Generated Story Details")
                    with gr.TabItem("Downloads"):
                        gr.Markdown("Download generated files:")
                        story_path_output = gr.File(label="Download Story Details (.md)", interactive=False)
                        # Comic download is available directly from the Image component

        generate_btn.click(
             fn=generate_comic,
             inputs=[
                 prompt_input,
                 art_style_input, 
                 num_panels,
                 panel_width,
                 panel_height,
                 num_steps,
                 guidance_scale,
                 create_layout
             ],
             outputs=[
                 comic_output, # Mapped to the first return value (PIL Image)
                 story_path_output, # Mapped to the second return value (story file path)
                 story_output # Mapped to the third return value (story markdown text)
             ],
             api_name="generate_comic"
        )
        gr.Markdown("--- \n *Powered by AI. Results vary. May take several minutes.*")
    return interface

# --- Main Launch Block ---
if __name__ == "__main__":
    print("\n" + "="*50)
    if not initialization_errors: print("Initialization check passed. Launching Gradio UI..."); print("Allow time for interface availability."); print("="*50 + "\n"); ui = create_ui(); ui.launch(share=True, debug=False) # share=True for external access if needed
    else: print("ERROR: Cannot launch UI due to critical init failures:"); [print(f"- Err {i+1}: {' '.join(msg.splitlines()[:2])}...") for i, msg in enumerate(initialization_errors)]; print("\nCheck model paths, dependencies (incl. opencv-python), CUDA, VRAM. Review logs."); print("="*50 + "\n")