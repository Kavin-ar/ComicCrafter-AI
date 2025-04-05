**ComicCrafter AI - User Manual**

**1. Introduction**

Welcome to ComicCrafter AI! This tool uses Artificial Intelligence to
help you create short comic book stories from simple text prompts. You
provide an idea, and the AI generates a storyline, title, moral,
individual panel descriptions, dialogue, and visual panels, optionally
combining them into a final comic page layout.

**2. Features**

-   **Story Generation:** Creates a short narrative (200-400 words) with
    a beginning, middle, and end based on your prompt.

-   **Character Extraction:** Attempts to identify the main characters
    from your prompt to feature in the story.

-   **Title & Moral Generation:** Automatically suggests a concise title
    and a brief moral or theme for the generated story.

-   **Panel Prompt Generation:** Creates specific visual descriptions
    for each comic panel (up to 9 panels) to guide the image generation
    AI.

-   **Dialogue Generation:** Generates short lines of dialogue for
    characters within each panel, based on the visual description and
    story context.

-   **AI Image Generation:** Uses a text-to-image AI model (Stable
    Diffusion XL) to create images for each panel based on the generated
    descriptions and selected art style.

-   **Multiple Art Styles:** Allows you to choose from various art
    styles (e.g., Comic Book, Anime, Fantasy, Pixel Art) to influence
    the visuals.

-   **Comic Page Layout:** Can automatically arrange the generated
    panels (including dialogues and panel numbers) into a final comic
    page image using configurable grid layouts.

-   **User-Friendly Interface:** Provides a simple web interface (built
    with Gradio) for entering prompts, adjusting settings, and viewing
    results.

-   **Customizable Settings:** Offers advanced options to control the
    number of panels, image dimensions, image quality, and how closely
    the images match the prompts.

**3. Requirements**

*(Based on standard practices for such tools, as specific dependencies
aren\'t fully listed)*

-   **Python:** The application is written in Python.

-   **Libraries:** Needs libraries like torch (PyTorch), transformers,
    diffusers, gradio, numpy, Pillow (PIL), and potentially
    openCV-python (for the panel layout feature).

-   **Hardware:** A CUDA-enabled GPU (NVIDIA graphics card) is highly
    recommended, especially for faster image generation. It can run on
    CPU, but it will be significantly slower. Sufficient RAM and VRAM
    are needed depending on the models used.

-   **Models:** Requires access to pre-trained Language Models (LLM) and
    Text-to-Image models specified in the config.json file. These models
    will be downloaded automatically on first use if not present locally
    (can be large downloads).

-   **Fonts:** Needs access to font files (like Arial, DejaVu Sans) for
    rendering text (title, panel numbers). The tool searches common
    system font locations.

**4. Configuration**

This file controls the core behaviour and models used:

-   **story_model_path:** Specifies the pre-trained Language Model used
    for generating stories, titles, morals, panel descriptions, and
    dialogues (e.g., mistralai/Mistral-7B-Instruct-v0.2).

-   **image_model_path:** Specifies the pre-trained Text-to-Image model
    used for generating panel visuals (e.g.,
    stabilityai/stable-diffusion-xl-base-1.0).

-   **disable_safety_checker:** Option to disable the built-in safety
    checker of the image generation model.

-   **default_settings:** Contains default values for generation
    parameters:

    -   temperature, top_k, top_p: Control the creativity/randomness of
        the LLM text generation.

    -   num_inference_steps, guidance_scale: Control the quality and
        prompt adherence of the generated images.

    -   image_width, image_height: Default dimensions for generated
        panel images.

    -   negative_prompt: A list of terms to guide the image generator
        away from certain features (e.g., text, watermarks, bad
        anatomy).

-   **prompt_template:** Defines the specific instructions given to the
    LLM for each generation task (narrative, title, moral, panel
    description, dialogue). These templates guide the AI on the expected
    output format and content constraints (e.g., word limits, focus
    areas).

**5. Installation & Setup**

*(Standard Python project setup inferred)*

1.  **Clone/Download:** Get the project files (main.py,
    comic_crafter_ai.py, panel_layout.py, config.json).

2.  **Hugging Face Hub:** It is mandatory to Log in to the hugging face
    hub to use this software. You can do this by running this code in
    terminal or in a separate code cell in Google Colab:

> *!huggingface-cli login (in Google Colab)*
>
> *huggingface-cli login (in Terminal)*
>
> Then enter your hugging face access token in the token box.

3.  **Install Dependencies:** Open a terminal or command prompt in the
    project directory and install the required Python libraries using
    this command:

> *pip install torch transformers diffusers accelerate bitsandbytes
> Pillow numpy gradio opencv-python*
>
> If you are running this in google colab, add a code cell and then
> install the required Python libraries using this command:
>
> *!pip install torch transformers diffusers accelerate bitsandbytes
> Pillow numpy gradio opencv-python*

4.  **Check Configuration:** Review config.json to ensure model paths
    are correct and defaults are suitable.

5.  **Run:** Execute the main script from the terminal: python main.py.
    This will start the Gradio web server.

> If you are running this in google colab, run this line in a separate
> code cell:
>
> *!python main.py*

6.  **Access UI:** Open the local URL provided in the terminal in your web browser.

> If you are running this in google colab, use the public URL to use the
> software. The code gives options to run in local URL and as well as
> public URL.

**6. How to Use (Gradio Interface)**

1.  **Enter Story Idea:** In the main text box (\"Enter Your Story
    Idea\"), type your concept (e.g., \"A robot chef who dreams of
    opening a bakery\").

2.  **Select Art Style:** Choose a visual style from the dropdown menu
    (e.g., \"Anime / Manga\", \"Pixel Art\"). The default is \"Comic
    Book\".

3.  **Adjust Advanced Settings (Optional):**

    -   Expand the \"Advanced Settings\" section.

    -   Use the sliders to change:

        -   **Number of Panels:** How many images the story will be
            split into (1-9).

        -   **Panel Width/Height:** The size of each generated image.

        -   **Image Quality Steps:** More steps generally mean better
            quality but slower generation.

        -   **Prompt Adherence (Guidance):** How strictly the image
            should follow the text description.

    -   Check/uncheck \"Generate Final Comic Page Image\" to
        enable/disable the final layout creation.

4.  **Generate:** Click the \"Generate Comic\" button.

5.  **Wait:** Generation can take several minutes, depending on the
    number of panels, settings, and your hardware. Progress updates
    should appear above the input area.

6.  **View Results:**

    -   **Comic Page Layout Tab:** If layout generation was enabled and
        successful, the final comic page image will appear here. You can
        right-click to save it or use the download button.

    -   **Story & Details Tab:** Displays the generated Title,
        Storyline, and Moral.

    -   **Downloads Tab:** Provides a link to download a text file (.md)
        containing the generated story details.

**7. Output Files**

-   **Comic Layout Image:** (Optional) A single image file (e.g., .png)
    containing all generated panels arranged in a grid with title, panel
    numbers, and dialogue bubbles. Saved in the outputs directory.

-   **Story Details File:** A markdown file (e.g., \_story_details.md)
    containing the generated title, storyline, and moral. Saved in the
    outputs directory.

-   *(Individual panel images are generated but might not be saved
    separately by default unless modified)*.

**8. Troubleshooting / Known Issues**

-   **Slow Generation:** Especially on CPU or lower-end GPUs, image
    generation can be very time-consuming. Reduce the number of panels
    or image quality steps.

-   **Model Loading Errors:** Ensure correct model paths in config.json
    and that you have enough VRAM/RAM. Check internet connection if
    models need downloading. Initialization errors are shown on startup
    if models fail to load.

-   **Placeholder Images:** If panels appear as solid color blocks, it
    indicates a failure during prompt generation or image generation
    (e.g., LLM error, safety filter trigger, insufficient prompt
    details). Check the console output for specific error messages.

-   **Font Issues:** If titles or panel numbers look wrong or are
    missing, the required font files might not be found on your system.

-   **OpenCV Errors:** The panel layout feature requires openCV-python.
    If it\'s not installed, the layout step will be skipped. An error
    message should appear during initialization if OpenCV is missing or
    fails to load.

-   **Dialogue Placement:** Dialogues and bubbles are drawn at fixed
    positions (currently top-center) within each panel and might not
    always align perfectly with the characters or action in the
    generated image.

-   **Repetitive/Inconsistent Results:** AI generation involves
    randomness. Story details, prompts, or images might sometimes be
    repetitive, inconsistent, or not perfectly aligned with the initial
    prompt. Try re-running or adjusting the prompt/settings.

-   **Resource Usage:** Running both LLM and Image Generation models is
    resource-intensive. The application attempts to manage memory by
    loading only one type of model (LLM or Image Gen) at a time,
    unloading the other.
