import os
import json
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from diffusers import AutoPipelineForText2Image, StableDiffusionXLPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import gc
import re
import math
import time
import traceback
from PIL import ImageDraw


# --- Constants (match main.py) ---
PROMPT_FAILURE_PLACEHOLDER_COLOR = (200, 150, 150)
GENERATOR_UNAVAILABLE_PLACEHOLDER_COLOR = (150, 150, 180)
NSFW_PLACEHOLDER_COLOR = (255, 150, 150)
GENERAL_ERROR_PLACEHOLDER_COLOR = (220, 200, 200)
OTHER_FAILURE_PLACEHOLDER_COLOR = (200, 200, 220)
LLM_LOAD_ERROR_PLACEHOLDER_COLOR = (180, 180, 220)
# --- End Constants ---

class ImprovedComicCrafterAI:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.default_settings = self.config.get("default_settings", {})
        self.prompt_templates = self.config.get("prompt_template", {})
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.llm_model_path = self.config.get('story_model_path', 'mistralai/Mistral-7B-Instruct-v0.2')
        self.image_model_path = self.config.get('image_model_path', 'stabilityai/stable-diffusion-xl-base-1.0')
        self.tokenizer = None
        self.llm_model = None
        self.image_generator = None
        self.is_llm_loaded = False
        self.is_image_generator_loaded = False
        self.story_characters = []
        if torch.cuda.is_available(): self._clear_gpu_memory()
        self.ensure_llm_loaded()
        if not self.is_llm_loaded: print("CRITICAL WARNING: LLM failed to load during initialization.")

    def _clear_gpu_memory(self):
        if self.device == "cuda": print("Clearing CUDA cache..."); gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); print("CUDA memory cleared.")

    def _unload_llm(self):
        if self.llm_model is not None: print("Unloading LLM..."); del self.llm_model; self.llm_model = None
        if self.tokenizer is not None: del self.tokenizer; self.tokenizer = None
        self.is_llm_loaded = False; print("LLM unloaded."); self._clear_gpu_memory()

    def _unload_image_generator(self):
        if self.image_generator is not None: print("Unloading Img Gen..."); del self.image_generator; self.image_generator = None
        self.is_image_generator_loaded = False; print("Img Gen unloaded."); self._clear_gpu_memory()

    def ensure_llm_loaded(self):
        if not self.is_llm_loaded and self.llm_model is not None: print("WARN: LLM flag inconsistent. Unloading."); self._unload_llm()
        if self.is_llm_loaded and self.llm_model is not None and self.tokenizer is not None: return True
        print("--- Ensuring LLM is loaded ---");
        if self.is_image_generator_loaded: print("Unloading image generator first..."); self._unload_image_generator()
        if self.is_llm_loaded or self.llm_model is not None or self.tokenizer is not None: print("ERROR: LLM state inconsistent."); self.is_llm_loaded = False; return False
        try:
            print(f"Loading LLM: {self.llm_model_path} (dtype: {self.compute_dtype})")
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_model_path);
            if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
            quant_cfg = None
            if self.device == "cuda": print("Applying 4-bit quantization..."); quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=self.compute_dtype, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            self.llm_model = AutoModelForCausalLM.from_pretrained(self.llm_model_path, torch_dtype=self.compute_dtype if quant_cfg is None else None, device_map="auto", quantization_config=quant_cfg, low_cpu_mem_usage=True)
            if self.llm_model.config.pad_token_id is None: self.llm_model.config.pad_token_id = self.tokenizer.eos_token_id
            self.is_llm_loaded = True; print("LLM loaded successfully."); self._clear_gpu_memory(); return True
        except Exception as e: print(f"FATAL ERROR loading LLM: {e}\n{traceback.format_exc()}"); self._unload_llm(); self.is_llm_loaded = False; return False

    def ensure_image_generator_loaded(self):
        if not self.is_image_generator_loaded and self.image_generator is not None: print("WARN: Img Gen flag inconsistent. Unloading."); self._unload_image_generator()
        if self.is_image_generator_loaded and self.image_generator is not None: return True
        print("--- Ensuring Image Generator is loaded ---")
        if self.is_llm_loaded: print("Unloading LLM first..."); self._unload_llm()
        if self.is_image_generator_loaded or self.image_generator is not None: print("ERROR: Img Gen state inconsistent."); self.is_image_generator_loaded = False; return False
        try:
            print(f"Loading Image Gen: {self.image_model_path} (dtype: {self.compute_dtype})")
            safety_args = {"safety_checker": None} if self.config.get("disable_safety_checker", True) else {}; print(f"Safety checker {'DIS' if safety_args else 'EN'}ABLED.")
            self.image_generator = StableDiffusionXLPipeline.from_pretrained(self.image_model_path, torch_dtype=self.compute_dtype, use_safetensors=True, variant="fp16" if self.compute_dtype == torch.float16 else None, **safety_args)
            if self.device == "cuda": print("Attempting VRAM optimizations..."); self.image_generator.enable_model_cpu_offload(); print("Enabled model CPU offload.") # Simplification for common case
            elif self.device == 'cpu': self.image_generator = self.image_generator.to(self.device)
            self.is_image_generator_loaded = True; print(f"Image generator loaded on {self.device}."); self._clear_gpu_memory(); return True
        except Exception as e: print(f"FATAL ERROR loading image generator: {e}\n{traceback.format_exc()}"); self._unload_image_generator(); self.is_image_generator_loaded = False; return False

    def _apply_mistral_instruct_format(self, prompt: str) -> str:
        return f"[INST] {prompt.strip()} [/INST]"

    def _generate_llm_output(self, prompt_text: str, max_new_tokens: int, temperature: float, top_k: int, top_p: float, retries: int = 2) -> Optional[str]:
        if not self.ensure_llm_loaded(): print("LLM load failed for generation."); return None
        formatted_prompt = self._apply_mistral_instruct_format(prompt_text); print(f"\n--- LLM Gen (MaxTok: {max_new_tokens}, T: {temperature}) ---")
        try: inputs = self.tokenizer(formatted_prompt, return_tensors="pt", max_length=2048, padding=True, truncation=True, return_attention_mask=True).to(self.device)
        except Exception as tok_err: print(f"FATAL TOKENIZATION ERROR: {tok_err}"); return None
        gen_text = None; last_err = None
        for attempt in range(retries):
            print(f"LLM Gen Attempt {attempt + 1}/{retries}...")
            try:
                with torch.no_grad(): outputs = self.llm_model.generate(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=max_new_tokens, num_return_sequences=1, temperature=temperature, do_sample=True, top_k=top_k, top_p=top_p, pad_token_id=self.tokenizer.eos_token_id, no_repeat_ngram_size=3)
                gen_ids = outputs[0][inputs.input_ids.shape[-1]:]; cur_out = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                cur_out = re.sub(r'^(Okay.*?dialogue:|Here.*?dialogue:|Dialogue:|Answer:|\[/INST\])\s*', '', cur_out, flags=re.I).strip('"`')
                if cur_out and len(cur_out.split()) >= 1 and "sorry" not in cur_out.lower() and "cannot fulfill" not in cur_out.lower(): gen_text = cur_out; print(f"Valid output: '{gen_text[:50]}...'"); break
                else: print(f"Warn: Attempt {attempt+1} invalid output ('{cur_out}')."); last_err = f"Invalid output: {cur_out}"
            except Exception as e: last_err = e; print(f"ERROR LLM Gen (Attempt {attempt+1}): {e}"); self._clear_gpu_memory(); time.sleep(1.5 if attempt < retries-1 else 0)
            if gen_text: break
        if gen_text: print("--- LLM Gen OK ---")
        else: print(f"--- LLM Gen FAILED ({retries} attempts, Last Err: {last_err}) ---")
        return gen_text

    def generate_image(self,
                       prompt: str,
                       num_inference_steps: int,
                       guidance_scale: float,
                       image_width: int,
                       image_height: int,
                       art_style: Optional[str] = None
                      ) -> Optional[np.ndarray]:
        ph_shape = (image_height, image_width, 3)
        if not self.ensure_image_generator_loaded(): print("Img Gen load failed."); return np.full(ph_shape, GENERATOR_UNAVAILABLE_PLACEHOLDER_COLOR, dtype=np.uint8)
        is_failed_prompt = isinstance(prompt, str) and prompt.startswith("F_") # Adjusted check
        if is_failed_prompt:
             print(f"Placeholder for failed prompt: {prompt}"); ph_color = LLM_LOAD_ERROR_PLACEHOLDER_COLOR if "LLM Load Error" in prompt else PROMPT_FAILURE_PLACEHOLDER_COLOR; return np.full(ph_shape, ph_color, dtype=np.uint8)
        if not isinstance(prompt, str) or not prompt or len(prompt.split()) < 3: print(f"Invalid/short prompt ('{prompt[:50]}...'). Placeholder."); return np.full(ph_shape, GENERAL_ERROR_PLACEHOLDER_COLOR, dtype=np.uint8)

        # Use selected art style or default if none provided
        style_suffix = art_style if art_style else ", vibrant comic book art style illustration, detailed lines, dynamic composition, cinematic lighting"
        enhanced_prompt = prompt + style_suffix

        print(f"\n--- Gen Img (Style: '{style_suffix}'): '{prompt[:100]}...' ---"); start = time.time()
        try:
            neg_prompt = self.default_settings.get("negative_prompt", ""); seed = int(time.time() * 1000) % (2**32); generator = torch.Generator(device="cuda" if self.device=="cuda" else "cpu").manual_seed(seed)
            img_out = self.image_generator(prompt=enhanced_prompt, negative_prompt=neg_prompt, width=image_width, height=image_height, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale, output_type="np", generator=generator).images
            if img_out is None or len(img_out) == 0: print("ERROR: No images returned."); return np.full(ph_shape, GENERAL_ERROR_PLACEHOLDER_COLOR, dtype=np.uint8)
            img_arr = img_out[0]
            if isinstance(img_arr, np.ndarray):
                 if img_arr.dtype in [np.float32, np.float16]: img_arr = (np.clip(img_arr, 0.0, 1.0) * 255).astype(np.uint8)
                 elif img_arr.dtype != np.uint8: img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
                 if img_arr.shape != ph_shape: print(f"ERROR: Img shape mismatch ({img_arr.shape} vs {ph_shape})."); return np.full(ph_shape, GENERAL_ERROR_PLACEHOLDER_COLOR, dtype=np.uint8)
            else: print("ERROR: Output not numpy array."); return np.full(ph_shape, GENERAL_ERROR_PLACEHOLDER_COLOR, dtype=np.uint8)
            end = time.time(); print(f"Image generated OK ({end - start:.2f}s)."); return img_arr
        except Exception as e: print(f"Error during image gen: {e}\n{traceback.format_exc()}"); self._clear_gpu_memory(); return np.full(ph_shape, GENERAL_ERROR_PLACEHOLDER_COLOR, dtype=np.uint8)

    def _extract_characters(self, prompt: str) -> List[str]:
        print("\n--- Extracting Characters ---")
        if not self.ensure_llm_loaded(): return self._extract_characters_fallback(prompt)
        extract_prompt = f'From: "{prompt}", list primary character names/types (max 3). Rules: ONLY comma-separated names/types. If none, output NONE.'
        chars = []
        try:
            response = self._generate_llm_output(extract_prompt, 60, 0.2, 10, 0.9, 1)
            if response is None or response.upper() == "NONE" or len(response) < 2: chars = self._extract_characters_fallback(prompt)
            else:
                potentials = re.split(r'[,\n]+', response); ignore={"prompt","character","output","list",":","name","nan","none","here","are","the","based","on","story","moral","title","narrative","no","specific","spaces","not","specified","unnamed","n/a","answer","step","only","names","types","max","entries"}
                f_names=[p.title() for n in potentials if (p:=n.strip('.').strip()) and 2<=len(p)<=35 and p.lower() not in ignore and not re.search(r'\d',p) and (re.match(r'^[A-Za-z][A-Za-z\s\'\-]*[A-Za-z]?$',p) if len(p)>0 else False)]
                u_names = sorted(list(set(f_names))); g_terms = {'People','Man','Woman','Boy','Girl','Person'}
                chars = [c for c in u_names if c not in g_terms] if len(u_names)>1 else u_names
                if not chars and u_names: chars = u_names # Keep generics if only option
                if not chars: chars = self._extract_characters_fallback(prompt)
        except Exception as e: print(f"LLM Char Extract Err: {e}. Fallback."); chars = self._extract_characters_fallback(prompt)
        self.story_characters = chars[:3]; print(f"Final characters: {self.story_characters}"); return self.story_characters

    def _extract_characters_fallback(self, prompt: str) -> List[str]:
        print("Executing fallback char extract...");
        potential = re.findall(r'\b(?:[A-Z][a-z]+|[A-Z]{2,})\b', prompt); phrases = re.findall(r'\b(?:a|an|the)\s+([A-Z][a-z]+)\b', prompt, re.I); potential.extend(phrases)
        common={'City','Metropolis','Street','Building','Sky','World','Danger','Storm','Field','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Farm','Castle','Forest','Mountain','River','Ocean','Book','Portal','Based','Prompt','Write','Short','Story','Comic','Featuring','Must','Neutral','Unbiased','Avoid','Stereotypes','Include','Panel','Descriptions','Image','Instructions','Title','Generate','Concise','Moral','Theme','Narrative','Beginning','Middle','End','Only','Output','Words'}
        ignore={"prompt","character","output","list",":","name","nan","none","here","are","the","based","on","story","moral","title","narrative","no","specific","spaces","not","specified","unnamed","n/a","answer","step","only","names","types","max","entries"}
        potential=[p.title() for p in potential if p not in common and len(p)>1 and p.lower() not in ignore]
        final_chars = sorted(list(set(potential)), key=len, reverse=True)[:3] or ["Character"]; print(f"Fallback chars: {final_chars}"); return final_chars

    def generate_story_only(self, user_prompt: str) -> Optional[Dict[str, Any]]:
        if not self.ensure_llm_loaded(): return {"title": "[F: LLM Load]", "storyline": "[F: LLM Load]", "moral": "[F: LLM Load]", "image_prompts": [], "dialogues": [], "characters": []}
        res = {"title": "[F: Gen]", "storyline": "[F: Gen]", "moral": "[F: Gen]", "image_prompts": [], "dialogues": [], "characters": []}
        self.story_characters = self._extract_characters(user_prompt); char_str = ", ".join(self.story_characters) if self.story_characters else "character(s)"; res['characters'] = self.story_characters
        tmpl = self.prompt_templates.get("narrative_prompt_template"); narr_prompt = tmpl.format(user_prompt=user_prompt, character_str=char_str) if tmpl else None
        story = self._generate_llm_output(narr_prompt, 800, 0.7, 50, 0.9, 2) if narr_prompt else None
        if story: res["storyline"] = re.sub(r'^\s*(Narrative:|Storyline:|Story:)\s*','',story, flags=re.I).strip(); print("Narrative generated.")
        else: res["storyline"] = "[F: LLM Narr Err]" if narr_prompt else "[F: Narr Template Missing]"; return res
        tmpl_t = self.prompt_templates.get("title_prompt_template"); title_prompt = tmpl_t.format(story_narrative=res["storyline"][:1000]) if tmpl_t else None
        title = self._generate_llm_output(title_prompt, 20, 0.5, 30, 0.9, 1) if title_prompt else None
        if title: res["title"]=re.sub(r'^\s*(Title:)\s*','',title,flags=re.I).strip('"`') or "[F: Empty Title]"
        else: res["title"] = "[F: LLM Title Err]" if title_prompt else "[F: Title Template Missing]"
        tmpl_m = self.prompt_templates.get("moral_prompt_template"); moral_prompt = tmpl_m.format(story_narrative=res["storyline"][:1000]) if tmpl_m else None
        moral = self._generate_llm_output(moral_prompt, 60, 0.6, 40, 0.9, 1) if moral_prompt else None
        if moral: res["moral"]=re.sub(r'^\s*(Moral:)\s*','',moral,flags=re.I).strip('"`') or "[F: Empty Moral]"
        else: res["moral"] = "[F: LLM Moral Err]" if moral_prompt else "[F: Moral Template Missing]"
        print("\n--- Final Story ---"); print(json.dumps({k:(v[:100]+'...' if isinstance(v,str) and len(v)>100 else v) for k,v in res.items() if k not in ['image_prompts', 'dialogues']}, indent=2)); print("--- End Story ---\n")
        return res

    def _generate_single_panel_prompt(self, panel_num: int, num_panels: int, story_context: str, user_prompt: str, characters: List[str]) -> str:
        fail_prefix = f"F_PANEL_{panel_num}"
        tmpl = self.prompt_templates.get("panel_prompt_template")
        if not tmpl: return f"{fail_prefix}: Template missing."
        char_str = ", ".join(characters) if characters else "character(s)"; stage = "beginning"
        if num_panels > 1: stage = "end" if panel_num / num_panels > 0.75 else ("middle" if panel_num / num_panels > 0.3 else "beginning")
        try: p_text = tmpl.format(panel_num=panel_num, num_panels=num_panels, user_prompt=user_prompt, character_str=char_str, story_context=story_context, stage=stage)
        except KeyError as ke: return f"{fail_prefix}: Template key err '{ke}'"
        except Exception as e: return f"{fail_prefix}: Format err {e}"
        desc = self._generate_llm_output(p_text, 80, 0.6, 40, 0.9, 2)
        if desc is None: return f"{fail_prefix}: LLM call failed."
        patterns = [r'^\s*(Panel\s*\d+\s*Vis.*?Desc.*?:|Panel\s*\d+:|Desc.*?:|Scene:|Here.*?desc.*?:)\s*', r'^\s*\[INST\].*?\[/INST\]\s*', r'^\s*Okay, here.*?:?\s*', r'^\s*[\d.\-\*]+\s+(?=\w)', r'\n?Remember.*?stereotypes.*$', r'\n?---\s*$', r'\s*Feel free.*?']
        cleaned = desc; [cleaned := re.sub(p, '', cleaned, flags=re.I | re.S).strip() for p in patterns]; cleaned = cleaned.replace('**','').replace('*','').strip('"`[](){}')
        if not cleaned or len(cleaned.split()) < 4: print(f"Warn: Panel {panel_num} prompt short after clean: '{desc}' -> '{cleaned}'"); return f"{fail_prefix}: Invalid LLM output (short)."
        return cleaned

    def generate_dialogue_for_panel(self, panel_description: str, characters: List[str], story_context: str) -> Optional[str]:
        if not self.ensure_llm_loaded(): return "[Dial Gen F: LLM Load Err]"
        if panel_description.startswith("F_"): return None 
        tmpl = self.prompt_templates.get("dialogue_prompt_template")
        if not tmpl: return "[Dial Gen F: Template Missing]"
        char_str = ", ".join(characters) if characters else "the character(s)"
        try: dial_prompt_txt = tmpl.format(character_str=char_str, panel_description=panel_description, story_context=story_context)
        except KeyError as ke: return f"[Dial Gen F: Template Key Err '{ke}']"
        except Exception as e: return f"[Dial Gen F: Format Err {e}]"
        dialogue = self._generate_llm_output(dial_prompt_txt, 50, 0.65, 45, 0.9, 2)
        if dialogue is None: return "[Dial Gen F: LLM Call Failed]"
        if dialogue.strip().upper() == "NONE": return None
        dialogue = re.sub(r'^\s*(Dialogue:|Line:)\s*', '', dialogue, flags=re.I).strip('"`')
        if not dialogue or len(dialogue.split()) == 0: return None
        print(f"--- Generated Dialogue: '{dialogue}' ---"); return dialogue

    def populate_panel_prompts(self, story_info: Dict[str, Any], user_prompt: str, num_panels: int, progress_callback: Optional[Callable[[str], None]] = None) -> Optional[List[str]]:
        if not self.ensure_llm_loaded(): print("ERR: Cannot gen prompts - LLM load failed."); return [f"F_PANEL_{i+1}: LLM Load Error" for i in range(num_panels)]
        prompts = []; story = story_info.get("storyline",""); title = story_info.get("title", user_prompt[:50]+"...")
        self.story_characters = story_info.get('characters')
        if not self.story_characters: print("Warn: Chars missing, re-extracting."); self.story_characters = self._extract_characters(user_prompt + (" "+story[:300] if story else "")); story_info['characters'] = self.story_characters
        if not story or story.startswith("[F:"): print("ERR: Cannot gen prompts - Story missing/failed."); return [f"F_PANEL_{i+1}: Story Missing/Failed" for i in range(num_panels)]
        print(f"\n>>> Populating {num_panels} panel prompts."); ctx = f"Title: {title}. Story: {story}"; ctx_snip = ctx[:1500] + ("..." if len(ctx)>1500 else "")
        for i in range(num_panels):
            pn = i+1
            if progress_callback: progress_callback(desc=f"Gen prompt {pn}/{num_panels}...")
            print(f"--- Gen Prompt Panel {pn}/{num_panels} ---");
            desc = self._generate_single_panel_prompt(pn, num_panels, ctx_snip, user_prompt, self.story_characters); prompts.append(desc)
        print(f">>> Finished generating {len(prompts)} prompts."); story_info['image_prompts'] = prompts; return prompts

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        print(f"Loading config: {config_path}"); default={"story_model_path":"mistralai/Mistral-7B-Instruct-v0.2","image_model_path":"stabilityai/stable-diffusion-xl-base-1.0","disable_safety_checker":True,"default_settings":{"temperature":0.7,"num_inference_steps":30,"guidance_scale":7.0,"image_width":1024,"image_height":768,"negative_prompt":"..."},"prompt_template":{"narrative_prompt_template":"[DEF]","title_prompt_template":"[DEF]","moral_prompt_template":"[DEF]","panel_prompt_template":"[DEF]","dialogue_prompt_template":"[DEF]"}}
        default["default_settings"]["negative_prompt"] = "text, words, letters, signature, watermark, blurry, deformed, bad anatomy, disfigured, poorly drawn face, mutation, mutated, extra limb, ugly, poorly drawn hands, missing limb, blurry, floating limbs, disconnected limbs, malformed hands, out of focus, long neck, long body, duplicate, cropped, low quality, jpeg artifacts, multiple panels, frame, border, username, artist name, error, glitch, noise, tiling, stereotype, caricature, biased depiction, duplicate characters, multiple instances of same character, clones, multiple views, multiple views of the same character, grid, multiple images, multiple scenes"
        def merge(d, u): m=d.copy(); [m.update({k: merge(m[k],v)}) if isinstance(v,dict) and k in m and isinstance(m[k],dict) else m.update({k:v}) for k,v in u.items() if v is not None]; return m
        try:
            if Path(config_path).is_file():
                 with open(config_path,'r',encoding='utf-8') as f: user_cfg=json.load(f)
                 if isinstance(user_cfg, dict): print("Merging user config..."); merged_cfg=merge(default, user_cfg); [merged_cfg.setdefault(k,v) for k,v in default.items()]; [merged_cfg[sk].setdefault(k,v) for sk in ["default_settings","prompt_template"] for k,v in default[sk].items()]; print("Config loaded/merged."); return merged_cfg
                 else: print("Warn: Config invalid JSON dict. Defaults used.")
            else: print(f"Warn: Config not found '{config_path}'. Defaults used.")
        except Exception as e: print(f"Error loading config '{config_path}': {e}. Defaults used.")
        print("Returning default config."); return default