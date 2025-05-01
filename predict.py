# predict.py
from cog import BasePredictor, Input, Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

class Predictor(BasePredictor):
    def setup(self):
        # load base + LoRA adapters
        self.tokenizer = AutoTokenizer.from_pretrained(".", trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            ".", device_map="auto", attn_implementation="eager"
        )
        self.model = PeftModel.from_pretrained(base_model, ".")
        self.model.eval()

    def predict(self, prompt: str = Input(description="Full prompt")) -> str:
        # generate (prompt includes your <start_of_turn>… tokens)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=64, temperature=0.7)
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # strip off the prompt if you like
        return text.split(prompt, 1)[-1].strip()
