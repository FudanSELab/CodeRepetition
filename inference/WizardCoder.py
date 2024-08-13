import sys

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

from BaseModel import BaseModel


class WizardCoder(BaseModel):
    def __init__(self, model_path="WizardLM/WizardCoder-15B-V1.0",
                 device="cuda:3"):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.model_name = "WizardCoder"

    def setup_model(self):
        load_8bit = False
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16
        ).to(self.device)
        self.setup_tokenizer()
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        if not load_8bit:
            self.model.half()
        self.model.eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

    def setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
