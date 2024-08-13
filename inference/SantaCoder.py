from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch

from BaseModel import BaseModel


class SantaCoder(BaseModel):
    def __init__(self, model_path="bigcode/santacoder",
                 device="cuda:3"):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.model_name = "SantaCoder"

    def setup_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device)
        self.setup_tokenizer()
        self.model.eval()

    def setup_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
