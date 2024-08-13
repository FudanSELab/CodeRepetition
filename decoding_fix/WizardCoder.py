import sys

import torch

from decoding_fix.BaseModel import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM


class WizardCoder(BaseModel):

    def __init__(self, model_path="WizardLM/WizardCoder-15B-V1.0", device="cuda:3"):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.load_model(model_path)
        self.model_name = "WizardCoder"
        self.max_input_len = 2048

    def load_model(self, model_path):
        # 在这里实现加载模型的代码
        # 例如，如果使用TensorFlow或PyTorch模型，可以在这里加载模型权重和配置
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=False,
            torch_dtype=torch.float16
        )
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.half().to(self.device).eval()
        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)
