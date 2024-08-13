import torch

from decoding_fix.BaseModel import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM


class StarCoder23(BaseModel):
    def __init__(self, model_path="/home/Data/models/starcoder2-15b-instruct-v0.1", device="cuda:4"):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.load_model(model_path)
        self.model_name = "StarCoder23"
        self.max_input_len = 8192

    def load_model(self, model_path):
        # 在这里实现加载模型的代码
        # 例如，如果使用TensorFlow或PyTorch模型，可以在这里加载模型权重和配置
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()
