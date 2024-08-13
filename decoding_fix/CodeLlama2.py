import torch

from decoding_fix.BaseModel import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM


class CodeLlama2(BaseModel):

    def __init__(self, model_path="/home/Data/models/CodeLlama-7b-Instruct-hf",
                 device="cuda:3"):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.load_model(model_path)
        self.model_name = "CodeLlama2"
        self.max_input_len = 2048

    def load_model(self, model_path):
        # 在这里实现加载模型的代码
        # 例如，如果使用TensorFlow或PyTorch模型，可以在这里加载模型权重和配置
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
