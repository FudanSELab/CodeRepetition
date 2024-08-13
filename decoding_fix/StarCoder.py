import torch

from decoding_fix.BaseModel import BaseModel

from transformers import AutoTokenizer, AutoModelForCausalLM


class StarCoder(BaseModel):
    def __init__(self, model_path="bigcode/starcoderbase", device="cuda:4",
                 access_token="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"):
        super().__init__()
        self.model_path = model_path
        self.device = device
        self.access_token = access_token
        self.load_model(model_path)
        self.model_name = "StarCoder"
        self.max_input_len = 8192

    def load_model(self, model_path):
        # 在这里实现加载模型的代码
        # 例如，如果使用TensorFlow或PyTorch模型，可以在这里加载模型权重和配置
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_auth_token=self.access_token)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            use_auth_token=self.access_token
        ).to(self.device)
        self.model.eval()
