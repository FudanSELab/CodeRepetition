import jsonlines
import torch
from tqdm import tqdm

from definitions import DATA_DIR, OUTPUT_DIR


class BaseModel:

    def __init__(self):
        self.model_path = None
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.max_input_len = None
        self.device = None

    def load_model(self, model_path):
        # 实现模型加载的代码，不同子类可以根据自己的模型类型和加载方式进行重写
        raise NotImplementedError("load_model() must be implemented in the subclass.")

    def generate(self, input_data, **kwargs):
        inputs = self.tokenizer.encode(input_data + '\n', return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(inputs, max_new_tokens=512, **kwargs)
        output_code = self.tokenizer.decode(outputs[0])

        del inputs
        del outputs

        torch.cuda.set_device(self.device)
        torch.cuda.empty_cache()
        return output_code

    def generate_code(self, decoding_strategy, params_id, **kwargs):
        with jsonlines.open(f"{DATA_DIR}/decoding_fix/formatted_dataset_{self.model_name}.jsonl", "r") as reader:
            with jsonlines.open(f"{OUTPUT_DIR}/{self.model_name}/{decoding_strategy}/{params_id}.jsonl", "w") as writer:
                for line in tqdm(reader):
                    input_data = line["input"]
                    output_code = self.generate(input_data, **kwargs)
                    line["output"] = output_code
                    if self.tokenizer is not None:
                        line["output_token_num"] = len(self.tokenizer.encode(output_code))
                    line["decoding_strategy"] = decoding_strategy
                    line["parameter"] = kwargs
                    writer.write(line)
