import jsonlines
from tqdm import tqdm


class BaseModel:

    def __init__(self):
        self.model_path = None
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = None
        self.max_new_tokens = 512

    # 加载模型和tokenizer
    def setup_model(self):
        raise NotImplementedError("setup_model() must be implemented in the subclass.")

    # 加载tokenizer
    def setup_tokenizer(self):
        raise NotImplementedError("setup_tokenizer() must be implemented in the subclass.")

    # 推理
    def inference(self, dataset_path, output_path):
        if not self.model:
            self.setup_model()
        with jsonlines.open(dataset_path, 'r') as reader:
            ID = 0
            with jsonlines.open(output_path, 'w') as writer:
                for js in tqdm(reader):
                    source = js['source']
                    prompt = js['prompt']
                    inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                    tokens = self.model.generate(inputs, max_new_tokens=self.max_new_tokens)
                    output = self.tokenizer.decode(tokens[0])
                    writer.write({'ID': ID, 'source': source, 'prompt': prompt, 'predict': output})
                    ID += 1

    # 统计输入输出gt的token数
    def calculate_token_num(self, dataset_path, output_path):
        if not self.tokenizer:
            self.setup_tokenizer()
        with jsonlines.open(dataset_path, 'r') as reader:
            with jsonlines.open(output_path, 'w') as writer:
                for line in tqdm(reader):
                    if line["model"] == self.model_name:
                        input_tokens = self.tokenizer.encode(line["input"], return_tensors="pt").to(self.device)
                        output_tokens = self.tokenizer.encode(line["output"], return_tensors="pt").to(self.device)
                        if line["ground_truth"] is not None:
                            gt_tokens = self.tokenizer.encode(line["ground_truth"], return_tensors="pt").to(self.device)
                        repetition_tokens = self.tokenizer.encode(line["repeat_part"], return_tensors="pt").to(
                            self.device)
                        line["input_token_num"] = len(input_tokens[0])
                        line["output_token_num"] = len(output_tokens[0])
                        line["ground_truth_token_num"] = len(gt_tokens[0]) if line["ground_truth"] is not None else 0
                        line["repetition_token_num"] = len(repetition_tokens[0])
                        writer.write(line)
