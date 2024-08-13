from definitions import DATASET, MODELS
from inference.CodeLlama1 import CodeLlama1

if __name__ == '__main__':
    dataset = DATASET[0]
    model = MODELS[0]
    device = "cuda:1"
    coder = CodeLlama1(device=device)
    coder.inference(f"dataset/{dataset}).jsonl", f"output/all/{model}/GreedySearch/{dataset}_{model}.jsonl")
