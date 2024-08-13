import os
from pathlib import Path

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = str(Path(ROOT_DIR) / "dataset")
OUTPUT_DIR = str(Path(ROOT_DIR) / "output")

MODELS = ["SantaCoder", "StarCoder", "WizardCoder", "Magicoder", "StarCoder21", "StarCoder22", "StarCoder23",
          "CodeLlama1", "CodeLlama2", "CodeLlama3", "CodeLlama4", "CodeLlama5", "CodeLlama",
          "DeepSeek1", "DeepSeek2", "DeepSeek", "DeepSeek4", "DeepSeek5", "DeepSeek6"]
DEEPSEEKS = ["DeepSeek1", "DeepSeek2", "DeepSeek", "DeepSeek4", "DeepSeek5", "DeepSeek6"]
DATASET = ["HumanEval", "Multi_HumanEval_java", "MBPP"]
DECODING_STRATEGY = ["GreedySearch", "BeamSearch", "NucleusSampling", "TopKSampling", "ContrastiveSearch",
                     "RepetitionPenalty"]

MODELS_SPECIAL_TOKENS = {
    "SantaCoder": [None, "<|endoftext|>"],
    "StarCoder": [None, "<|endoftext|>"],
    "WizardCoder": [None, "<|endoftext|>"],
    "Magicoder": ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"],
    "StarCoder2": [None, "<|endoftext|>"],
    "StarCoder2_15b": [None, "<|endoftext|>"],
    "StarCoder23": [None, "<|endoftext|>"],
    "CodeLlama1": ["<s>", "</s>"],
    "CodeLlama2": ["<s>", "</s>"],
    "CodeLlama3": ["<s>", "</s>"],
    "CodeLlama4": ["<s>", "</s>"],
    "CodeLlama5": ["<s>", "</s>"],
    "CodeLlama6": ["<s>", "</s>"],
    "DeepSeek1": ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"],
    "DeepSeek2": ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"],
    "DeepSeek3": ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"],
    "DeepSeek4": ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"],
    "DeepSeek5": ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"],
    "DeepSeek6": ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"]
}
