import json

from decoding_fix.DeepSeek1 import DeepSeek1

if __name__ == '__main__':
    decoding_strategy = "NucleusSampling"
    params_id = 0
    with open(f"parameter/{decoding_strategy}.json", "r") as f:
        kwargs = json.load(f)[params_id]
    coder = DeepSeek1()
    coder.generate_code(decoding_strategy=decoding_strategy, params_id=params_id, **kwargs)
