import os

import jsonlines
from tqdm import tqdm

from postprocess_fix.fix import fix_repeats_in_line

from definitions import DATA_DIR, OUTPUT_DIR, DEEPSEEKS, DECODING_STRATEGY, MODELS_SPECIAL_TOKENS

sp_tokens = MODELS_SPECIAL_TOKENS["DeepSeek"]


def fix_Greedy():
    for model in DEEPSEEKS:
        with jsonlines.open(f"{DATA_DIR}/formatted_dataset_{model}.jsonl", "r") as reader:
            with jsonlines.open(f"{OUTPUT_DIR}/fixed/formatted_dataset_{model}_fix.jsonl", "w") as writer:
                for line in tqdm(reader):
                    if not line["source"].startswith("Multi"):
                        line = fix_repeats_in_line(line, sp_tokens,
                                                   "Java" if line["source"].startswith("Multi") else "Python")
                    writer.write(line)


def fix_all():
    for model in DEEPSEEKS:
        for decode_method in DECODING_STRATEGY[1:]:
            for params_id in range(3):
                file_path = f"{OUTPUT_DIR}/{model}/{decode_method}/{params_id}.jsonl"
                if not os.path.exists(file_path):
                    break
                output_path = f"{OUTPUT_DIR}/decoding_fix/{model}/{decode_method}/{params_id}_fix.jsonl"
                with jsonlines.open(file_path, "r") as reader:
                    with jsonlines.open(output_path, "w") as writer:
                        for line in tqdm(reader):
                            if not line["source"].startswith("Multi"):
                                line = fix_repeats_in_line(line, sp_tokens,
                                                           "Java" if line["source"].startswith("Multi") else "Python")
                            writer.write(line)