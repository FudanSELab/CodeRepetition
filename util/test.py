import json
import os
import subprocess
import tempfile

import jsonlines
from tqdm import tqdm

from definitions import DATA_DIR, OUTPUT_DIR

import threading

sp_tokens = ["<｜begin▁of▁sentence｜>", "<｜end▁of▁sentence｜>"]


class CodeExecutionThread(threading.Thread):
    def __init__(self, code):
        threading.Thread.__init__(self)
        self.code = code
        self.exception = None

    def run(self):
        try:
            exec(self.code)
        except Exception as e:
            self.exception = e


def test_HumanEval(line, dataset, timeout=0.5):
    prefix_len = 0
    if sp_tokens[0]:
        prefix_len = len(sp_tokens[0])
    code = line["output"][prefix_len:]
    # 去掉后缀
    suffix_start = code.find(sp_tokens[1])
    if suffix_start >= 0:
        code = code[:suffix_start]
    test = ""
    for task in dataset:
        if task["task_id"] == line["source"].split(":")[1]:
            test = task["test"] + f"\ncheck({task['entry_point']})"
            break

    test_code = code + test

    if test_code.find("while True") >= 0:
        return False
    # print(test_code)
    # print("11111111111111111")
    code_thread = CodeExecutionThread(test_code)
    code_thread.daemon = True  # 设置线程为守护线程

    code_thread.start()
    code_thread.join(timeout)

    if code_thread.is_alive():
        # 如果代码还在运行，说明超时了
        return False

    if code_thread.exception is not None:
        # 如果代码在执行过程中发生异常
        return False

    return True


def test_MBPP(line, dataset, timeout=0.5):
    prefix_len = 0
    if sp_tokens[0]:
        prefix_len = len(sp_tokens[0])
    code = line["output"][prefix_len:]
    # 去掉后缀
    suffix_start = code.find(sp_tokens[1])
    if suffix_start >= 0:
        code = code[:suffix_start]
    test = ""
    for task in dataset:
        if str(task["task_id"]) == line["source"].split(":")[1]:
            test = "\n\n" + "\n".join(task["test_list"])
            break
    test_code = code + test
    if test_code.find("power_base_sum(987654321987654321, 987654321987654321)") >= 0:
        return False
    # print(test_code)
    code_thread = CodeExecutionThread(test_code)
    code_thread.daemon = True  # 设置线程为守护线程

    code_thread.start()
    code_thread.join(timeout)

    if code_thread.is_alive():
        # 如果代码还在运行，说明超时了
        return False

    if code_thread.exception is not None:
        # 如果代码在执行过程中发生异常
        return False

    return True


def test_HumanEval_java(line, dataset, timeout=5):
    prefix_len = 0
    if sp_tokens[0]:
        prefix_len = len(sp_tokens[0])
    code = line["output"][prefix_len:]
    # 去掉后缀
    suffix_start = code.find(sp_tokens[1])
    if suffix_start >= 0:
        code = code[:suffix_start]
    test = ""
    for task in dataset:
        if task["task_id"] == line["source"].split(":")[1]:
            test = task["test"]
            break

    test_code = code + test
    # print(test_code)
    # 创建一个临时文件夹来保存Java文件
    with tempfile.TemporaryDirectory() as tmpdir:
        # 定义Java文件名
        java_file_name = "Main.java"
        java_file_path = os.path.join(tmpdir, java_file_name)

        # 将拼装好的代码写入Java文件
        with open(java_file_path, 'w') as file:
            file.write(test_code)

        try:
            # 编译Java文件
            compile_process = subprocess.run(['javac', java_file_path], check=True, stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE, timeout=timeout)
            # print("Compilation output:\n", compile_process.stdout.decode())

            # 运行编译后的Java程序
            run_process = subprocess.run(['java', '-cp', tmpdir, "Main"], check=True, stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE, timeout=timeout)
            # print("Execution output:\n", run_process.stdout.decode())

        except subprocess.CalledProcessError as e:
            # print("An error occurred:\n", e.stderr.decode())
            return False
        return True


def test_line(line, datasets):
    # print(line["source"])
    if line["source"].startswith("HumanEval"):
        return test_HumanEval(line, datasets[0])
    elif line["source"].startswith("MBPP"):
        return test_MBPP(line, datasets[1])
    else:
        return test_HumanEval_java(line, datasets[2])


if __name__ == '__main__':
    with open(f"{DATA_DIR}/HumanEval.json", "r") as f:
        dataset1 = json.load(f)
    with open(f"{DATA_DIR}/Multi_HumanEval_java.json", "r") as f:
        dataset2 = json.load(f)
    with open(f"{DATA_DIR}/MBPP.json", "r") as f:
        dataset3 = json.load(f)

    total = 0
    pass_num = 0
    file_path_old = f"{DATA_DIR}/formatted_dataset_DeepSeek2.jsonl"
    file_path_new = f"{OUTPUT_DIR}/DeepSeek1/BeamSearch/0.jsonl"
    length = 0
    with jsonlines.open(file_path_new, "r") as f:
        for line in tqdm(f):
            length += 1
            total += 1
            if test_line(line, [dataset1, dataset2, dataset3]):
                pass_num += 1

    print(f"pass_num: {pass_num}, total: {total}, acc: {format(pass_num / total * 100, '.1f')}%")
