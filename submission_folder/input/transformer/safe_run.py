import argparse
import logging

from subprocess import *
from concurrent.futures import ThreadPoolExecutor as Pool
import os
import sys
import resource
import psutil
from random import *
import time
import math
import json

from os import system
from glob import glob

SUCCESS, TLE, MLE, RTE, RUNNING = 0,1,2,3,-1
exit_names = ["SUCCESS", "TLE", "MLE", "RTE", "RUNNING"]

start_time = time.time()


MEMORY_LIMIT = 4*4096 * 0.95 # MB
TIME_LIMIT   = 12*60*60 - 60 * 15 # Seconds

class Process:
    def __init__(self, cmd, timeout, maxmemory, id):
        self.fout = open('store/tmp/%d.out'%id,'w')
        self.ferr = open('store/tmp/%d.err'%id,'w')
        sys.stdout.flush()
        self.cmd = cmd
        self.process = Popen(cmd.split(), stdout=self.fout, stderr=self.ferr, shell=False)
        self.pid = self.process.pid
        logging.info(f'{cmd}, {self.process.pid}')
        self.mp = psutil.Process(self.pid)
        self.memused, self.timeused = 0, 0
        self.start_time = time.time()
        self.timeout = timeout
        self.maxmemory = maxmemory

    def update(self):
        self.memory = self.mp.memory_info().rss/2**20
        self.memused = max(self.memused, self.memory)
        self.timeused = time.time()-self.start_time
        if self.memory > self.maxmemory:
            return (MLE, self.timeused, self.memused)
        if self.timeused > self.timeout:
            return (TLE, self.timeused, self.memused)
        if not self.memory:
            if self.process.wait():
                return (RTE, self.timeused, self.memused)
            else:
                return (SUCCESS, self.timeused, self.memused)
        return (RUNNING, self.timeused, self.memused)

    def __del__(self):
        self.fout.close()
        self.ferr.close()


class Command:
    def __init__(self, cmd, expected_time = TIME_LIMIT, expected_memory = MEMORY_LIMIT, slack = 1.5):
        self.cmd = cmd
        self.time = expected_time
        self.mem = expected_memory
        self.slack = slack

    def __lt__(self, other):
        return self.time < other.time


def runAll(cmd_list, threads, budget_factor):
    THREAD_LIMIT = threads

    ret_stats = {}

    dt = 0.4
    running = []
    cmdi = 0
    assert budget_factor > 0
    per_job_budget = TIME_LIMIT / budget_factor * threads

    def callback(process, status, timeused, memused):
        logging.info(f'{exit_names[status]}, {process.cmd}, {" %.5fs"%timeused}, {"%.5fMB"%memused}')
        sys.stdout.flush()
        if status == RTE:
            logging.warning(f'status != RTE!! WARNING! {process.cmd}')

        ret_stats[process.cmd] = (status, timeused, memused)

    while len(running) or cmdi < len(cmd_list):
        while cmdi < len(cmd_list) and len(running) < THREAD_LIMIT:
            cmd = cmd_list[cmdi]
            process = Process(f'{cmd.cmd} --time-budget {per_job_budget}', cmd.time*cmd.slack, cmd.mem*cmd.slack, cmdi)
            running.append(process)
            cmdi += 1

        torem = []
        mems = []
        for r in running:
            status, timeused, memused = r.update()
            mems.append(r.memory)
            if status != RUNNING:
                callback(r, status, timeused, memused)
                torem.append(r)

        if len(torem):
            for r in torem:
                running.remove(r)
        elif sum(mems) > MEMORY_LIMIT:
            r = running[mems.index(max(mems))]
            r.process.kill()
            callback(r, MLE, r.timeused, r.memused)
            running.remove(r)

        time.sleep(dt)

        current_time = time.time()
        elapsed_time = current_time - start_time
        per_job_budget = (TIME_LIMIT - elapsed_time) * threads / max(budget_factor - cmdi, 1)
        if elapsed_time >= TIME_LIMIT:
            logging.info(f"\nTIME LIMIT REACHED after {elapsed_time:.2f} seconds")
            logging.info(f"Currently running {len(running)} processes")
            for r in running:
                r.process.kill()
                callback(r, TLE, r.timeused, r.memused)
            break

    logging.info(f"Execution Summary: (cmdi, {cmdi}), (threads, {threads})")
    logging.info(f"Total tasks: {len(cmd_list)}")
    logging.info(f"Completed tasks: {len(ret_stats)}")
    logging.info(f"Success: {sum(1 for status, _, _ in ret_stats.values() if status == SUCCESS)}")
    logging.info(f"TLE: {sum(1 for status, _, _ in ret_stats.values() if status == TLE)}")
    logging.info(f"MLE: {sum(1 for status, _, _ in ret_stats.values() if status == MLE)}")
    logging.info(f"RTE: {sum(1 for status, _, _ in ret_stats.values() if status == RTE)}")
    return ret_stats


def merge_json_files(directory, output_file, context):
    merged_data = {}

    # Iterate over all JSON files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            
            # Read and update merged_data with the contents of each file
            with open(file_path, 'r') as file:
                merged_data.update(json.load(file))

    # Write the merged data to the output file
    with open(output_file, 'w') as outfile:
        json.dump(merged_data, outfile, indent=4)

    logging.info(f"Merged JSON data written to {output_file} for {context}")


def main():
    parser = argparse.ArgumentParser(description="Run ARC tasks with a specified model.")
    parser.add_argument("--checkpoint-path", type=str, default="../../input/transformer_model/Transformer_best.pt",
                        help="Path to the model checkpoint")
    parser.add_argument("--maximum-task-count", type=int, default=2000,
                        help="Maximum number of tasks")
    parser.add_argument("--source", type=str, default='arc-agi_evaluation',
                        help="the json file to be processed")
    parser.add_argument("--process-count", type=int, default=1,
                        help="Number of parallel processes")
    args = parser.parse_args()

    # Remove existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
    log_filename = f'transformer_{time.strftime("%Y%m%d_%H%M%S")}.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - transformer - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )

    logging.info(f'args{args}')

    system("mkdir -p output")
    system("mkdir -p submission")
    system("mkdir -p store/tmp")

    input_path = '../../input/arc-prize-2024/'
    test_path = os.path.join(input_path, f'{args.source}_challenges.json')

    # ...............................................................................
    with open(test_path,'r') as f:
        data = json.load(f)
        tasks_name = sorted(data.keys(), key=lambda x: len(json.dumps(data[x])), reverse=False)

    task_list = []
    start_index = 0 # 154 + 83 + 40
    for i in range(start_index, min(start_index + args.maximum_task_count, len(tasks_name))):
        task_list.append(Command(f"python3 -m src.adaptive_submission --input-path {input_path} --checkpoint-path {args.checkpoint_path} --source {args.source} --task-id {tasks_name[i]}"))

    logging.info(f'task_list len {len(task_list)}')

    runAll(task_list, args.process_count, len(tasks_name))

    merge_json_files('submission', 'submission_candidates.json', 'transformer')

        
if __name__ == "__main__":
    main()
