import cProfile
import pstats
from pstats import SortKey
from torch.utils.data import DataLoader
from tqdm import tqdm
from line_profiler import LineProfiler
from src.prepare_data import load_dataset

dataset_file = "./intermediate_data/prepared_dataset.pth"

dataset, data_sources, _ = load_dataset(dataset_file)
print(f'dataset length: {len(dataset)}')

def process_data():
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    seq_lengths = []
    print('tqdm starts here')
    for index, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Processing batches"):
        seq_lengths.extend([len(seq) for seq in batch])
        if index > 10:
            break
    return seq_lengths

def run_profiling():
    # cProfile
    cProfile.run('process_data()', 'stats')

    # Print cProfile stats
    print("\ncProfile Statistics:")
    p = pstats.Stats('stats')
    p.sort_stats(SortKey.TIME).print_stats(10)

    # Line Profiler
    lp = LineProfiler()
    lp.add_function(process_data)
    lp_wrapper = lp(process_data)
    lp_wrapper()
    
    print("\nLine Profiler Statistics:")
    lp.print_stats()

if __name__ == "__main__":
    run_profiling()


