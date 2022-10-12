import torch
from pathlib import Path
from tqdm.auto import tqdm
import argparse

def average(models, weights=None):
        state_dicts = [model['model'] for model in models]
        with torch.no_grad():
            merged = {}
            for key in state_dicts[0]:
                merged[key] = torch.sum(torch.stack([sd[key] * weight for sd, weight in zip(state_dicts, weights)]), axis=0)
            return merged

def main(output_dir, weights):

    experts = list(Path('/checkpoint/suching/mod/_modular_gpt3_small_80K/modular_gpt3_small_80K_LR=0.0005').glob('MODEL*24000*0.0005'))
    print(experts)
    experts = [torch.load(e / 'checkpoint_last.pt') for e in tqdm(experts)]
    merged_expert = experts[0].copy()
    merged_expert['model'] = average(experts, weights=weights)
    torch.save(merged_expert, output_dir / 'checkpoint_last.pt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--weights", type=str)
    args = parser.parse_args()
    main(args.output_dir, [float(x) for x in args.weights.split(',')])