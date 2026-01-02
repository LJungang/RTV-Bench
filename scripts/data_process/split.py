import json
import math
import os
import argparse
from pathlib import Path

def split_annotation(anno_path, args):
    num_splits = args.num_splits
    with open(anno_path) as f:
        data = json.load(f)
    
    total = len(data)
    per_split = math.ceil(total / num_splits)
    
    output_dir = Path(anno_path).parent / "splits"
    output_dir.mkdir(exist_ok=True)
    
    for i in range(num_splits):
        start = i * per_split
        end = min((i+1)*per_split, total)
        split_data = data[start:end]
        
        split_path = output_dir / f"split_{i+1}.json"
        with open(split_path, 'w') as f:
            json.dump(split_data, f, indent=2)
            
    print(f"generate {num_splits} subsets to {output_dir}.")

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--num_splits',required=True,default=4,type=int)
    args = parser.parse_args()
    split_annotation("./rtv-bench/QA.json",args)
