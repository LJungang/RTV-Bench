import argparse
import os
import json

def main(args):
    target_dir = args.target_dir
    results = []

    for item in os.listdir(target_dir):
        if item.endswith('.json')and item.startswith('result'):
            with open(os.path.join(target_dir, item), 'r') as f:
                data = json.load(f)
                results.extend(data)

    parent_dir = os.path.dirname(target_dir)
    
    output_file = os.path.join(parent_dir, args.save_file_name+'.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"The merged JSON file has been saved at: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_dir', default='', required=True)
    parser.add_argument('--save_file_name', default='', required=True)
    args = parser.parse_args()
    main(args)
