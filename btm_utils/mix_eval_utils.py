import os
import re
from argparse import ArgumentParser

NOT_MODEL_FOLDERS = ['demix', 'mod', 'slurm_logs', 'eval']

def add_args():
    parser = ArgumentParser()
    parser.add_argument('--regex-str', type=str)
    parser.add_argument('--id-str', type=str)
    parser.add_argument('--model-folder', type=str)
    parser.add_argument('--model-type', type=str)
    parser.add_argument('--checkpoint-ids', type=str)
    parser.add_argument('--experts-to-use', type=str, default='0,1,2,3,4,5,6,7')
    return parser.parse_args()


def main(
    regex_str, id_str, model_folder, model_type, checkpoint_ids, experts_to_use):
    models = []
    experts_to_use = [int(e) for e in experts_to_use.split(',')]
    str_name = regex_str.replace('.*', '').replace('.', '')
    eval_name = str_name
    checkpoint_ids = checkpoint_ids.split(',')
    all_runs = [f for f in os.listdir(model_folder) if f not in NOT_MODEL_FOLDERS]
    regex = re.compile(regex_str)
    selected_folders = sorted([folder for folder in all_runs if regex.match(folder)])
    num_gpus = len([f for f in os.listdir(model_folder) if re.compile('.*checkpoint_last.*').match(f)]) - 1
    result_folder = os.path.join(model_folder, 'evals', eval_name, str(target_domain_id))
    for i in experts_to_use:
        if model_type == 'demix':
            rank = i * num_gpus // 8
            models.append(f'{model_folder}/checkpoint_{checkpoint_ids[i]}-rank-{rank}.pt')
        elif model_type == 'mod':
            models.append(f'{model_folder}/{selected_folders[i]}/checkpoint_{checkpoint_ids[i]}.pt')
        models.append(generalsist_model)
    print(result_folder, ':'.join(models))


if __name__=='__main__':
    args = add_args()
    main(args.regex_str, args.id_str, args.model_folder, args.model_type, args.experts_to_use)

