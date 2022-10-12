from fairseq.moe_checkpoint_utils import merge_expert_and_shared_state
from argparse import ArgumentParser
import os
import re


def find_folders(CHECKPOINTS_TOP_FOLDER, re_string=None):
    NEW_MODEL_FOLDERS = []
    for name, folders, files in os.walk(CHECKPOINTS_TOP_FOLDER):
        regex = re.compile(re_string) if re_string else None
        if 'checkpoint_last.pt' not in files and 'checkpoint_last-shared.pt' not in files:
            continue #no last checkpoint found
        if regex and not regex.match(name):
            continue
        new_name = os.path.relpath(name, CHECKPOINTS_TOP_FOLDER)
        NEW_MODEL_FOLDERS.append(new_name)
    return NEW_MODEL_FOLDERS

def add_args():
    parser = ArgumentParser()
    parser.add_argument('--old-folder', type=str)
    parser.add_argument('--new-folder', type=str)
    parser.add_argument('--subfolder', type=str)
    parser.add_argument('--new-subfolder', type=str)
    parser.add_argument('--phase-one-ratio', type=float, default=None)
    parser.add_argument('--phase-one-update-num', type=int, default=None)
    parser.add_argument('--domain-id', type=str, default=None)
    return parser.parse_args()

def main(CHECKPOINTS_TOP_FOLDER, NEW_MODEL_TOP_FOLDER, subfolder, new_subfolder, domain_id, phase_one_ratio=None, phase_one_update_num=None):
    # only the master process should do the copying
    import shutil
    if (phase_one_ratio and phase_one_update_num) or (phase_one_ratio is None and phase_one_update_num is None):
        raise RuntimeError("Only one of --phase-one-ratio and --phase-one-update-num can be set")
    distributed_rank = int(os.environ['SLURM_PROCID'])
    copy = True
    if distributed_rank != 0:
        copy = False
    # create subfolder if none provided
    if not new_subfolder:
        new_subfolder = subfolder
    old_folder = os.path.join(CHECKPOINTS_TOP_FOLDER, subfolder)
    # find checkpoint to copy from
    if phase_one_update_num == -1:
        src_checkpoint_update_id = "checkpoint_last"
    else:
        files = [f for f in os.listdir(old_folder) if os.path.isfile(os.path.join(old_folder, f))]
        checkpoint_update_ids = list(set([f[:-3].split('-')[0] for f in files]))
        checkpoint_update_ids = [f for f in checkpoint_update_ids if f.count('_') == 2]
        update_nums = [int(f.split("_")[2]) for f in checkpoint_update_ids]
        sort_factor = 1
        if phase_one_ratio:
            max_update_num = max(update_nums)
            src_update_num = int(phase_one_ratio * max_update_num)
            if phase_one_ratio > 0.5:
                sort_factor = -1
            zipped_name_and_num = [(a, b) for (a, b) in zip(update_nums, checkpoint_update_ids)]
            zipped_name_and_num.sort(key=lambda i: sort_factor * i[0])
            src_checkpoint_update_id = zipped_name_and_num[min(range(len(zipped_name_and_num)), key=lambda i: abs(zipped_name_and_num[i][0]-src_update_num))][1]
        elif phase_one_update_num != -1:
            src_update_num = phase_one_update_num
            sort_factor = 1
            zipped_name_and_num = [(a, b) for (a, b) in zip(update_nums, checkpoint_update_ids)]
            zipped_name_and_num.sort(key=lambda i: sort_factor * i[0])
            src_checkpoint_update_id = zipped_name_and_num[min(range(len(zipped_name_and_num)), key=lambda i: abs(zipped_name_and_num[i][0]-src_update_num))][1]
    # copy checkpoint - dense model 
    if 'checkpoint_last.pt' in files: 
        new_domain_folder_path = os.path.join(NEW_MODEL_TOP_FOLDER, new_subfolder)
        src_filename = os.path.join(old_folder, f'{src_checkpoint_update_id}.pt')
        os.makedirs(new_domain_folder_path, exist_ok=True)
        filename = os.path.join(new_domain_folder_path, 'checkpoint_last.pt')
        if not os.path.exists(filename):
            if copy:
                shutil.copyfile(src_filename, filename)
            print('src_filename', src_filename)
        else:
            print("False")
    # copy checkpoint - demix model, filename)
    elif 'checkpoint_last-shared.pt' in files: 
        import torch
        new_domain_folder_path = os.path.join(NEW_MODEL_TOP_FOLDER, new_subfolder)
        expert_path = os.path.join(old_folder, f'{src_checkpoint_update_id}-rank-{domain_id}.pt')
        os.makedirs(new_domain_folder_path, exist_ok=True)
        with open(expert_path, "rb") as f:
            expert_state = torch.load(f, map_location=torch.device("cpu"))
        with open(re.sub('rank-[0-9]+', 'shared', expert_path), "rb") as f:
            shared_state = torch.load(f, map_location=torch.device("cpu"))
        state = merge_expert_and_shared_state(expert_state, shared_state)
        filename = os.path.join(new_domain_folder_path, 'checkpoint_last.pt')
        if not os.path.exists(filename):
            if copy:
                with open(filename, 'wb') as f:
                    torch.save(state, filename)
            print('expert_path', expert_path)
        else:
            print("False")


if __name__=='__main__':
    args = add_args()
    main(args.old_folder, args.new_folder, args.subfolder, args.new_subfolder, args.domain_id, args.phase_one_ratio, args.phase_one_update_num)
