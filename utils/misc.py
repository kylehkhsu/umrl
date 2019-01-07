from pyhtmlwriter.Element import Element
from pyhtmlwriter.TableRow import TableRow
from pyhtmlwriter.Table import Table
from pyhtmlwriter.TableWriter import TableWriter
import os
import ipdb
import re
from collections import defaultdict
import torch
import copy
from a2c_ppo_acktr.utils import get_vec_normalize


def save_model(args, policy, envs, iteration, sub_dir='ckpt'):
    os.makedirs(os.path.join(args.log_dir, sub_dir), exist_ok=True)
    if args.cuda:
        policy = copy.deepcopy(policy).cpu()    # apparently a really ugly way to save to CPU
    save_model = [policy, getattr(get_vec_normalize(envs.envs), 'ob_rms', None)]
    torch.save(save_model, os.path.join(args.log_dir, sub_dir, 'iteration_{}.pt'.format(iteration)))


def load_model(log_dir, iteration, sub_dir='ckpt'):
    return torch.load(os.path.join(log_dir, sub_dir, 'iteration_{}.pt'.format(iteration)))


def make_html(root_dir, sub_dir='vis', extension='.mp4'):
    contents = os.listdir(os.path.join(root_dir, sub_dir))
    regexp = re.compile('iteration_*(\d+)-*(task_*.+){}'.format(extension), flags=re.ASCII)
    iter_to_media = defaultdict(list)
    for filename in contents:
        match = regexp.search(filename)
        if match:
            iter = int(match[1])
            task_info = match[2]
            iter_to_media[iter].append((filename, task_info))

    table = Table()
    for iter in sorted(iter_to_media.keys(), reverse=True):
        row = TableRow(rno=iter)

        e = Element()
        e.addTxt('iteration {}'.format(iter))
        row.addElement(e)

        for (filename, task_info) in sorted(iter_to_media[iter], key=lambda x: x[1]):
            e = Element()
            e.addTxt(task_info)
            e.addVideo(os.path.join(sub_dir, filename))
            row.addElement(e)

        table.addRow(row)
    tw = TableWriter(table, outputdir=root_dir, rowsPerPage=len(iter_to_media))
    tw.write()


if __name__ == '__main__':
    make_html('./output/debug/half-cheetah/20190106/rl2_tasks-direction-two_run3')
