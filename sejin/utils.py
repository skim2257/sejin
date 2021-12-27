import sys
import warnings
import traceback
import gc
import glob
import os

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

def seven(txt):
    txt = str(txt)
    while(len(txt) < 7):
        txt = "0" + txt
    return txt

def find_batch_size(model, input_size):
    model = model.to(device)
    for i in torch.logspace(8, 0, 9, base=2, dtype=int):
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))
        try:
            x = torch.zeros((i, *input_size)).to(device)
            y = model(x)
            print(y.shape)
            print(f"Batch size {i} fit!")
            return i
        except Exception as e:
            print(f"Batch size {i} doesn't fit with error:", e)
            print(traceback.format_exc())
        finally:
            del x
        
        gc.collect()
        

def model_summary(model, file=sys.stderr):
    def repr(model):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = model.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        total_params = 0
        for key, module in model._modules.items():
            mod_str, num_params = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
            total_params += num_params
        lines = extra_lines + child_lines

        for name, p in model._parameters.items():
            total_params += reduce(lambda x, y: x * y, p.shape)

        main_str = model._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        if file is sys.stderr:
            main_str += ', \033[92m{:,}\033[0m params'.format(total_params)
        else:
            main_str += ', {:,} params'.format(total_params)
        return main_str, total_params

    string, count = repr(model)
    if file is not None:
        print(string, file=file)
    return count

def crawler(path):
    series_folders = []
    for n, (path, directories, files) in enumerate(os.walk(path)):
    #     print (f'ls {path}')
    # #     for directory in directories:
    # #         print(f"\td{directory}")
        if len(glob.glob(os.path.join(path, "*.dcm"))) > 2:
            series_folders.append(path)
    #     if n > 5:
    #         break

    print(len(series_folders))
    