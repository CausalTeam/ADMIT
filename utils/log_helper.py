import pickle
import os

def save_obj(obj, dir, name):
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(dir + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(dir, name):
    with open(dir + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def save_args(args):
    args_dict = args.args_to_dict
    save_obj(args_dict, args.log_dir, 'args_dict')

    with open(os.path.join(args.log_dir, 'args.txt'),'w') as f:
        args = ['{} : {}'.format(key, args_dict[key]) for key in args_dict]
        f.write('\n'.join(args))   
