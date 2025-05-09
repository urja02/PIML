import logging
import os
import shutil
def setup_logging(args, mode="train"):

    # make exp_name using some logic of the args
    exp_name = "/work/pi_eokte_umass_edu/Urja/increased_layers_training"
    # get unique number of experiment
    log_root = os.path.join(args.log_dir, exp_name)
    if os.path.exists(log_root):
        avail_nums = os.listdir(log_root)
        avail_nums = [-1] + [int(d) for d in avail_nums if d.isdigit()]
        log_num = max(avail_nums) + 1
    else:
        log_num = 0
    log_num = str(log_num)
    print("Logging in exp {}, number {}".format(exp_name, log_num))

    # get log directories and setup logger
    weight_dir = os.path.join(args.log_dir, exp_name, log_num, "checkpoints")
    plot_dir = os.path.join(args.log_dir, exp_name, log_num, "plots")
    pred_dir = os.path.join(args.log_dir, exp_name, log_num, "preds")
    os.makedirs(weight_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)

    log_path = os.path.join(args.log_dir, exp_name, log_num, "log.txt")
    logging.basicConfig(filename=log_path,
                    filemode='a',
                    format='%(asctime)s | %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO, force=True)
    logging.info("starting new experiment")

    # Copying the source python files
    src_dir = os.path.join(args.log_dir, exp_name, log_num, "src")
    os.makedirs(src_dir, exist_ok=True)
    cwd = os.getcwd()
    def copy_file(rel_path):
        src = os.path.join(cwd, rel_path)
        dst = os.path.join(src_dir, rel_path)
        shutil.copy(src, dst)
    for fname in os.listdir(cwd):
        if os.path.isdir(os.path.join(cwd, fname)):
            os.makedirs(os.path.join(src_dir, fname), exist_ok=True)
            for fname2 in os.listdir(os.path.join(cwd, fname)):
                if fname2.endswith(".py"):
                    copy_file(os.path.join(fname, fname2))
        if fname.endswith(".py"):
            copy_file(fname)

    return weight_dir, plot_dir, pred_dir