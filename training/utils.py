import logging
import os
import shutil
def setup_logging(args, mode="train"):

    # get unique number of experiment
    log_root = args.log_dir
    if os.path.exists(log_root):
        avail_nums = os.listdir(log_root)
        avail_nums = [-1] + [int(d) for d in avail_nums if d.isdigit()]
        log_num = max(avail_nums) + 1
    else:
        log_num = 0
    log_num = str(log_num)
    print("Logging in log_dir {}, number {}".format(log_root, log_num))

    # get log directories and setup logger
    weight_dir = os.path.join(args.log_dir, log_num, "checkpoints")
    
    os.makedirs(weight_dir, exist_ok=True)

    log_path = os.path.join(args.log_dir, log_num, "log.txt")
    logging.basicConfig(filename=log_path,
                    filemode='a',
                    format='%(asctime)s | %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO, force=True)
    logging.info("starting new experiment")

    

    return weight_dir