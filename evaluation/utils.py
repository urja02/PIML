import logging
import os
import numpy as np
import pickle


def convert_units(xs, zs):
    xs_converted = xs * 2.54
    ZS_converted = [i * 2.54 for i in zs]
    xs = np.round(xs_converted, 2)
    ZS_new = [np.round(zs, 2) for zs in ZS_converted]
    return xs, ZS_new

def build_pred_graph(batched_graph_test, final_y_pred):
    pred_graph = {}
    current_index = 0
    for i in range(batched_graph_test.batch_size):
        res_test = len(batched_graph_test[i].y)
        pred_values = final_y_pred[current_index:current_index+res_test]
        pred_graph[i] = pred_values
        current_index += res_test
    return pred_graph

def setup_plotting():
    # get unique number of experiment
    plot_root = "plots"
    if os.path.exists(plot_root):
        avail_nums = os.listdir(plot_root)
        avail_nums = [-1] + [int(d) for d in avail_nums if d.isdigit()]
        exp_num = max(avail_nums) + 1
    else:
        exp_num = 0
    exp_num = str(exp_num)
    print("Logging in plot_dir {}, number {}".format(plot_root, exp_num))


    exp_plot_dir = os.path.join(plot_root, exp_num)
    
    os.makedirs(exp_plot_dir, exist_ok=True)

    log_path = os.path.join(plot_root, exp_num, "log.txt")
    logging.basicConfig(filename=log_path,
                    filemode='a',
                    format='%(asctime)s | %(message)s',
                    datefmt='%m-%d %H:%M:%S',
                    level=logging.INFO, force=True)
    logging.info("generating plots for experiment {}".format(exp_num))

    return exp_plot_dir

def load_pickle(path):
    with open(path, 'rb') as fp:
        return pickle.load(fp)