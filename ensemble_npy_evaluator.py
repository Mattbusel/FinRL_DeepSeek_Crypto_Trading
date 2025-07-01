import numpy as np
import os

from metrics import *

directory_path = "exps"

data_dict = {}

for filename in os.listdir(directory_path):
    if filename.endswith(".npy"):
        file_path = os.path.join(directory_path, filename)
        data = np.load(file_path)

        if "net_assets" in filename:
            returns = []
            for t in range(len(data) - 1):
                r_t = data[t]
                r_t_plus_1 = data[t + 1]
                return_t = (r_t_plus_1 - r_t) / r_t
                returns.append(return_t)
            returns = np.array(returns)

            final_sharpe_ratio = sharpe_ratio(returns)
            final_max_drawdown = max_drawdown(returns)
            final_roma = return_over_max_drawdown(returns)

            print(filename, f"sharpe: {final_sharpe_ratio}, roma: {final_roma}, max d: {}")
            
        if "correct_preds" in filename:
            print(filename, f"win rate: {np.count_nonzero(data == 1) / len(data)}, loss rate: {np.count_nonzero(data == -1) / len(data)}" )

        data_dict[filename] = data
