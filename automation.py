import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
import numpy as np
import psutil
import sys


def parallel_processing(df, i):
    def parse_data(data):
        return pd.DataFrame(data.values.reshape(-1, 1))

    try:
        filename = f"./mxfold_inputs/{x}_chunk_{i}.fasta"
        parse_data(df).to_csv(filename, index=False, header=False)

        outputs = subprocess.run(
            ["mxfold2", "predict", filename],
            stdout=subprocess.PIPE,
        ).stdout.splitlines()

        os.remove(filename)

        return np.array(outputs)

    except KeyboardInterrupt as e:
        import sys

        os.remove(filename)
        sys.exit(1)


def read_and_process_data(input_filename):
    data = pd.read_csv("./mxfold_inputs/" + input_filename)
    data["BC_order"] = data["BC_order"].apply(lambda x: f">{x}")
    data["peGRNA sequenCe"] = data["peGRNA sequenCe"].apply(
        lambda x: x.replace("T", "U")
    )

    return data


if __name__ == "__main__":
    # Input
    for x in tqdm(
        [i for i in sorted(os.listdir("./mxfold_inputs/")) if i.endswith(".csv")]
    ):

        # system initialization
        logical = False
        df_results = np.array([])
        num_procs = psutil.cpu_count(logical=logical)
        if len(sys.argv) > 1:
            num_procs = int(sys.argv[1])

        big_df = read_and_process_data(x)
        splitted_df = np.array_split(big_df, num_procs)

        with ProcessPoolExecutor(max_workers=num_procs) as executor:
            rvals = [
                executor.submit(parallel_processing, df=df, i=i)
                for i, df in enumerate(splitted_df)
            ]
            for rval in as_completed(rvals):
                try:
                    df_results = np.hstack((df_results, rval.result())).astype(str)
                except Exception as ex:
                    print(str(ex))
                    pass

        # Concatenate results
        pd.DataFrame(df_results).to_csv(f"./mxfold_outputs/{x}_prediction.csv", index=False, header=False)
