import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
import dask.dataframe as dd
import dask.array as da
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

        with open(f"./mxfold_outputs/{x}_chunk_{i}.fa", "w") as f:
            subprocess.run(
                ["mxfold2", "predict", filename],
                stdout=f,
            )

        df = (
            pd.DataFrame(pd.read_csv(f"./mxfold_outputs/{x}_chunk_{i}.fa", header=None)[0]
            .values
            .astype(str)
            .reshape([-1, 3])
        )
        )
        df.columns = ["BC_order", "peGRNA sequenCe", "mxfold2_prediction"]
        
        os.remove(filename)
        os.remove(f"./mxfold_outputs/{x}_chunk_{i}.fa")

        return df

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
    for x in [i for i in sorted(os.listdir("./mxfold_inputs/")) if i.endswith(".csv")]:

        # system initialization
        logical = False
        df_results = []
        num_procs = psutil.cpu_count(logical=logical)
        if len(sys.argv) > 1:
            num_procs = int(sys.argv[1])

        big_df = read_and_process_data(x)
        splitted_df = np.array_split(big_df, num_procs // 8)

        with ProcessPoolExecutor(
            max_workers=len(splitted_df) if len(splitted_df) <= num_procs else num_procs
        ) as executor:
            rvals = [
                executor.submit(parallel_processing, df=df, i=i)
                for i, df in enumerate(splitted_df)
            ]
            for rval in as_completed(rvals):
                try:
                    df_results.append(rval.result()) 

                except Exception as ex:
                    print(str(ex))
                    pass

        # Concatenate results
        df = pd.DataFrame([], columns=["BC_order", "peGRNA sequenCe", "mxfold2_prediction"])
        
        for df_result in df_results:
            df = pd.concat([df, df_result], axis=0  )
        
        df.to_csv(f"./mxfold_outputs/{x}_prediction.csv", index=False, header=True)
