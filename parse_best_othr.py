import numpy as np
import pickle
import pandas as pd
import os, argparse

def parse_best_K(pkl_path, save_path):
    best_k_file = pkl_path
    with open(best_k_file, "rb") as fp:
        results_dict = pickle.load(fp)

    df = pd.DataFrame(results_dict).T.reset_index()
    df.columns = ['image','k','thr','MUCov', 'MWCov']
    df = df.fillna(0)
    # merge df by the max MWCov groupby "image"
    df_max = df.groupby(by="image", as_index=False)["MWCov"].max()
    df_val = pd.merge(df_max,df,how="left")
    
    # delete the duplicated row though some rules
    result_df = pd.DataFrame(columns = df_val.columns)
    for image in df_val.image.unique():
        sub_df = df_val[df_val.image == image]
        if len(sub_df) == 1:
            result_df = result_df.append(sub_df)
        else:
            result_df = result_df.append(sub_df[sub_df.MUCov == sub_df.MUCov.max()].iloc[0])
    # save file
    result_df.to_csv(save_path, index=False)
    print("mean MWCov:", result_df["MWCov"].mean())
    print("save file: ", save_path)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="parse the best k pkl file and get the csv file"
    )
    parser.add_argument(
        "--pkl_path",
        help="The pkl file",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        help="The save path",
        type=str,
    )
    args = parser.parse_args()
    
    parse_best_K(args.pkl_path, args.save_path)