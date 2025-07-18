import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(
        raw_csv,
        processed_dest_path,
        test_size= 0.2,
        random_state= 50
):
    os.makedirs(processed_dest_path, exist_ok= True)

    df = pd.read_csv(raw_csv)
    X = df.drop("target", axis= 1).values
    y = df["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size= test_size,
        random_state= random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_df = pd.DataFrame(X_train_scaled, columns= df.columns[: -1])
    train_df["target"] = y_train
    test_df = pd.DataFrame(X_test_scaled, columns= df.columns[: -1])
    test_df["target"] = y_test

    train_path = os.path.join(processed_dest_path, "train.csv")
    test_path = os.path.join(processed_dest_path, "test.csv")

    train_df.to_csv(train_path, index= False)
    test_df.to_csv(test_path, index= False)

    print(f"Saved train dataset to: {train_path}")
    print(f"Saved test dataset to: {test_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description= "Preprocess synthetic data generated for regression")
    p.add_argument("--raw", "-r", default= "data/raw/latest.csv", help= "Path to raw synthetic data csv file")
    p.add_argument("--out", "-o", default= "data/processed/", help= "Output folder for train/test csv files")
    p.add_argument("--test", "-t", type= float, default= 0.2, help= "Fraction for test split")
    p.add_argument("--seed", "-s", type= int, default= 50, help= "Random state for split")
    args = p.parse_args()

    preprocess(
        raw_csv= args.raw,
        processed_dest_path= args.out,
        test_size= args.test,
        random_state= args.seed
    )