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
    
    # Ensure that the output folder exists
    os.makedirs(processed_dest_path, exist_ok= True)

    # Load raw data
    df = pd.read_csv(raw_csv)
    X = df.drop("target", axis= 1).values
    y = df["target"].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size= test_size,
        random_state= random_state
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Build dataframes
    train_df = pd.DataFrame(X_train_scaled, columns= df.columns[: -1])
    train_df["target"] = y_train
    test_df = pd.DataFrame(X_test_scaled, columns= df.columns[: -1])
    test_df["target"] = y_test

    # Get base name from the raw file name
    name_base = os.path.splitext(os.path.basename(raw_csv))[0]

    # Paths for versioned files
    train_versioned = os.path.join(processed_dest_path, f"train_{name_base}.csv")
    test_versioned = os.path.join(processed_dest_path, f"test_{name_base}.csv")

    # Paths for "latest" files
    train_latest = os.path.join(processed_dest_path, "train_latest.csv")
    test_latest = os.path.join(processed_dest_path, "test_latest.csv")

    # Save versioned files
    train_df.to_csv(train_versioned, index= False)
    test_df.to_csv(test_versioned, index= False)    

    # Overwrite the "latest" files 
    train_df.to_csv(train_latest, index= False)
    test_df.to_csv(test_latest, index= False)

    # Print report
    print(f"Saved train dataset to: {train_versioned}")
    print(f"Saved test dataset to: {test_versioned}")
    print(f"Also updated train latest: {train_latest}")
    print(f"Also updated test latest: {test_latest}")

    return train_versioned, test_versioned, train_latest, test_latest

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