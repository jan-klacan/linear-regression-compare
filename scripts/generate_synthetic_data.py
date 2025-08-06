import os
import argparse
import pandas as pd
import json
from sklearn.datasets import make_regression

def generate(
        dest_path,
        n_samples= 1000,
        n_features= 5,
        noise= 0.1,
        random_state= 50
):
    os.makedirs(dest_path, exist_ok= True)
    file_name = f"synthetic_{n_samples}x{n_features}_noise_{noise}.csv"
    file_path = os.path.join(dest_path, file_name)

    X, y = make_regression(
        n_samples = n_samples,
        n_features = n_features,
        noise = noise,
        random_state = random_state
    )

    df = pd.DataFrame(X, columns= [f"feature_{i+1}" for i in range(n_features)])
    df["target"] = y
    df.to_csv(file_path, index= False)

    print(f"Synthetic dataset saved to: {file_path}")

    latest_path = os.path.join(dest_path, "latest.csv")
    df.to_csv(latest_path, index= False)
    print(f"Also saved a copy to: {latest_path}")

    meta = {
        "base": os.path.splitext(os.path.basename(file_path))[0]
    }
    meta_path = os.path.join(dest_path, "latest_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent= 2)
    print(f"Metadata for the base file name saved to: {meta_path}")

    return file_path, latest_path, meta_path

if __name__ == "__main__":
    p = argparse.ArgumentParser(description= "Generate synthetic data for regression")
    p.add_argument("--dest", "-d", default= "data/raw/", help= "Output folder")
    p.add_argument("--samples", "-n", type= int, default= 1000, help= "Number of samples")
    p.add_argument("--features", "-f", type= int, default= 5, help= "Number of features")
    p.add_argument("--noise", "-no", type= float, default= 0.1, help= "Noise level")
    p.add_argument("--seed", "-s", type= int, default= 50, help= "Random state")
    args = p.parse_args()

    generate(
        dest_path= args.dest,
        n_samples= args.samples,
        n_features= args.features,
        noise= args.noise,
        random_state= args.seed
    )