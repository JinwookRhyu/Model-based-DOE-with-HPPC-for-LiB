import numpy as np

with np.load("optimal1_low_A_N20_0.0_0.2_0.8_1.0_0.0_0.2_0.8_1.0_0.8_1.0_unbalanced_duplicate2_092725.npz") as old:
    # Choose the keys you want to keep
    keep_keys = [k for k in old.files if k != "raw"]

    # Build a dict of arrays to save
    new_data = {k: old[k] for k in keep_keys}

    # Save to a new file
    np.savez("reduced_optimal1_low_A_N20_0.0_0.2_0.8_1.0_0.0_0.2_0.8_1.0_0.8_1.0_unbalanced_duplicate2_092725", **new_data)