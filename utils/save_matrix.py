import numpy as np

def save_matrix(matrix, name):
    np.save(f"/home/dc755/l65_be301_dc755/saved_matrices/{name}", matrix)
    print(f"Matrix saved as {name}.npy")