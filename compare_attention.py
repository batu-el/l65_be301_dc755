import numpy as np

def compare_attention(matrix1, matrix2):
    """
    Compare two attention matrices.
    """
    loaded_attention1 = np.load(matrix1)
    loaded_attention2 = np.load(matrix2)
    print("Attention matrices loaded.")

    # mask the attention matrices
    result = np.logical_and(loaded_attention1, loaded_attention2)

    result_int = result.astype(int)

    print(result_int)

    return result_int

def compare_attention_list(matrix_list):
    stacked_matrices = np.stack(matrix_list)
    result = np.logical_and.reduce(stacked_matrices)
    result_int = result.astype(int)

    print(result_int)
    return result_int