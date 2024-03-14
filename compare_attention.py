import numpy as np
import os
from train_texas import save_attention_matrix
from network_analysis import run_thresholded_attention_analysis


def compare_attention(matrix1, matrix2):
    """
    Mask two attention matrices.
    """
    loaded_attention1 = np.load(matrix1)
    loaded_attention2 = np.load(matrix2)
    print("Attention matrices loaded.")

    # mask the attention matrices
    result = np.logical_and(loaded_attention1, loaded_attention2)

    result_int = result.astype(int)

    print(result_int)

    return result_int

def load_attention_directory(directory):
    """
    Load all attention matrices in a directory.
    """
    import os
    matrix_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            matrix_list.append(np.load(os.path.join(directory, filename)))
    return matrix_list

def compare_attention_list(matrix_list):
    stacked_matrices = np.stack(matrix_list)
    result = np.logical_and.reduce(stacked_matrices)
    result_int = result.astype(int)

    print(result_int)
    return result_int

def analysis_directory(directory="texas_attention_matrices", title="Texas"):
    """
    For every attention matrix in a directory, create a new subdirectory for the analysis of that matrix.
    Then switch to that directory and run the analysis in there
    """
    for filename in os.listdir(directory):
        if filename.endswith(".npy"):
            matrix = np.load(os.path.join(directory, filename))
            new_directory = os.path.join(directory, filename.split(".")[0])
            if not os.path.exists(new_directory):
                print(new_directory)
                os.makedirs(new_directory)
                print("Present working directory: ", os.getcwd())
                os.chdir(new_directory)
                print("Present working directory: ", os.getcwd())
                run_thresholded_attention_analysis(matrix, title)
                # change back to the original directory
                os.chdir("../..")
                print("Present working directory: ", os.getcwd())

if __name__ == "__main__":
    # matrix_list = load_attention_directory("texas_attention_matrices_001")
    # save_attention_matrix(compare_attention_list(matrix_list), "texas_attention_matrices_001", "texas_attention_matrix_seed_masked")
    # masked = np.load("texas_attention_matrices_001/texas_attention_matrix_seed_masked.npy")
    # # count the number of 1s
    # print(np.count_nonzero(masked))
    # print(masked)
    # seed_1 = np.load("texas_attention_matrices_001/texas_attention_matrix_seed_1.npy")
    # # count the number of 1s
    # print(np.count_nonzero(seed_1))
    # print(seed_1)

    analysis_directory("texas_attention_matrices_001", "Texas")

    # matrix_list = load_attention_directory("cornell_attention_matrices")
    # save_attention_matrix(compare_attention_list(matrix_list), "cornell_attention_matrices", "cornell_attention_matrix_seed_masked")

    # masked = np.load("cornell_attention_matrices/cornell_attention_matrix_seed_masked.npy")
    # # count the number of 1s
    # print(np.count_nonzero(masked))
    # print(masked)
    # seed_1 = np.load("cornell_attention_matrices/cornell_attention_matrix_seed_1.npy")
    # # count the number of 1s
    # print(np.count_nonzero(seed_1))
    # print(seed_1)

    # analysis_directory("cornell_attention_matrices")

    # matrix_list = load_attention_directory("wisconsin_attention_matrices")
    # save_attention_matrix(compare_attention_list(matrix_list), "wisconsin_attention_matrices", "wisconsin_attention_matrix_seed_masked")

    # masked = np.load("wisconsin_attention_matrices/wisconsin_attention_matrix_seed_masked.npy")
    # # count the number of 1s
    # print(np.count_nonzero(masked))
    # print(masked)
    # seed_1 = np.load("wisconsin_attention_matrices/wisconsin_attention_matrix_seed_1.npy")
    # # count the number of 1s
    # print(np.count_nonzero(seed_1))
    # print(seed_1)

    # analysis_directory("wisconsin_attention_matrices", "Wisconsin")