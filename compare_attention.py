import numpy as np
import os
from train_texas import save_attention_matrix
from network_analysis import run_thresholded_attention_analysis
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from utils.web_kb import texas_data, cornell_data, wisconsin_data
from torch_geometric.utils import to_dense_adj


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
    for filename in sorted(os.listdir(directory)):
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

def process_attention_matrices(directory, masked_directory, masked_filename, analysis_title, index):
    matrix_list = load_attention_directory(directory)
    save_attention_matrix(compare_attention_list(matrix_list[:index]), masked_directory, masked_filename)
    print("Masked attention matrix saved.")
    masked = np.load(os.path.join(masked_directory, masked_filename + ".npy"))
    # count the number of 1s
    print(np.count_nonzero(masked))
    print(masked)
    seed_1 = np.load(os.path.join(directory, "texas_attention_matrix_seed_1.npy"))
    # count the number of 1s
    print(np.count_nonzero(seed_1))
    print(seed_1)
    return np.count_nonzero(masked)

def graph_non_zero_count(directory, masked_directory, masked_filename, analysis_title, index):
    non_zero_counts = []
    for i in range(1, index + 1):
        non_zero_counts.append(process_attention_matrices(directory, masked_directory, f"{masked_filename}_{i}", analysis_title, i))
    print(non_zero_counts)

    import matplotlib.pyplot as plt
    plt.plot(range(1, index + 1), non_zero_counts)
    plt.xlabel("Number of seeds")
    plt.ylabel("Number of non-zero entries")
    plt.title("Number of non-zero entries in the attention matrix")
    plt.savefig("non_zero_entries.png")
    plt.show()

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

    # analysis_directory("texas_attention_matrices_001", "Texas")

    # matrix_list = load_attention_directory("cornell_attention_matrices_001")
    # save_attention_matrix(compare_attention_list(matrix_list), "cornell_attention_matrices_001", "cornell_attention_matrix_seed_masked")

    # masked = np.load("cornell_attention_matrices_001/cornell_attention_matrix_seed_masked.npy")
    # # count the number of 1s
    # print(np.count_nonzero(masked))
    # print(masked)
    # seed_1 = np.load("cornell_attention_matrices_001/cornell_attention_matrix_seed_1.npy")
    # # count the number of 1s
    # print(np.count_nonzero(seed_1))
    # print(seed_1)

    # analysis_directory("cornell_attention_matrices_001", "Cornell")

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

    # matrix_list = load_attention_directory("attention_matrix_seeds/texas_attention_matrices_001")
    # save_attention_matrix(compare_attention_list(matrix_list[:96]), "attention_matrix_seeds/texas_attention_matrices_001/masked", "texas_attention_matrix_seed_masked_96")
    # print("Masked attention matrix saved.")
    # masked = np.load("attention_matrix_seeds/texas_attention_matrices_001/masked/texas_attention_matrix_seed_masked_96.npy")
    # # count the number of 1s
    # print(np.count_nonzero(masked))
    # print(masked)
    # seed_1 = np.load("attention_matrix_seeds/texas_attention_matrices_001/texas_attention_matrix_seed_1.npy")
    # # count the number of 1s
    # print(np.count_nonzero(seed_1))
    # print(seed_1)

    graph_non_zero_count("attention_matrix_seeds/texas_attention_matrices_01", "attention_matrix_seeds/texas_attention_matrices_01/masked", "texas_attention_matrix_seed_masked", "Texas", 100)
    
    masked_96 = np.load("attention_matrix_seeds/texas_attention_matrices_001/masked/texas_attention_matrix_seed_masked_96.npy")
    thresholded_96 = np.where(masked_96 > 0, 1, 0)

    is_diagonal_zero = np.sum(np.diag(thresholded_96) == 0)
    print("Is diagonal all 0s? ", is_diagonal_zero)

    masked_100 = np.load("attention_matrix_seeds/texas_attention_matrices_0001/masked/texas_attention_matrix_seed_masked_100.npy")
    thresholded_100 = np.where(masked_100 > 0, 1, 0)
    # run_thresholded_attention_analysis(thresholded_96, "Texas")
    data = texas_data()
    data.dense_adj = to_dense_adj(data.edge_index, max_num_nodes=data.x.shape[0])[0].cpu().numpy()

    is_diagonal_zero = np.sum(np.diag(data.dense_adj) == 0)
    print("Is diagonal all 0s? ", is_diagonal_zero)
    precision = precision_score(data.dense_adj.flatten(), thresholded_100.flatten())
    recall = recall_score(data.dense_adj.flatten(), thresholded_100.flatten())
    f1 = f1_score(data.dense_adj.flatten(), thresholded_100.flatten())
    mcc = matthews_corrcoef(data.dense_adj.flatten(), thresholded_100.flatten())
    print("Precision: ", precision, "Recall: ", recall, "F1: ", f1, "MCC: ", mcc)