import numpy as np
import random
from lca import lca_preprocessing, lca_query, Node, binary_tree_to_arr

def arr_to_cartesian(arr, to_add = 0):
    if len(arr) == 0:
        return None, []

    split_index = np.argmin(arr)
    root = Node(split_index + to_add)
    root.left, index_to_node_left = arr_to_cartesian(arr[0 : split_index], to_add = to_add)
    root.right, index_to_node_right = arr_to_cartesian(arr[split_index + 1:], to_add = to_add + split_index + 1)
    index_to_node = index_to_node_left + [root] + index_to_node_right
    return root, index_to_node

def rmq_preprocess(arr):
    tree, index_to_node = arr_to_cartesian(arr)
    preprocessed_tree = lca_preprocessing(tree)
    return (preprocessed_tree, index_to_node)

def rmq_query(preprocessed_arr, start_index, end_index):
    preprocessed_tree, index_to_node = preprocessed_arr
    start_node = index_to_node[start_index]
    end_node = index_to_node[end_index - 1] # Do not include end_index in query
    lca = lca_query(start_node, end_node, preprocessed_tree)
    return lca.data

def test_rmq():
    arr = np.random.randint(0, 100, size = 100)
    print("Array:")
    print(arr)
    preprocessed_arr = rmq_preprocess(arr)

    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            expected = i + np.argmin(arr[i:j])
            actual = rmq_query(preprocessed_arr, i, j)
            assert(expected == actual)
    print("All tests passed!")

if __name__ == '__main__':
    test_rmq()
