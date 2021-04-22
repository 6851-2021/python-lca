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
    # print("LCA:", lca)
    # print("LCA data:", lca.data)
    return lca.data

def print_tree(root, val="data", left="left", right="right"):
    def display(root, val=val, left=left, right=right):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if getattr(root, right) is None and getattr(root, left) is None:
            line = '%s' % getattr(root, val)
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if getattr(root, right) is None:
            lines, n, p, x = display(getattr(root, left))
            s = '%s' % getattr(root, val)
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if getattr(root, left) is None:
            lines, n, p, x = display(getattr(root, right))
            s = '%s' % getattr(root, val)
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = display(getattr(root, left))
        right, m, q, y = display(getattr(root, right))
        s = '%s' % getattr(root, val)
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    lines, *_ = display(root, val, left, right)
    for line in lines:
        print(line)


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
