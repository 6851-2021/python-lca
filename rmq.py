import numpy as np
import math
import itertools

#################################################
### Solution 1: O(1) query, O(n log n) space ####
#################################################

def preprocess_naive(arr):
    num_elem = len(arr)
    num_powers_of_2 = math.ceil(math.log(num_elem, 2)) + 1

    solutions = np.full((num_elem, num_powers_of_2), -1)

    # Stor answer from every start point...
    for start_idx in range(num_elem):
        # ...for every interval lenth = power of 2
        for i in range(num_powers_of_2):
            end_idx = start_idx + 2 ** i
            solutions[start_idx, i] = np.argmin(arr[start_idx : end_idx])

    return solutions

def query(arr, solutions, start, end):
    largest_pow = math.floor(math.log(end - start, 2))
    interval_size = 2 ** largest_pow

    start_1 = start
    start_2 = end - interval_size

    interval_1_argmin = start_1 + solutions[start_1, largest_pow]
    interval_2_argmin = start_2 + solutions[start_2, largest_pow]

    if arr[interval_1_argmin] <= arr[interval_2_argmin]:
        return interval_1_argmin
    else:
        return interval_2_argmin

def test_method_1():
    arr = [0, 1, 2, 1, 0, 1, 2, 3, 2, 3, 2, 1, 2, 1, 0, 1]
    print("Array:")
    print(arr)
    soln = preprocess_naive(arr)
    print("--------------")
    print("Preprocessing:")
    print(soln)
    print("--------------")
    print("Query (2, 5):", query(arr, soln, 2, 5))
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            expected = i + np.argmin(arr[i:j])
            actual = query(arr, soln, i, j)
            assert(expected == actual)
    print("All tests passed!")

###########################################
### Solution 2: O(1) query, O(n) space ####
###########################################

# Input:  numpy array
# Output: (chunk_size, num_chunks, top_array, bottom_array)
#         where top_array[i] is the argmin of ith chunk
#         and   bottom_array[i] is a view of a numpy array `l`
#               such that l[start, end] is the argmin of the 
#               values between `start' and `end' in the ith chunk
def preprocess_with_indirection(arr):
    # 1) Split array into chunks of 1/2 lg n size
    chunk_size = math.floor(1/2 * math.log(len(arr), 2))
    num_chunks = math.ceil(len(arr)/chunk_size)

    # 2) Construct full lookup table
    #    * Enumerate all possible 2^{chunk_size} +- sqeuences
    lookup = np.zeros(shape = ((2, ) * (chunk_size - 1)) + (chunk_size, chunk_size + 1))
    for step_sequence in itertools.product([0,1], repeat = chunk_size - 1):
        sequence = np.zeros(shape = (chunk_size, ), dtype = int)
        print("CHUNK SIZE:", chunk_size)
        for i in range(1, chunk_size):
            sequence[i] = sequence[i - 1] + (-1 if step_sequence[i - 1] == 0 else 1)
    
    #    * For each, compute the answers to all possible queries
        for start_query in range(0, chunk_size):
            for end_query in range(start_query + 1, chunk_size + 1):
                lookup_index = tuple(step_sequence) + (start_query, end_query)
                print("lookup:", lookup_index)
                print("    with start/end:", start_query, end_query)
                print("    with sequence:", sequence[start_query : end_query])
                print("    with value:", np.argmin(sequence[start_query : end_query]))
                lookup[lookup_index] = np.argmin(sequence[start_query : end_query])

    # 3) Construct "top" array by brute force and "bottom" array of pointers
    chunk_summaries = np.zeros(num_chunks)
    bottom_lookup = [] # a list of *views* onto the full lookup table
    for i in range(num_chunks):
        start_chunk = i * chunk_size
        end_chunk = (i + 1) * chunk_size
        chunk = arr[start_chunk : end_chunk] - arr[start_chunk]
        chunk_sequence = [0 if d == -1 else 1 for d in np.diff(chunk)]
        chunk_summaries[i] = start_chunk + np.argmin(chunk)
        print("chunk_sequence:", tuple(chunk_sequence))
        bottom_lookup.append(lookup[tuple(chunk_sequence)])

    return (chunk_summaries, bottom_lookup)

def test_method_2():
    arr = np.array([0, 1, 2, 1, 0, 1, 2, 3, 2, 3, 2, 1, 2, 1, 0, 1])
    print("Array:")
    print(arr)
    top, bot = preprocess_with_indirection(arr)
    print("--------------")
    print("Preprocessing (top):")
    print(top)
    print("--------------")
    print("Preprocessing (bot):")
    print(bot)
    print("--------------")


if __name__ == '__main__':
    test_method_2()