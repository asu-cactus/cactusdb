import sys
import torch
import tensorflow as tf
import time

def perform_matrix_multiplication(framework, left_matrix_rows, left_matrix_cols, right_matrix_cols):
    if framework == 0:  # PyTorch
        left_matrix = torch.randn(left_matrix_rows, left_matrix_cols)
        weight_matrix = torch.randn(left_matrix_cols, right_matrix_cols)  # Adjust weight matrix size as needed
        startTime = time.time()
        result_matrix = torch.matmul(left_matrix, weight_matrix)
    elif framework == 1:  # TensorFlow
        left_matrix = tf.random.normal((left_matrix_rows, left_matrix_cols))
        weight_matrix = tf.random.normal((left_matrix_cols, right_matrix_cols))  # Adjust weight matrix size as needed
        startTime = time.time()
        result_matrix = tf.matmul(left_matrix, weight_matrix)
    else:
        print("Invalid framework specified.")
        return None
    return startTime

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 script.py <framework> <left_matrix_rows> <left_matrix_cols> <right_matrix_cols>")
        sys.exit(1)

    framework = int(sys.argv[1])
    left_matrix_rows = int(sys.argv[2])
    left_matrix_cols = int(sys.argv[3])
    right_matrix_cols = int(sys.argv[4])

    startTime = perform_matrix_multiplication(framework, left_matrix_rows, left_matrix_cols, right_matrix_cols)
    endTime = time.time()
    print("total time:", endTime - startTime)