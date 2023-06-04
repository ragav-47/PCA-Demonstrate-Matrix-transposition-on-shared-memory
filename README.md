# CUDA Matrix Transposition
## Aim:
The aim of this experiment is to compare the performance of different matrix transposition implementations using CUDA.

## Procedure:
1)The code implements various matrix transposition techniques using shared memory in CUDA.
<br>2)The different implementations include:
  <br> setRowReadRow: Transpose matrix using row-major ordering for both read and write operations.
  <br> setColReadCol: Transpose matrix using column-major ordering for both read and write operations.
  <br> setColReadCol2: Transpose matrix using column-major ordering for write operation and row-major ordering for read operation.
  <br> setRowReadCol: Transpose matrix using row-major ordering for write operation and column-major ordering for read operation.
  <br> setRowReadColDyn: Transpose matrix using dynamic shared memory and row-major ordering for write operation and column-major ordering for read operation.
  <br> setRowReadColPad: Transpose matrix using row-major ordering for write operation and column-major ordering for read operation, with padding.
   <br>setRowReadColDynPad: Transpose matrix using dynamic shared memory, row-major ordering for write operation, column-major ordering for read operation, with padding.
<br>3)The code measures the execution time of each implementation using CUDA events.
<br>4)The results of the matrix transposition are verified by comparing the output with the expected result.
<br>5)The performance of each implementation is compared based on their execution times.

## Output:
![240358702-65657470-fbdb-4c1c-881a-2b1bd6eb1e97](https://github.com/ragav-47/PCA-Demonstrate-Matrix-transposition-on-shared-memory/assets/75235488/32907522-fdef-4692-aec4-7f9bb754ce99)

## Result:
The experiment aims to compare the performance of different matrix transposition techniques using shared memory in CUDA. By measuring the execution time of each implementation, we can identify the most efficient technique for matrix transposition.
