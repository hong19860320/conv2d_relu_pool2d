# Conv2D+RELU+Pool2D optimization
The conv2d + relu + pool2d structure is commonly used in deep learning models, such as resnet-50, and its performance optimization is particularly important. So we assume the parameters of these three operators are as follows, and try to use some optimization methods to improve its performance.
- Conv2D
  - kernel=[3,3]
  - stride=[1,1]
  - pad=[1,1]
  - dilation=[1,1]
  - group=1
- RELU
- Pool2D
  - method=average
  - kernel=[3,3]
  - stride=[1,1]
  - pad=[1,1]
  - exclusive=false

# Naive implementation
- Description
  - Execute the conv2d, relu and pool2d operators in sequence, and the results of each operator are temporarily stored in memory, the topological relationship is as follows:
input -> Conv2D -> temp_buffer0 -> RELU -> temp_buffer1 -> Pool2D -> output
- FLOPs and R/W intructions
  - Conv2D
    - FLOPs = 2 * N * OC * OH * OW * IC * 3 * 3
    - Load Insts = 2 * N * OC * OH * OW * IC * 3 * 3
    - Write Insts = N * OC * OH * OW
  - RELU
    - FLOPs = N * OC * OH * OW
    - Load Insts = N * OC * OH * OW
    - Write Insts = N * OC * OH * OW
  - Pool2D
    - FLOPs = N * OC * OH * OW * 3 * 3 + 1
    - Load Insts = N * OC * OH * OW * 3 * 3
    - Write Insts = N * OC * OH * OW

# Optimize 1. Conv2D+RELU+Pool2D fused
- Description
  - Fuse Conv2D+RELU+Pool2D into a function, we combine conv2d and relu into one operation, that is, after the convolution operation is performed, the result is obtained by the RELU operation based on registers, instead of saving into a temporary buffer and then reading data from the buffer, avoiding a Load/Write operation to reduce the memory traffic. Furthermore, we use a small buffer of size block[3][OW+2] to temporarily store the intermediate results of conv2d and calculate the result of pool2d directly after processing each row, it is more friendly to the cache. The topological relationship is as follows: input -> Conv2D_RELU -> a small buffer -> Pool2D -> output

  The pseudo-code is shown as follows:
  ```
    for (bs:N)
      for (oc:OC)
        block[3][OW+2] // 3 represents 3 rows of result of convolution, 2 represents the padding of the left and right zero-filling of the conv2d result of each row, and its value depends on the kernel of pool2d.
        for (oh:OH)
          process a row for conv2d, write at block[2][...]
          if oh > 0
            process a row for pool2d based on block, store into output
          copy block[1][...] to block[0][...]
          copy block[2][...] to block[1][...]
  ``` 

- FLOPs and R/W intructions
  - Conv2D_RELU
    - FLOPs = 2 * N * OC * OH * OW * IC * 3 * 3 + N * OC * OH * OW
    - Load Insts = 2 * N * OC * OH * OW * IC * 3 * 3
    - Write Insts = N * OC * OH * OW
  - Pool2D
    - FLOPs = N * OC * OH * OW * 3 * 3 + 1
    - Load Insts = N * OC * OH * OW * 3 * 3
    - Write Insts = N * OC * OH * OW

# Optimize 2. Conv2D+RELU+Pool2D fused + OpenMP
- Description
Based on the "Optimize 1. Conv2D+RELU+Pool2D fused", use OpenMP and multithreading to parallel process the data in the channel dimension. The processing flow is shown as follow:
  ```
  c0 thread0
  c1 thread1
  c2 thread2
  c3 thread3
  c4        thread0
  c5        thread1
  c6        thread2
  c7        thread3
  c8               thread0
  c9               thread1
  c10              thread2
  c11              thread3
  ...
  ```

- FLOPs and R/W intructions
  - Conv2D_RELU
    - FLOPs = 2 * N * OC * OH * OW * IC * 3 * 3 + N * OC * OH * OW
    - Load Insts = 2 * N * OC * OH * OW * IC * 3 * 3
    - Write Insts = N * OC * OH * OW
  - Pool2D
    - FLOPs = N * OC * OH * OW * 3 * 3 + 1
    - Load Insts = N * OC * OH * OW * 3 * 3
    - Write Insts = N * OC * OH * OW

# Environment and results
- CPU: Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz * 10 cores
- Memory: 96 GB
## Peak GLOPS
The peak GLOPS of each core: 512/32x2x2x2.2=140.8 GFLOPS
## Results
|Method|Input image size|Input channel size|Output channel size|Amount of computation(GFLOPs)|Peak computation capability(GFLOPS)|Actual cost(ms)|Computation efficiency(%)|
|---|---|---|---|---|---|---|---|
|Naive implementation|224,224|16|32|0.478478337|140.8|646.6|0.52%|
|Optimize 1. Conv2D+RELU+Pool2D fused|224,224|16|32|0.478478337|140.8|602.2|0.56%|
|Optimize 2. Conv2D+RELU+Pool2D fused + OpenMP(4 threads)|224,224|16|32|0.478478337|140.8*4=563.2|178.3|0.47%|
## How to reproduce the results
```
Ubuntu 16.04/18.04
$ ./build_and_run.sh
```
## Analysis
Compared with the "Naive implementation", fuse relu can reduce the memory traffic by reducing the number of memory accesses to improve the performance.Furthermore, a small buffer is used to temporarily store the convolution data, and the pool2d operation is performed immediately after a row of convolution calculation is completed, it is more friendly to the cache. Then we use multi-cores and multi-threading to process the data in parallel in the c dimension, which can obtain a large performance improvement, but its performance is not proportionally improved due to memory traffic.
