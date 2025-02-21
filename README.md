# cuda_bitonic_sort
The third project for Parallel and Distributed Systems 2024-25.

To run on **Aristotelis**:
1) Login to Aristotelis and clone the repository.
2) Navigate into the cloned project's main folder.
3) Execute:
   - module ```load gcc/12.2.0```
   - module ```load cuda/12.2.1-bxtxsod```
4) Execute ```make all```
5) Use the ```.sbatch``` file available in the repository to run a job.
6) The number of elements in the array are given as input into the program like so:
   ```./build/cuda_bitonic_sort 27```
   you can choose between **v0**, **v1** and **cuda_bitonic_sort** (v2).
7) Note that the appropriate partitions to run the implementations are **Ampere** and **GPU**.

If something doesn't work and you can not fix it right away please contact me on Zulip. Thank you!
