1.2: Use the ipynb. Number of iterations is 1, on Ant and Hopper, no DAgger. Number of layers is 2, size is 64. Learning rate is 5e-3. Eval_batch_size is 5000, but all other parameters should be default.

1.3: Same as above, but use Hopper, and vary size from 16, to 32, to 64, to 128, to 256. I read off the points from Tensorboard to produce the graph.

2: Same as above, on Ant and Hopper, DAgger is on, with 10 iterations. Eval Batch Size is still 5000. I read off points from Tensorboard to produce the graph.