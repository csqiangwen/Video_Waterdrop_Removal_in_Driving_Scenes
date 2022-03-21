OMP_NUM_THREADS=32 python3 ./train.py --gpu_ids 5,7 --batchSize 4 --save_freq 1000 --niter_decay 50000 --shuffle --continue_train --which_iter 1660000
