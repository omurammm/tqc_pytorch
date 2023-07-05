# utd run !!!!!!!!!!!!!!!!

trap "kill 0" EXIT
for i in 3
do
    for j in 0
    do
        for k in 1 2 3 4
        do
            python main.py --env Walker2d-v2 --bottom_quantiles_to_drop_per_net ${i} --top_quantiles_to_drop_per_net ${j} --move_mean_quantiles ${k} --move_mean_from_origin --seed 0 --utd 20 --max_timesteps 30000000 --save_model_freq 10000  --start_training_data 5000 &
        done
    done
done
wait