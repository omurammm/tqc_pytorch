trap "kill 0" EXIT
for i in 5
do
    for j in 0
    do
        for k in 0 1 2 3
        do
            python main.py --env Walker2d-v2 --bottom_quantiles_to_drop_per_net $i --top_quantiles_to_drop_per_net $j --move_mean_quantiles $k --move_mean_from_origin --seed 0 &
        done
    done
done
wait