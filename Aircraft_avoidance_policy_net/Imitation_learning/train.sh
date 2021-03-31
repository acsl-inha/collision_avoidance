#!/bin/sh
for i in 20 40 60 80 
do
        for j in 20 40 60 80
        do
                for k in 20 40 60 80
                do
                        echo "Start train node ${i}_${j}_${k}"
                        python3 colision_avoidance_net_idx.py --num_nodes "${i}" "${j}" "${k}" --num_layers 1 1 1 --index 0 &
                        python3 colision_avoidance_net_idx.py --num_nodes "${i}" "${j}" "${k}" --num_layers 1 1 1 --index 1 &
                        python3 colision_avoidance_net_idx.py --num_nodes "${i}" "${j}" "${k}" --num_layers 1 1 1 --index 2 &
                        python3 colision_avoidance_net_idx.py --num_nodes "${i}" "${j}" "${k}" --num_layers 1 1 1 --index 3
                done
        done
done
