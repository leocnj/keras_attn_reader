#!/usr/bin/env bash
exp_name=$1

echo "model to tweak is $exp_name"

nohup ./tweak_asap.sh $exp_name 1 16 gpu2 > log/item1_$exp_name.out &
nohup ./tweak_asap.sh $exp_name 2 16 gpu2 > log/item2_$exp_name.out &
nohup ./tweak_asap.sh $exp_name 3 16 gpu2 > log/item3_$exp_name.out &
nohup ./tweak_asap.sh $exp_name 4 16 gpu1 > log/item4_$exp_name.out &
nohup ./tweak_asap.sh $exp_name 5 16 gpu1 > log/item5_$exp_name.out &
nohup ./tweak_asap.sh $exp_name 6 16 gpu1 > log/item6_$exp_name.out &
nohup ./tweak_asap.sh $exp_name 7 16 gpu0 > log/item7_$exp_name.out &
nohup ./tweak_asap.sh $exp_name 8 16 gpu0 > log/item8_$exp_name.out &
nohup ./tweak_asap.sh $exp_name 9 16 gpu0 > log/item9_$exp_name.out &
nohup ./tweak_asap.sh $exp_name 10 16 gpu0 > log/item10_$exp_name.out &
