#!/bin/bash

for i in {1..10}
do
	python generate.py experts/Humanoid-v1.pkl Humanoid-v2 10000 "/home/pim/chumanoid/dir$i" &
done

wait
echo "All done"
