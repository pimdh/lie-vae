#!/bin/bash

for i in {1..10}
do
	python generate.py experts/Humanoid-v1.pkl Humanoid-v2 200000 "../data/humanoid/dir$i" &
done

wait
echo "All done"
