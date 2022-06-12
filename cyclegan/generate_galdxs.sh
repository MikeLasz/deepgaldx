#!/bin/bash
for epoch in {0..10}
do 
  python3 sample_galdx.py --epoch $epoch
done
