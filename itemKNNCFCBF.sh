#!/bin/bash

for i in 'ICM_product_type_no' 'ICM_graphical_appearance_no'
do
	screen -dmS itemKNNCFCBF_$i
	screen -S itemKNNCFCBF_$i -X stuff "conda activate recsys^M python -m run_hyperparameter_search_itemKNN $i ^M"
done
