#!/bin/bash

for i in 'UCM_num_sale_months_2018' 'UCM_num_transactions_2018' 'UCM_num_missing_months_2019' 
do
	screen -dmS UserKNNCFCBF_$i
	screen -S UserKNNCFCBF_$i -X stuff "conda activate recsys^M python -m run_hyperparameter_search_userKNN $i ^M"
done
