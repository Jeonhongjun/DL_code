#!/usr/bin/env bash

echo 'Viewcount predictor running'

echo 'Remove env'
rm -rf env

echo 'Install package'
virtualenv env
source env/bin/activate
env/bin/pip install -r requirements_vcn.txt

echo 'Viewcount install Done'

python3 -m unittest -q viewcount_unittest.timeoutTest
