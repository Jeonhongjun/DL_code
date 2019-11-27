#!/usr/bin/env bash

echo -e '\nViewcount predictor running'

echo -e '\nRemove env'
rm -rf env

echo -e '\nInstall package'

virtualenv --python=python3.6 env

source env/bin/activate
env/bin/pip install -r requirements_vcn.txt

echo -e '\nViewcount install Done'

python -m unittest -q viewcount_unittest.connection_Test

echo -e '\nVCN_TC_001 success'

python -m unittest -q viewcount_unittest.preprocessing_Test

echo -e '\nVCN_TC_002 success'

python -m unittest -q viewcount_unittest.response_Test

echo -e '\nVCN_TC_003 success'

python -m unittest -q viewcount_unittest.viewcount_Test

echo -e '\nVCN_TC_005 success'

python -m unittest -q viewcount_unittest.loader_Test

echo -e '\nVCN_TC_007 success'

python -m unittest -q viewcount_unittest.timeoutTest

echo -e '\nVCN_TC_008 success'

echo -e '\nTest Done'
