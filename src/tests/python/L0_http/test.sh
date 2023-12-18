#!/bin/bash

set -e
source ../common/util.sh
set +e


MODELDIR=${MODELDIR:=`pwd`/models}
DATADIR=${DATADIR:="/public/bisheng/bisheng_test_data"}
TRITON_DIR=${TRITON_DIR:="/opt/bisheng-rt"}
SERVER=${TRITON_DIR}/bin/rtserver


### LLM / Generate REST API Endpoint Tests ###

pushd ${TRITON_DIR}/tests/L0_http

# Helper library to parse SSE events
# https://github.com/mpetazzoni/sseclient
pip3 install sseclient-py

SERVER_ARGS="f -config ./tests/L0_http/config/server_config.pbtxt"
SERVER_LOG="./inference_server_generate_endpoint_test.log"
CLIENT_LOG="./generate_endpoint_test.log"
run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

## Python Unit Tests
TEST_RESULT_FILE='test_results.txt'
PYTHON_TEST=generate_endpoint_test.py
EXPECTED_NUM_TESTS=14
set +e
python $PYTHON_TEST >$CLIENT_LOG 2>&1
if [ $? -ne 0 ]; then
    cat $CLIENT_LOG
    RET=1
else
    check_test_results $TEST_RESULT_FILE $EXPECTED_NUM_TESTS
    if [ $? -ne 0 ]; then   
        cat $CLIENT_LOG
        echo -e "\n***\n*** Test Result Verification Failed\n***"
        RET=1
    fi
fi
set -e

kill $SERVER_PID
wait $SERVER_PID

popd
