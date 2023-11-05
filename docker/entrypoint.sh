#！/bin/bash

echo "Welcome To Use Bisheng-RT, Developed by DataElem, Inc."

pushd /opt/biheng-rt

# Update Hasp License Tokens
BIN="./bin/rtserver"
if [ $# -ge 1 ]; then
  # arg1: --serveraddr=your_ip"
  address="$1"
  echo "Use enterprise mode, need the activated lisence server address"
  BIN="./bin/rtserver.enter"
  if [[ ! -d /root/.hasplm ]]; then
    mkdir -p /root/.hasplm
  fi
  HASP_FILE="/root/.hasplm/hasp_32042.ini"
  printf "${address:2}\nrequestlog=1\nerrorlog=1\nbroadcastsearch=0" > $HASP_FILE
else
  echo "Use community mode, private model can not be loaded"
fi

# Start the service
$BIN f
# tail -f /dev/null