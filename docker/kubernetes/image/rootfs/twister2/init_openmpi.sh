#! /bin/bash
####################################################################
# This script starts a Twister2 worker in jobs that use openmpi
# It:
#   It starts an sshd server for password free ssh access
#   gets the job package and unpacks it, all pods are supposed to start only one container
#   mpirun will start the worker processes in pods
#
# if this is the first pod in the job
#   sets the classpath
#   starts the mpimaster class
# if this is not the first pod, this script will just wait to infinity
# mpirun will start the worker process later on
####################################################################

###################  first start sshd #############################
/start_sshd.sh >/tmp/sshd.log 2>&1 &

# if it can not start sshd, exit with a log message
return_code=$?
if [ $return_code -ne 0 ]; then
  echo -n "$return_code" > /dev/termination-log
  exit $return_code
fi

# get the sshd process id
echo "sshd started pid=$(ps auwx |grep [s]sh |  awk '{print $2}')"

###################  get the job package #############################
./get_job_package.sh

# check whether job package downloaded successfully
if [ $? -ne 0 ]; then
  echo "Since the job package can not be retrieved, sleeping to infinity"
  sleep infinity
fi

# update the classpath with the user job jar package
CLASSPATH=$POD_MEMORY_VOLUME/$JOB_ARCHIVE_DIRECTORY/$USER_JOB_JAR_FILE:$CLASSPATH
LOGGER_PROPERTIES_FILE=$POD_MEMORY_VOLUME/$JOB_ARCHIVE_DIRECTORY/$LOGGER_PROPERTIES_FILE

# write host ip to file
echo ${HOST_IP} >> hostip.txt

# write node-ip list to file
echo ${ENCODED_NODE_INFO_LIST} >> node-info-list.txt

###################  check whether this is the first pod #############################
# if this is the first pod in the first StatefulSet, HOSTNAME ends with "-0-0"
# in that case, it should run mpimaster
# otherwise, it should start sshd and sleep to infinity

echo "My hostname: $HOSTNAME"
length=${#HOSTNAME}
# echo "length of $HOSTNAME: $length"

lastFourChars=$(echo $HOSTNAME| cut -c $(($length-3))-$length)
# echo "last two chars: $lastFourChars"

if [ "$lastFourChars" = "-0-0" ]; then
  echo "This is the first pod in the first StatefulSet of the job: $HOSTNAME"
  echo "Starting $CLASS_TO_RUN .... "
  exec java -Djava.util.logging.config.file=$LOGGER_PROPERTIES_FILE $CLASS_TO_RUN
  echo "$CLASS_TO_RUN is done. Starting to sleep infinity ..."
  sleep infinity
else
  echo "This is not the first pod: $HOSTNAME"
  echo "Starting to sleep to infinity ..."
  sleep infinity
fi

