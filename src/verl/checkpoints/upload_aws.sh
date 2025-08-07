#!/bin/bash

# Uploads the best model checkpoint to an S3 bucket.
# Example) bash uplaod_aws.sh wbl-off

FILENAME=$1
BASENAME=$(basename "$FILENAME")
echo "Uploading $FILENAME to S3 bucket dld-llm-train"
cd $FILENAME
cd best
tar -cf - actor | zstd -o ./$BASENAME.tar.zst
aws s3 cp $BASENAME.tar.zst s3://dld-llm-train-tokyo
rm $BASENAME.tar.zst
cd ../../../
