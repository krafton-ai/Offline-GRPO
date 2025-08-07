#!/bin/bash

# Uploads the best model checkpoint to an S3 bucket.
# Example) bash uplaod_aws.sh wbl-off
for FILENAME in $1/*; do
    # Check if FILENAME is a directory
    if [ ! -d "$FILENAME" ]; then
        continue
    fi
    BASENAME=$(basename "$FILENAME")
    echo "Uploading $FILENAME to S3 bucket dld-llm-train"
    cd $FILENAME
    cd best
    tar -cf - actor | zstd -o ./$BASENAME.tar.zst
    aws s3 cp $BASENAME.tar.zst s3://dld-llm-train-tokyo
    rm $BASENAME.tar.zst
    cd ../../../
done
