FILENAME=$1
NODE_NUMBER=$2

echo "Uploading $FILENAME to S3 bucket dld-llm-train"
tar -cf - $FILENAME | zstd -o ./tensorboard_node${NODE_NUMBER}_${FILENAME}.tar.zst
aws s3 cp ./tensorboard_node${NODE_NUMBER}_${FILENAME}.tar.zst s3://dld-llm-train-tokyo
