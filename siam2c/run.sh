if [ -z "$1" ]
  then
    echo "Need input base model!"
    echo "Usage: bash `basename "$0"` \$BASE_MODEL"
    exit
fi


ROOT=`git rev-parse --show-toplevel`
export PYTHONPATH=$ROOT:$PYTHONPATH

mkdir -p nl-logs

base=$1

python -u $ROOT/tools/train_siammask_refine.py \
    --config=config.json -b 8 \
    -j 4 --pretrained $base \
    --epochs 10 \
    --clip 5 \
    -p 50 \
    2>&1 | tee nl-logs/nl-train.log
