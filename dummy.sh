LANG=$1
POS=$2
SPLIT=$3
SEED=$4
TRAINSAMPLES=$5
MODE=$6

PHONOLOGY_PARAMS=()
if [ "$MODE" = "ff" ]
then
  PHONOLOGY_PARAMS+=(--use-phonology)
elif [ "$MODE" = "fp" ]
then
  PHONOLOGY_PARAMS+=(--use-phonology)
  PHONOLOGY_PARAMS+=(--self-attn)
fi
# If MODE="gg", add no flags

python run_transducer.py --dynet-seed "$SEED" --dynet-mem 1000 --dynet-autobatch ON --transducer=haem "${PHONOLOGY_PARAMS[@]}" --train-samples="$TRAINSAMPLES" --input=100 --feat-input=20 --action-input=100 --enc-hidden=200 --dec-hidden=200 --enc-layers=1 --dec-layers=1 --mlp=0 --nonlin=ReLU --dropout=0.5 --optimization=ADADELTA --l2=0 --batch-size=1 --decbatch-size=1 --patience=51 --epochs=5 --align-cls --tag-wraps=both --iterations=150 --param-tying --mode=mle --beam-width=0 --pretrain-epochs=0 --sample-size=20 --scale-negative=1 "$LANG"."$POS"/"$LANG"."$POS"."$SPLIT".train.txt "$LANG"."$POS"/"$LANG"."$POS"."$SPLIT".dev.txt "$SEED"/Outputs_"$(date +'%Y-%m-%d %H%M%S')"_"$LANG"_"$POS"_"$SPLIT"_"$MODE"_None_"$SEED"_"$TRAINSAMPLES" --test-path="$LANG.$POS/$LANG.$POS.$SPLIT.test.txt" --reload-path=dummy