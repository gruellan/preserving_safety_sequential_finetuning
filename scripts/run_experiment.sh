#!/bin/bash
# train + eval one experiment end to end
# usage: bash run_experiment.sh <order> <model> <method>

ORDER=$1
MODEL=$2
METHOD=$3

case $ORDER in
order_1) TASKS="code sum qa" ;;
order_2) TASKS="qa sum code" ;;
order_3) TASKS="sum code qa" ;;
order_4) TASKS="code qa sum" ;;
order_5) TASKS="sum qa code" ;;
order_6) TASKS="qa code sum" ;;
esac

case $MODEL in
mistral)
  CONFIG=config/experiment_config.yaml
  OUT=/workspace/dissertation_outputs
  RESULTS=results
  BASE=mistralai/Mistral-7B-Instruct-v0.2
  LAYER=16
  POS=-1
  ;;
llama)
  CONFIG=config/experiment_config_llama.yaml
  OUT=/workspace/llama_outputs
  RESULTS=llama_results
  BASE=meta-llama/Meta-Llama-3-8B-Instruct
  LAYER=12
  POS=-5
  ;;
gemma)
  CONFIG=config/experiment_config_gemma.yaml
  OUT=/workspace/gemma_outputs
  RESULTS=gemma_results
  BASE=google/gemma-2-9b-it
  LAYER=21
  POS=-5
  ;;
esac

BUFFER="--der-buffer-path ${DER_BUFFER:-data/der_buffer_filtered.jsonl} --teacher-cache-dir ${TEACHER_CACHE:-$OUT/teacher_logits_cache}"

case $METHOD in
plain)
  DER=""
  SUFFIX=""
  ;;
plain_matched)
  # budget/batching-matched Plain control: replay trainer path (group_by_length
  # off, buffer concatenated, same update budget as ER/DER/DER++) but task-CE
  # only so it differs from the replay methods only in the loss terms
  DER="--der --der-variant der++ --der-alpha 0 --der-beta 0 $BUFFER"
  SUFFIX="_plainmatched"
  ;;
er)
  DER="--der --der-variant der++ --der-alpha 0 --der-beta 1 $BUFFER"
  SUFFIX="_er_b150"
  ;;
der)
  DER="--der --der-variant der --der-alpha 1 $BUFFER"
  SUFFIX="_der_b150"
  ;;
derpp)
  DER="--der --der-variant der++ --der-alpha 1 --der-beta 1 $BUFFER"
  SUFFIX="_derpp_b150"
  ;;
esac

RUN=$ORDER$SUFFIX

# optional smoke-test. unset -> normal full run
TRAIN_ARGS=""
[ -n "$MAX_TRAIN" ] && TRAIN_ARGS="$TRAIN_ARGS --max-train-samples $MAX_TRAIN"
[ -n "$MAX_TEST" ] && TRAIN_ARGS="$TRAIN_ARGS --max-test-samples $MAX_TEST"
[ -n "$PDB" ] && TRAIN_ARGS="$TRAIN_ARGS --per-device-batch $PDB"
[ -n "$GA" ] && TRAIN_ARGS="$TRAIN_ARGS --grad-accum $GA"
RQ4_ARGS=""
[ -n "$RQ4_MAX_PROMPTS" ] && RQ4_ARGS="$RQ4_ARGS --max-prompts $RQ4_MAX_PROMPTS"
[ -n "$RQ4_ABLATE_N" ] && RQ4_ARGS="$RQ4_ARGS --ablate-n $RQ4_ABLATE_N"

ADAPTERS=""
for t in $TASKS; do
  ADAPTERS="$ADAPTERS $OUT/$RUN/$t"
done

echo "========================= training $RUN ($MODEL) ========================="
# SKIP_TASK_EVALS=1 skips run.py's TRACE task-accuracy matrix, for safety focused evals
python run.py --config $CONFIG --order-name $RUN --tasks $TASKS --results-dir $RESULTS $DER $TRAIN_ARGS ${SKIP_TASK_EVALS:+--skip-evals}

echo "============ harmbench asr ============"
python -m eval.eval_all_checkpoints \
  --base-model $BASE --order-name $RUN --results-dir $RESULTS \
  --adapter-dirs $ADAPTERS --test

echo "============ robustness + over-refusal ============"
python -m eval.eval_robustness \
  --base-model $BASE --results-dir $RESULTS \
  --spec $RUN $ADAPTERS --test

if [ -z "$SKIP_RQ4" ]; then
echo "============ refusal geometry + ablation ============"
RQ4_RESULTS=$RESULTS python -m eval.rq4 --phase all \
  --conditions $RUN --adapters-root $OUT \
  --base-model $BASE --layer $LAYER --pos $POS \
  --harmful orbench_toxic --harmless xstest $RQ4_ARGS
fi

echo "========================= done, results in $RESULTS/$RUN ========================="
