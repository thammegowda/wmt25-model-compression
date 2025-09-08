#!/usr/bin/env bash
# Created by: TG Gowda on 2025-07-31

# sanity check all submissions in the /model/ directory

EXEC=echo

while [[ "$1" != "" ]]; do
    case $1 in
        -y|--yes)
            EXEC="bash -c"
            ;;
    esac
    shift
done

tag=eval02
for s in mcptqsr tcd-kreasof-slim mcsr-v2 vicomtech; do
    #export SUB_ID=$s
    #$EXEC "SUB_ID=$s amlt run amlt/a100.yml :sanity=$s-sanity04"
    # debug
    #$EXEC "SUB_ID=$s amlt run -y amlt/a100.yml :debug=debug03-$s"
    # evaluate
    $EXEC "SUB_ID=$s amlt run -y amlt/a100.yml :evaluate=$tag-$s"
done

# run baseline
$EXEC "SUB_ID=baseline amlt run -y amlt/a100.yml :evaluate=$tag-baseline -x --baseline"

