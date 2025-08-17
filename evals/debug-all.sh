#!/usr/bin/env bash
# Created by: TG Gowda on 2025-07-31

# sanity check all submissions in the /model/ directory

EXEC=echo

while [[ "$1" != "" ]]; do
    case $1 in
        -y|--yes)
            #EXEC="exec"
            EXEC="bash -c"
            ;;
    esac
    shift
done

for s in mcptqsr tcd-kreasof-slim mcsr-wmt25 vicomtech; do
    #export SUB_ID=$s
    $EXEC "SUB_ID=$s amlt run amlt/a100.yml :debug=debug01-$s"
done

