#!/bin/bash
# Stop one run, including actors and engines in independent sessions.
#   tools/b100/stop.sh <name> [engine_dir_basename]
NAME=$1
STOP_TIMEOUT=${STOP_TIMEOUT:-900}
name_re=$(printf '%s' "$NAME" | sed 's/[][\\.^$*+?(){}|]/\\&/g')
roots=()
while read -r pid; do
    roots+=("$pid")
done < <(pgrep -u "$USER" -f "[r]l[.]train --name ${name_re}([[:space:]]|$)")
while read -r pid; do
    roots+=("$pid")
done < <(pgrep -u "$USER" -f "[r]l[.]eval_watch --run_dir runs/${name_re}([[:space:]]|$)")

pids=() parents=() commands=() tracked=()
while read -r pid ppid command; do
    pids+=("$pid")
    parents+=("$ppid")
    commands+=("$command")
    tracked+=(0)
done < <(ps -ww -u "$USER" -o pid=,ppid=,args=)
for ((i=0; i<${#pids[@]}; i++)); do
    for pid in "${roots[@]}"; do
        if [[ ${pids[$i]} == "$pid" ]]; then tracked[$i]=1; fi
    done
done
changed=1
while ((changed)); do
    changed=0
    for ((i=0; i<${#pids[@]}; i++)); do
        [[ ${tracked[$i]} == 1 ]] && continue
        for ((j=0; j<${#pids[@]}; j++)); do
            if [[ ${tracked[$j]} == 1 && ${parents[$i]} == "${pids[$j]}" ]]; then
                tracked[$i]=1
                changed=1
                break
            fi
        done
    done
done

alive() {
    local row stat command
    row=$(ps -ww -p "${pids[$1]}" -o stat=,args=) || return 1
    read -r stat command <<< "$row"
    [[ $stat != Z* && $command == "${commands[$1]}" ]]
}

signal_tree() {
    local i
    for ((i=${#pids[@]}-1; i>=0; i--)); do
        if [[ ${tracked[$i]} == 1 ]] && alive "$i"; then
            kill -"$1" "${pids[$i]}" 2>/dev/null
        fi
    done
}

for ((i=0; i<${#pids[@]}; i++)); do
    for pid in "${roots[@]}"; do
        if [[ ${pids[$i]} == "$pid" ]] && alive "$i"; then
            kill -TERM "$pid" 2>/dev/null
        fi
    done
done
start=$SECONDS
next_report=30
while true; do
    live=0
    root_live=0
    for ((i=0; i<${#pids[@]}; i++)); do
        if [[ ${tracked[$i]} == 1 ]] && alive "$i"; then
            ((live+=1))
            for pid in "${roots[@]}"; do
                if [[ ${pids[$i]} == "$pid" ]]; then ((root_live+=1)); fi
            done
        fi
    done
    ((root_live == 0)) && break
    elapsed=$((SECONDS-start))
    if ((elapsed >= STOP_TIMEOUT)); then
        echo "stop $NAME: timeout after ${STOP_TIMEOUT}s; killing $live remaining processes"
        signal_tree KILL
        break
    fi
    if ((elapsed >= next_report)); then
        echo "stop $NAME: $live processes still alive (${elapsed}s)"
        next_report=$((next_report+30))
    fi
    sleep 1
done
signal_tree KILL
for ((attempt=0; attempt<5; attempt++)); do
    ((attempt > 0)) && sleep 1
    remaining=()
    for ((i=0; i<${#pids[@]}; i++)); do
        if [[ ${tracked[$i]} == 1 ]] && alive "$i"; then remaining+=("${pids[$i]}"); fi
    done
    ((${#remaining[@]} == 0)) && exit 0
done
echo "stop $NAME: remaining PIDs: ${remaining[*]}" >&2
exit 1
