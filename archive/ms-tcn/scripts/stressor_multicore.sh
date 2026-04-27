#!/usr/bin/env bash

set -eu -o pipefail

trap exit_all INT
exit_all() {
	pkill -P $$
}

run() {
	echo "❯ $*"
	"$@"
	echo "      ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾"
}

# stepwise load curve: each step is 20 seconds
declare -a load_curve_stepwise=(
	0:20
	20:20
	40:20
	60:20
	80:20
	100:20
	80:20
	60:20
	40:20
	20:20
	0:20
)

# default load curve: varying durations
declare -a load_curve_default=(
	0:5
	10:20
	25:20
	50:20
	75:20
	100:30
	75:20
	50:20
	25:20
	10:20
	0:5
)

main() {
	local total_time=0
	local repeats=5
	local curve_type="default"
	local num_cores=4  # Default to 4 cores for VM simulation

	while getopts "t:r:c:n:" opt; do
		case $opt in
			t) total_time=$OPTARG ;;
			c) curve_type=$OPTARG ;;
			n) num_cores=$OPTARG ;;
			*) echo "Usage: $0 [-t total_time_in_seconds] [-c curve_type(default|stepwise)] [-n num_cores]" >&2; exit 1 ;;
		esac
	done

	# Select load curve based on curve_type
	local -a load_curve
	case $curve_type in
		"default") load_curve=("${load_curve_default[@]}") ;;
		"stepwise") load_curve=("${load_curve_stepwise[@]}") ;;
		*) echo "Invalid curve type. Use 'default' or 'stepwise'" >&2; exit 1 ;;
	esac

	# Get total CPU count
	local total_cpus
	total_cpus=$(nproc)

	# Validate num_cores doesn't exceed available cores
	if [ "$num_cores" -gt "$total_cpus" ]; then
		echo "Error: Requested $num_cores cores but only $total_cpus available" >&2
		exit 1
	fi

	# Build CPU affinity mask (cores 0 to num_cores-1)
	local cpu_mask="0-$((num_cores - 1))"

	# calculate the total duration of one cycle of the load curve
	local total_cycle_time=0
	for x in "${load_curve[@]}"; do
		local time="${x##*:}"
		total_cycle_time=$((total_cycle_time + time))
	done

	# calculate the repeats if total_time is provided
	if [ "$total_time" -gt 0 ]; then
		repeats=$((total_time / total_cycle_time))
	fi

	echo "Configuration:"
	echo "  Total CPUs available: $total_cpus"
	echo "  CPUs to use: $num_cores (cores $cpu_mask)"
	echo "  Total time: $total_time seconds"
	echo "  Repeats: $repeats"
	echo "  Curve type: $curve_type"
	echo ""

	# sleep 5 so that first run and the second run look the same
	echo "Warmup .."
	run taskset -c "$cpu_mask" stress-ng --cpu "$num_cores" --cpu-method ackermann --cpu-load 0 --timeout 5

	for i in $(seq 1 "$repeats"); do
		echo "Running: $i/$repeats"
		for x in "${load_curve[@]}"; do
			local load="${x%%:*}"
			local time="${x##*:}s"
			run taskset -c "$cpu_mask" stress-ng --cpu "$num_cores" --cpu-method ackermann --cpu-load "$load" --timeout "$time"
		done
	done
}

main "$@"
