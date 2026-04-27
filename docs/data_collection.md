# DECODE-RAPL v2 Data Collection Guide

Complete guide to collecting training data for DECODE-RAPL v2.

## Requirements

- **stress-ng >= 0.19.00** - Required for `--syscall-method` option
  - Install from: https://github.com/ColinIanKing/stress-ng
  - Fedora 39 ships with 0.17.x which is too old
  - Verify version: `stress-ng --version`
- **Go 1.x** - For building the data collector
- **Root access** - Required for RAPL energy measurements (reads `/sys/class/powercap/`)
- **Python 3.x** - For validation and visualization scripts

## Overview

The data collection system:
1. Generates combinatorial workload configurations (2025 combinations for 20-core system)
2. Runs each workload for 45 seconds using stress-ng
3. Collects CPU metrics + RAPL power at 16ms sampling rate
4. Saves to CSV files with workload metadata in filename

## Step 1: Generate Workload Configuration

```bash
cd scripts/
./generate_config.sh
```

This script:
- Auto-detects your CPU core count
- Generates scaled worker counts for each stressor type
- Creates `workload_config.sh` with combinatorial settings
- Shows estimated total run time

**Output example:**
```
Detected 20 logical cores.
Generating scaled workload configuration...

Workload Run Estimate:
  - Time per combination: 62 seconds (45s stress + 2s capture + 15s cooldown)
  - Total combinations: 2025
  ESTIMATED TOTAL TIME: 34 hours, 51 minutes, 28 seconds
```

### Workload Levels Generated

For a 20-core system, `generate_config.sh` creates:
```bash
CPU_WORKERS=(0 4 8 12 16)      # 5 levels: idle, 20%, 40%, 60%, 80%
SYS_WORKERS=(0 4 8 12 16)      # 5 levels
IO_WORKERS=(0 4 8)             # 3 levels
PIPE_WORKERS=(0 4 8)           # 3 levels
VM_WORKERS=(0 4 8)             # 3 levels
CACHE_WORKERS=(0 4 8 12 16 20) # 6 levels (can use >cores for cache)
# Total: 5 × 5 × 3 × 3 × 3 × 6 = 4050 combinations
```

Note: The all-zero combination (idle baseline) is included, not skipped.

## Step 2: Test Collection (Optional but Recommended)

Before running the full 35-hour collection, test with a small subset:

```bash
# Edit workload_config.sh for minimal test
nano workload_config.sh

# Change to:
CPU_WORKERS=(0 4 8)      # 3 levels
SYS_WORKERS=(0 4)        # 2 levels
IO_WORKERS=(0 4)         # 2 levels
PIPE_WORKERS=(0)         # Skip
VM_WORKERS=(0)           # Skip
CACHE_WORKERS=(0)        # Skip
# Total: 3 × 2 × 2 = 12 combinations (~12 minutes)
```

Run test:
```bash
export DATA_DIR=/tmp/rapl-test-data
mkdir -p $DATA_DIR

# IMPORTANT: Must use sudo -E to preserve DATA_DIR
sudo -E ./start_collection.sh

# Monitor progress in another terminal
tail -f /tmp/rapl-test-data/run.log

# Validate results
python validate_collection.py /tmp/rapl-test-data
```

## Step 3: Full Collection

Restore the full configuration and run on the target machine:

```bash
cd scripts/

# Regenerate full config (if you edited it)
./generate_config.sh

# Set data directory
export DATA_DIR=/opt/rapl-training-data
mkdir -p $DATA_DIR

# Start collection (runs in background)
sudo -E ./start_collection.sh
```

**What happens:**
- Copies `workload_config.sh` to `$DATA_DIR/` for reproducibility
- Starts background collection process
- Writes PID and state to `/tmp/decode-rapl-collection.state`
- Logs progress to `$DATA_DIR/run.log`
- Creates CSV files: `run_N_of_TOTAL-cpu*-sys*-io*-pipe*-vm*-cache*.csv`

## Monitoring Collection

### View Progress

```bash
# Watch live log
tail -f /opt/rapl-training-data/run.log

# Check completed runs
wc -l /opt/rapl-training-data/.progress.txt

# List CSV files
ls /opt/rapl-training-data/*.csv | wc -l
```

### Collection State File

State is stored in `/tmp/decode-rapl-collection.state`:
```bash
PID=12345
DATA_DIR=/opt/rapl-training-data
START_TIME=2025-10-19_14:30:00
```

This allows `stop_collection.sh` to find and manage the collection without environment variables.

## Stopping Collection

To stop gracefully (kills processes, keeps progress):

```bash
cd scripts/
sudo ./stop_collection.sh
```

**What it does:**
- Reads state from `/tmp/decode-rapl-collection.state`
- Sends SIGTERM to collection process
- Kills all stress-ng workers
- Shows completion status
- Displays resume instructions

**Output example:**
```
Found state file: /tmp/decode-rapl-collection.state
Collection PID: 12345
Data directory: /opt/rapl-training-data

Stopping collection process...
Completed runs: 458 / 2025
Progress saved to: /opt/rapl-training-data/.progress.txt

To resume collection:
  export DATA_DIR=/opt/rapl-training-data
  sudo -E ./start_collection.sh
```

## Resuming Interrupted Collection

The collection script automatically resumes from where it stopped:

```bash
export DATA_DIR=/opt/rapl-training-data
sudo -E ./start_collection.sh
```

**How it works:**
- Reads `$DATA_DIR/.progress.txt` for completed runs
- Uses `$DATA_DIR/workload_config.sh` (not the one in scripts/)
- Skips already-completed workload combinations
- Continues from next pending run

**Important:** The snapshotted config in `$DATA_DIR/workload_config.sh` is used for resume, ensuring consistency even if you modify `scripts/workload_config.sh`.

## Configuration Snapshot Pattern

On **first run**, `start_collection.sh` copies the workload config:
```bash
scripts/workload_config.sh → $DATA_DIR/workload_config.sh
```

This ensures:
- **Reproducibility**: Each collection knows its exact configuration
- **Safe resume**: Interrupted collections use the same config
- **Parallel collections**: Different data directories can have different configs

**Subsequent runs** (resume) use the config from `$DATA_DIR/workload_config.sh`, not from `scripts/`.

## Output Files

Each collection creates:

### CSV Data Files
```
run_1_of_2025-cpu0-sys0-io0-pipe0-vm0-cache0.csv       # Idle baseline
run_2_of_2025-cpu0-sys0-io0-pipe0-vm0-cache4.csv
run_3_of_2025-cpu0-sys0-io0-pipe0-vm0-cache8.csv
...
run_2025_of_2025-cpu16-sys16-io8-pipe8-vm8-cache20.csv
```

**CSV Format:**
```csv
timestamp_unix,user_percent,system_percent,iowait_percent,ctx_switches_per_sec,package_power_watts
1760876345.676925,0.000,0.000,0.000,2162.5,26.240
1760876345.693161,0.000,0.000,0.000,2155.8,26.086
...
```

- Sampling rate: 16ms (62.5 Hz)
- Duration: ~3000 samples per CSV (~48 seconds)

### Metadata Files
- `run.log` - Human-readable progress log with timestamps
- `.progress.txt` - List of completed run names (for resume)
- `workload_config.sh` - Snapshot of configuration used
- `nohup.log` - stress-ng output (for debugging)

## Validation

After collection completes, validate the data:

```bash
cd scripts/
python validate_collection.py $DATA_DIR
```

This checks:
- All expected CSV files exist
- File sizes are reasonable (>10KB)
- Power values are in expected range (20-100W)
- No negative power (RAPL wrap-around)
- Context switches are reasonable
- Idle baseline was collected

## Troubleshooting

### Issue: Password Prompt in Background

**Problem:** `sudo` asks for password when running in background

**Solution:** Run with `sudo -E` from outside the script:
```bash
sudo -E ./start_collection.sh  # Correct
./start_collection.sh          # Wrong - will prompt for password
```

### Issue: Short CSV Files (~9KB instead of ~200KB)

**Problem:** stress-ng rejected command options

**Diagnosis:** Check `nohup.log` for errors:
```bash
grep "unrecognized\|not known" $DATA_DIR/nohup.log
```

**Common causes:**
- stress-ng version too old (need >= 0.19.00)
- Invalid stress-ng options

### Issue: Negative Power Values

**Problem:** RAPL counter wrapped around

**Solution:** This is fixed in the current collector. If you see negative power:
1. Recompile the collector (it now reads actual RAPL max from sysfs)
2. Rerun affected collections

### Issue: Low System% for Syscall Workload

**Problem:** Pure syscall workload shows <5% system%

**Expected:** This is correct due to multi-core averaging. Fast syscalls complete quickly, leaving CPU mostly idle. The average across all cores is naturally low.

### Issue: Collection Not Found by stop_collection.sh

**Problem:** `stop_collection.sh` says "No state file found"

**Solution:**
- Check if collection is actually running: `ps aux | grep run_workloads`
- Check state file: `cat /tmp/decode-rapl-collection.state`
- If missing, manually kill: `sudo pkill -f run_workloads.sh`

## Performance Considerations

- **Disk space:** ~200KB per CSV × 2025 files = ~400MB for full collection
- **CPU usage:** 100% for stress-ng workers (by design)
- **Memory:** Minimal (<100MB for collector)
- **I/O:** Low (one CSV write per minute)

## Next Steps

After data collection completes, proceed to preprocessing:

```bash
python scripts/prepare_training_data.py \
  --data-dir $DATA_DIR \
  --output-dir data/processed \
  --tau 1 4 8 \
  --skip-startup 100 \
  --seed 42
```

See [Preprocessing Guide](preprocessing.md) for details.
