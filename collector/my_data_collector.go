package main

import (
	"bufio"
	"context"
	"encoding/csv"
	"fmt"
	"io"
	"log"
	"os"
	"os/signal"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// --- Configuration ---

// SampleInterval is the polling rate. 16ms is 0.016s
const SampleInterval = 16 * time.Millisecond

// Paths to search for the RAPL package energy file.
var raplPaths = []string{
	"/sys/class/powercap/intel-rapl:0/energy_uj",
	"/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj",
}

// Paths to search for the RAPL max energy range file.
var raplMaxPaths = []string{
	"/sys/class/powercap/intel-rapl:0/max_energy_range_uj",
	"/sys/class/powercap/intel-rapl/intel-rapl:0/max_energy_range_uj",
}

// procStatPath is the path to the CPU stats file.
const procStatPath = "/proc/stat"

// --- End Configuration ---

// CPUTimes holds the parsed "jiffies" from /proc/stat
type CPUTimes struct {
	user   uint64
	system uint64
	idle   uint64
	iowait uint64
}

// Stats holds all metrics from a single poll.
type Stats struct {
	cpu CPUTimes
	ctx uint64
}

// dataBuffer holds all collected rows in memory until shutdown.
var dataBuffer [][]string

// csvHeader is the header row for the output file.
var csvHeader = []string{
	"timestamp_unix",
	"user_percent",
	"system_percent",
	"iowait_percent",
	"ctx_switches_per_sec",
	"package_power_watts",
}

func main() {
	// --- 1. Argument Parsing ---
	if len(os.Args) != 3 || os.Args[1] != "--outfile" {
		log.Fatalf("Usage: %s --outfile <path-to-csv>", os.Args[0])
	}
	outfile := os.Args[2]

	// --- 2. Graceful Shutdown Setup (Idiomatic Go) ---
	// Create a context that is canceled on SIGINT or SIGTERM
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	// --- 3. Open Files (Once) ---
	raplFile, err := findAndOpenRapl()
	if err != nil {
		log.Fatal(err)
	}
	defer raplFile.Close()

	// Read RAPL max energy range for wrap-around detection
	raplMaxEnergy, err := readRaplMaxEnergy()
	if err != nil {
		log.Printf("Warning: Could not read RAPL max energy range: %v", err)
		log.Printf("Wrap-around detection disabled")
		raplMaxEnergy = 0
	} else {
		log.Printf("RAPL max energy range: %d µJ", raplMaxEnergy)
	}

	statFile, err := os.Open(procStatPath)
	if err != nil {
		log.Fatalf("Failed to open %s: %v", procStatPath, err)
	}
	defer statFile.Close()

	// We use a bufio.Scanner for high-speed parsing of /proc/stat
	statScanner := bufio.NewScanner(statFile)
	// We use a small buffer for reading the RAPL file
	raplBuf := make([]byte, 64)

	log.Println("Starting data collection...")

	// --- 4. Initialize Loop Variables ---
	var lastStats Stats
	var lastEnergy uint64
	var lastTime time.Time

	// Prime the "last" values
	lastStats, err = readProcStat(statFile, statScanner)
	if err != nil {
		log.Fatalf("Failed to read initial stats: %v", err)
	}
	lastEnergy, err = readRaplEnergy(raplFile, raplBuf)
	if err != nil {
		log.Fatalf("Failed to read initial energy: %v", err)
	}
	lastTime = time.Now()

	// Start the 16ms ticker
	ticker := time.NewTicker(SampleInterval)
	defer ticker.Stop()

	// --- 5. Main Collection Loop ---
	for {
		select {
		case <-ticker.C:
			// --- Poll ---
			currentTime := time.Now()
			currentStats, err := readProcStat(statFile, statScanner)
			if err != nil {
				log.Printf("Warning: failed to read stats: %v", err)
				continue
			}
			currentEnergy, err := readRaplEnergy(raplFile, raplBuf)
			if err != nil {
				log.Printf("Warning: failed to read energy: %v", err)
				continue
			}

			// --- Calculate Deltas ---
			timeDelta := currentTime.Sub(lastTime).Seconds()
			if timeDelta == 0 {
				continue
			}

			// CPU Jiffies Deltas
			userDelta := currentStats.cpu.user - lastStats.cpu.user
			sysDelta := currentStats.cpu.system - lastStats.cpu.system
			idleDelta := currentStats.cpu.idle - lastStats.cpu.idle
			iowaitDelta := currentStats.cpu.iowait - lastStats.cpu.iowait
			totalCPUDelta := userDelta + sysDelta + idleDelta + iowaitDelta

			// Context Switch Delta
			ctxDelta := float64(currentStats.ctx - lastStats.ctx)

			// Energy Delta (handle wrap-around)
			energyDelta := int64(currentEnergy - lastEnergy)
			if energyDelta < 0 && raplMaxEnergy > 0 {
				// We wrapped around
				energyDelta += int64(raplMaxEnergy)
			}
			powerWatts := (float64(energyDelta) / 1_000_000) / timeDelta

			// --- Calculate Percentages and Rates ---
			var userPct, sysPct, iowaitPct float64
			if totalCPUDelta > 0 {
				userPct = (float64(userDelta) / float64(totalCPUDelta)) * 100.0
				sysPct = (float64(sysDelta) / float64(totalCPUDelta)) * 100.0
				iowaitPct = (float64(iowaitDelta) / float64(totalCPUDelta)) * 100.0
			}

			ctxPerSec := ctxDelta / timeDelta

			// --- Buffer Data ---
			row := []string{
				fmt.Sprintf("%.6f", float64(currentTime.UnixNano())/1e9),
				fmt.Sprintf("%.3f", userPct),
				fmt.Sprintf("%.3f", sysPct),
				fmt.Sprintf("%.3f", iowaitPct),
				fmt.Sprintf("%.1f", ctxPerSec),
				fmt.Sprintf("%.3f", powerWatts),
			}
			dataBuffer = append(dataBuffer, row)

			// --- Update "last" values for next loop ---
			lastTime = currentTime
			lastStats = currentStats
			lastEnergy = currentEnergy

		case <-ctx.Done():
			// --- 6. Shutdown and Write File ---
			log.Println("Shutdown signal received. Writing data to CSV...")
			if err := writeCSV(outfile); err != nil {
				log.Fatalf("Failed to write CSV: %v", err)
			}
			log.Printf("Successfully wrote %d rows to %s", len(dataBuffer), outfile)
			return
		}
	}
}

// findAndOpenRapl searches for a valid RAPL file and opens it.
func findAndOpenRapl() (*os.File, error) {
	// Try standard paths first
	for _, p := range raplPaths {
		if f, err := os.Open(p); err == nil {
			log.Printf("Found RAPL energy file: %s", p)
			return f, nil
		}
	}
	// Fallback: search /sys/class/powercap/
	var foundPath string
	err := filepath.Walk("/sys/class/powercap/", func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && info.Name() == "energy_uj" && strings.Contains(path, "intel-rapl") {
			foundPath = path
			return io.EOF // Stop walking
		}
		return nil
	})

	if foundPath != "" {
		log.Printf("Found fallback RAPL file: %s", foundPath)
		return os.Open(foundPath)
	}
	if err != nil && err != io.EOF {
		return nil, fmt.Errorf("error searching for RAPL file: %v", err)
	}
	return nil, fmt.Errorf("could not find any intel-rapl 'energy_uj' file")
}

// readRaplMaxEnergy reads the RAPL max energy range for wrap-around detection
func readRaplMaxEnergy() (uint64, error) {
	// Try standard paths
	for _, p := range raplMaxPaths {
		data, err := os.ReadFile(p)
		if err == nil {
			s := strings.TrimSpace(string(data))
			return strconv.ParseUint(s, 10, 64)
		}
	}

	// Fallback: search /sys/class/powercap/
	var foundPath string
	err := filepath.Walk("/sys/class/powercap/", func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() && info.Name() == "max_energy_range_uj" && strings.Contains(path, "intel-rapl") {
			foundPath = path
			return io.EOF // Stop walking
		}
		return nil
	})

	if foundPath != "" {
		data, err := os.ReadFile(foundPath)
		if err != nil {
			return 0, err
		}
		s := strings.TrimSpace(string(data))
		return strconv.ParseUint(s, 10, 64)
	}
	if err != nil && err != io.EOF {
		return 0, fmt.Errorf("error searching for max_energy_range_uj: %v", err)
	}
	return 0, fmt.Errorf("could not find max_energy_range_uj file")
}

// readProcStat efficiently reads /proc/stat
func readProcStat(f *os.File, scanner *bufio.Scanner) (Stats, error) {
	var stats Stats
	var cpuLineFound, ctxtLineFound bool

	// Rewind the file descriptor to the beginning
	if _, err := f.Seek(0, 0); err != nil {
		return stats, fmt.Errorf("failed to seek /proc/stat: %v", err)
	}
	// Reset the scanner to read from the rewound file
	scanner = bufio.NewScanner(f)

	for scanner.Scan() {
		line := scanner.Text()

		if strings.HasPrefix(line, "cpu ") {
			fields := strings.Fields(line)
			if len(fields) < 6 {
				continue
			}
			stats.cpu.user, _ = strconv.ParseUint(fields[1], 10, 64)
			// fields[2] is "nice", we skip it
			stats.cpu.system, _ = strconv.ParseUint(fields[3], 10, 64)
			stats.cpu.idle, _ = strconv.ParseUint(fields[4], 10, 64)
			stats.cpu.iowait, _ = strconv.ParseUint(fields[5], 10, 64)
			cpuLineFound = true
		} else if strings.HasPrefix(line, "ctxt ") {
			fields := strings.Fields(line)
			if len(fields) < 2 {
				continue
			}
			stats.ctx, _ = strconv.ParseUint(fields[1], 10, 64)
			ctxtLineFound = true
		}

		if cpuLineFound && ctxtLineFound {
			break // Found both lines, stop parsing
		}
	}

	if err := scanner.Err(); err != nil {
		return stats, fmt.Errorf("scanner error: %v", err)
	}
	if !cpuLineFound || !ctxtLineFound {
		return stats, fmt.Errorf("could not parse all required lines from /proc/stat")
	}

	return stats, nil
}

// readRaplEnergy reads the energy_uj value.
// It uses ReadAt to avoid changing the file offset, which is safer.
func readRaplEnergy(f *os.File, buf []byte) (uint64, error) {
	n, err := f.ReadAt(buf, 0)
	if err != nil && err != io.EOF {
		return 0, err
	}
	// Trim whitespace (like newline)
	s := strings.TrimSpace(string(buf[:n]))
	return strconv.ParseUint(s, 10, 64)
}

// writeCSV writes the entire buffer to the output file.
func writeCSV(outfile string) error {
	f, err := os.Create(outfile)
	if err != nil {
		return fmt.Errorf("failed to create %s: %v", outfile, err)
	}
	defer f.Close()

	w := csv.NewWriter(f)
	if err := w.Write(csvHeader); err != nil {
		return fmt.Errorf("failed to write header: %v", err)
	}
	if err := w.WriteAll(dataBuffer); err != nil {
		return fmt.Errorf("failed to write data: %v", err)
	}
	w.Flush()
	return w.Error()
}
