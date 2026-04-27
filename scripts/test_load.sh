sleep 30

stress-ng --cpu 1 --cpu-load 25 --timeout 30s
sleep 10

stress-ng --cpu 2 --cpu-load 50 --timeout 30s
sleep 10

stress-ng --cpu 4 --cpu-load 75 --timeout 30s
sleep 10

stress-ng --cpu 8 --timeout 30s
sleep 10

stress-ng --hdd 2 --timeout 30s
sleep 10

stress-ng --vm 2 --vm-bytes 1G --timeout 30s
sleep 10

stress-ng --cpu 2 --hdd 1 --vm 1 --vm-bytes 512M --timeout 30s
sleep 10

stress-ng --cpu 4 --timeout 5s
sleep 3
stress-ng --cpu 1 --timeout 5s
sleep 3
stress-ng --cpu 8 --timeout 5s
sleep 10

echo "Test 10: Return to idle"
sleep 30
