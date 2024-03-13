export AUDIOCRAFT_SERVICE_PORT=$((${SERVICE_PORT}+1))

conda activate AudioCraft
nohup python3 services/audiocraft_service.py > services_logs/audiocraft.out 2>&1 &
echo "AudioCraft is loaded sucessfully."