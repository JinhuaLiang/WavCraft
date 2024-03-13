export AUDIOSR_SERVICE_PORT=$((${SERVICE_PORT}+3))

conda activate AudioSR
nohup python3 services/audiosr_service.py > services_logs/audiosr.out 2>&1 &
echo "AudioSR is loaded sucessfully."