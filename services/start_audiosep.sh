export AUDIOSEP_SERVICE_PORT=$((${SERVICE_PORT}+2))

conda activate AudioEditor
nohup python3 services/audiosep_service.py > services_logs/audiosep.out 2>&1 &
echo "AudioSep is loaded sucessfully."