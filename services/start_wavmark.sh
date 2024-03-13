export WAVMARK_SERVICE_PORT=$((${SERVICE_PORT}+5))

conda activate AudioEditor
nohup python3 services/wavmark_service.py > services_logs/wavmark_service.out 2>&1 &
echo "WavMark is loaded sucessfully."