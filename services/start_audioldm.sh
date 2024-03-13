export AUDIOLDM_SERVICE_PORT=$((${SERVICE_PORT}+4))

conda activate AudioInpainting
nohup python3 services/audioldm_service.py > services_logs/audioldm.out 2>&1 &
echo "AudioLDM is loaded sucessfully."