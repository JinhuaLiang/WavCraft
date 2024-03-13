conda activate AudioEditor
source ./scripts/chatgpt.sh

mkdir -p services_logs

export SERVICE_PORT=8088
export SERVICE_URL=127.0.0.1
export MAX_SCRIPT_LINES=999

export AUDIOCRAFT_SERVICE_PORT=$((${SERVICE_PORT}+1))
export AUDIOSEP_SERVICE_PORT=$((${SERVICE_PORT}+2))
export AUDIOSR_SERVICE_PORT=$((${SERVICE_PORT}+3))
export AUDIOLDM_SERVICE_PORT=$((${SERVICE_PORT}+4))
export WAVMARK_SERVICE_PORT=$((${SERVICE_PORT}+5))
