mkdir -p services_logs

export SERVICE_PORT=8088
export SERVICE_URL=127.0.0.1
export MAX_SCRIPT_LINES=999

# Start AudioCraft service
source services/start_audiocraft.sh
# Start AudioSep service
source services/start_audiosep.sh
# Start AudioSR service
source services/start_audiosr.sh
# Start AudioLDM service
source services/start_audioldm.sh
# Start WavMark service
source services/start_wavmark.sh
# WavCraft
conda activate WavCraft