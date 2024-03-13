import os


# Extract values for each application
audiocraft_service_port = int(os.environ.get('AUDIOCRAFT_SERVICE_PORT'))
audiosep_service_port = int(os.environ.get('AUDIOSEP_SERVICE_PORT'))
audiosr_service_port = int(os.environ.get('AUDIOSR_SERVICE_PORT'))
audioldm_service_port = int(os.environ.get('AUDIOLDM_SERVICE_PORT'))
wavmark_service_port = int(os.environ.get('WAVMARK_SERVICE_PORT'))

# Execute the commands 
os.system(f'kill $(lsof -t -i :{audiocraft_service_port})')
os.system(f'kill $(lsof -t -i :{audiosep_service_port})')
os.system(f'kill $(lsof -t -i :{audiosr_service_port})')
os.system(f'kill $(lsof -t -i :{audioldm_service_port})')
os.system(f'kill $(lsof -t -i :{wavmark_service_port})')