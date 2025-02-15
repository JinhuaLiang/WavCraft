conda env create -f venvs/audiocraft.yml
conda env create -f venvs/audioldm.yml
conda env create -f venvs/audiosr.yml
conda env create -f venvs/wavcraft.yml
# Prepare third-party repos
# Comment some of them if they are unnecessary
mkdir ext/
cd ext/

git clone https://github.com/haoheliu/AudioLDM.git

git clone https://github.com/Audio-AGI/AudioSep.git

wget https://uplex.de/audiowmark/releases/audiowmark-0.6.1.tar.gz
tar -xzvf audiowmark-0.6.1.tar.gz
cd audiowmark-0.6.1
./configure
make
make install