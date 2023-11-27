## test on python 3.10.0
pip install pip -U
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install TTS
pip install onnxruntime
pip install scenedetect
pip install opencv-python
pip install ultralytics
pip install tensorflow==2.8.0
pip install deepface
pip install numpy==1.22.2
pip install transformers
pip install pydub
pip install basicsr
pip install facexlib
pip install gfpgan
pip install audiostretchy
pip install numpy==1.22.2
pip install ctranslate2
pip install faster_whisper
pip install ffmpeg_python==0.2.0
pip install pyannote.audio
pip install moviepy
pip install cutlet
pip install unidic-lite
pip install face-alignment==1.3.4
pip install ninja==1.10.2.3
pip install dlib --verbose
# pip install librosa==0.9.2

apt update
apt install ffmpeg

# video-retalking ディレクトリに移動
cd video-retalking

# チェックポイント用のディレクトリを作成
mkdir -p ./checkpoints

# 必要なファイルをダウンロード
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/30_net_gen.pth -O ./checkpoints/30_net_gen.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/BFM.zip -O ./checkpoints/BFM.zip
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/DNet.pt -O ./checkpoints/DNet.pt
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ENet.pth -O ./checkpoints/ENet.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/expression.mat -O ./checkpoints/expression.mat
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/face3d_pretrain_epoch_20.pth -O ./checkpoints/face3d_pretrain_epoch_20.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GFPGANv1.3.pth -O ./checkpoints/GFPGANv1.3.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/GPEN-BFR-512.pth -O ./checkpoints/GPEN-BFR-512.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/LNet.pth -O ./checkpoints/LNet.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/ParseNet-latest.pth -O ./checkpoints/ParseNet-latest.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/RetinaFace-R50.pth -O ./checkpoints/RetinaFace-R50.pth
wget https://github.com/vinthony/video-retalking/releases/download/v0.0.1/shape_predictor_68_face_landmarks.dat -O ./checkpoints/shape_predictor_68_face_landmarks.dat

# BFM.zipを解凍
unzip -d ./checkpoints/BFM ./checkpoints/BFM.zip
