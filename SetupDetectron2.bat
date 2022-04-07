REM Create a virtual environment for detectron2
:: conda create -n detectron_env python=3.8
:: conda activate detectron_env

REM Install prerequisite libraries
conda install pytorch torchvision torchaudio cudatoolkit -c pytorch
pip install cython
pip install opencv-python

REM Install detectron2
git clone https://github.com//facebookresearch/detectron2.git
cd detectron2
pip install -e .


