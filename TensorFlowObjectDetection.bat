:: Create a new virtual environment
python -m venv tfod

:: Activate your virtual environment
.\tfod\Scripts\activate

:: Install dependencies and add virtual environment to the Python Kernel
python -m pip install --upgrade pip
pip install ipykernel
python -m ipykernel install --user --name=tfodj

