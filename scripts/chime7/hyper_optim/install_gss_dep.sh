
NEMO_PATH=$1
pip install espnet 
git clone https://github.com/espnet/espnet.git /workspace/espnet 
pip uninstall -y 'cupy-cuda118' 
pip install --no-cache-dir -f https://pip.cupy.dev/pre/ "cupy-cuda11x[all]==12.1.0" 
pip install git+http://github.com/desh2608/gss 
pip install optuna 
pip install lhotse==1.14.0 
pip install --upgrade jiwer 
./ngc_install_lm.sh ${NEMO_PATH}