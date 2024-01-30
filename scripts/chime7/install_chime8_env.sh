pip install espnet
git clone https://github.com/espnet/espnet.git /workspace/espnet
pip install --no-cache-dir -f https://pip.cupy.dev/pre/ "cupy-cuda11x[all]==12.3.0"
pip install git+http://github.com/desh2608/gss
pip install optuna
pip install lhotse==1.14.0
pip install --upgrade jiwer
pip install git+https://github.com/kpu/kenlm
sh ./run_install_lm.sh ${PWD}/../../../NeMo