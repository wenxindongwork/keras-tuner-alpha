sudo sed -i 's/#$nrconf{restart} = '"'"'i'"'"';/$nrconf{restart} = '"'"'a'"'"';/g' /etc/needrestart/needrestart.conf
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.11
sudo apt-get install -y python3-pip python-is-python3
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
python -m pip install --upgrade pip

pip install -U kithara[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --extra-index-url  https://download.pytorch.org/whl/cpu 
