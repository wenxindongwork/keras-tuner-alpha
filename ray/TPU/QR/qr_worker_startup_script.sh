#! /bin/bash


# Function to check and release dpkg lock
check_dpkg_lock() {
    if sudo fuser -s /var/lib/dpkg/lock-frontend ; then
      sudo systemctl stop unattended-upgrades
    fi
}

check_dpkg_lock
sudo sed -i 's/#$nrconf{restart} = '"'"'i'"'"';/$nrconf{restart} = '"'"'a'"'"';/g' /etc/needrestart/needrestart.conf

check_dpkg_lock
sudo add-apt-repository -y ppa:deadsnakes/ppa

check_dpkg_lock
sudo apt-get update

check_dpkg_lock
sudo apt-get install -y python3.11

check_dpkg_lock
sudo apt-get install -y python3-pip python-is-python3

check_dpkg_lock
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

check_dpkg_lock
sudo python -m pip install --upgrade pip

check_dpkg_lock
sudo pip install -U kithara[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --extra-index-url  https://download.pytorch.org/whl/cpu 
