Cuda Linux
 (instala pelo site)
 
 yum install kernel-devel hernel-header
 yum install cuda
 yum install cudo-toolkit
 yum install freeglut-devel libX11-devel libXi-devel libXmu-devel make mesa-libGLU-devel

export PATH=/usr/local/cuda/bin:/usr/local/cuda/nsight-compute-2019.5.0${PATH:+:${PATH}} 


__UBUNTU__
Instalação sem atualização, simples e sem drivers de terceiros

apt install gcc make git
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo ubuntu-drivers devices
sudo apt install nvidia-driver-440
sudo ubuntu-drivers autoinstall
reboot

nvida-smi
OK

sudo sh cuda_10.2.89_440.33.01_linux.run (Sem o Driver)
nano /etc/ld.so.conf
+ include /usr/local/cuda-10.2/lib64

nano ~/.profile
+ export PATH=$PATH:/usr/local/cuda-10.2/bin
source ~/.profile

cd ~/Documentos
git clone http://gitlab.jacobiano.ddnsgeek.com:8100/tiago/tcc_2020
cd ~/Documentos/tcc_2020/Mestrado

sudo apt install freeglut3-dev

nvcc transitorio.cu -o transitorio -lglut -lGLU -lGL -lm -D LINUX -Xcompiler -fopenmp -lgomp

sudo app install python3 pytjhon3-pip
sudo pip3 install putils

python3 simula_linux.py

__FIM__

source ~/.profile

wget http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/

cuda_10.2.89_440.33.01_linux.ru

apt install gcc make git

nvcc transitorio.cu -o transitorio -lglut -lGLU -lGL -lm -D LINUX -Xcompiler -fopenmp -lgomp

ln -sf libnvidia-fbc.so.340.108 /usr/lib64/libnvidia-fbc.so.1

'libnvidia-fbc.so.440.33.01

ln -sf libnvidia-opencl.so.340.108 /usr/lib64/libnvidia-opencl.so.1

/usr/local/cuda-6.5/bin/nvcc transitorio.cu -o transitorio -lglut -lGLU -lGL -lm -D LINUX -Xcompiler -fopenmp -lgomp
