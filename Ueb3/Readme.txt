How to install OCR-Text.
apt uninstall libopencv
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir contrib && cd contrib
git clone https://github.com/opencv/opencv_contrib.git
apt install tesseract-ocr (And maybe others depending on what tesseract in contrib will need)
cd ..
mkdir build
cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=contrib/modules .. > output.txt
cat output | grep tesseract (There has to be a tesseract: Yes. if not, see what it needs and install)
make -j4
sudo make install

