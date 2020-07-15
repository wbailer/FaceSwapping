FROM tensorflow/tensorflow:latest-py3

RUN apt-get update
RUN apt-get -y install git-all

RUN pip install scipy
RUN pip install pillow
RUN pip install requests
RUN pip install tqdm
RUN pip install opencv-python

RUN cd /notebooks 
RUN git clone https://github.com/carpedm20/DCGAN-tensorflow.git
WORKDIR /notebooks/DCGAN-tensorflow

RUN python download.py mnist celebA

RUN mkdir mnt

#RUN python main.py --dataset celebA --input_height=108 --train --crop

CMD /bin/bash
