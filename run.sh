docker run --gpus all -p 5001:5000 -p 6001:6006 -p $(cat port.txt):8888 -it -v ${PWD}:/usr/local/bin/jpl_config:rw --name=$(cat container_name.txt) $(cat image_name.txt)
