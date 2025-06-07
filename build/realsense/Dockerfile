FROM ml

RUN <<EOF
set -xue

pip install \
    pyrealsense2==2.55.1.6486 \
    ultralytics==8.3.54 \
    catboost==1.2.7 \
    lightgbm==4.5.0 \
    xgboost \
    lime\
    shap\
    ipython\
    toml\
    torch-geometric\
    pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

EOF

RUN apt-get update && apt-get install -y libusb-1.0-0 && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
