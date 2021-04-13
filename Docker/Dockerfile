FROM ubuntu:20.04


####################
## Install elastix & registration data
####################
# Install required packages
RUN apt-get update \
    && apt-get install wget libgomp1 bzip2 -y \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /LICENSES/ELASTIX

# Download elastix and install 5.0.1
RUN wget https://www.github.com/SuperElastix/elastix/releases/download/5.0.1/elastix-5.0.1-linux.tar.bz2 \
    && tar --strip=2 -C /usr/local/bin -xjf elastix-5.0.1-linux.tar.bz2 elastix-5.0.1-linux/bin/elastix elastix-5.0.1-linux/bin/transformix \
    && tar --strip=2 -C /usr/lib/ -xjf elastix-5.0.1-linux.tar.bz2 elastix-5.0.1-linux/lib/libANNlib-5.0.so.1 elastix-5.0.1-linux/lib/libANNlib-5.0.so \
    && tar --strip=1 -C /LICENSES/ELASTIX -xjf elastix-5.0.1-linux.tar.bz2 elastix-5.0.1-linux/LICENSE elastix-5.0.1-linux/NOTICE\
    && rm /elastix-5.0.1-linux.tar.bz2

# Get the parameter maps

RUN mkdir -p /DATA/ELASTIX \
    && wget https://raw.githubusercontent.com/SuperElastix/ElastixModelZoo/master/models/Par0064/Par0064_rigid.txt -O /DATA/ELASTIX/parameter_map_0.txt \
    && wget https://raw.githubusercontent.com/SuperElastix/ElastixModelZoo/master/models/Par0064/Par0064_affine.txt -O /DATA/ELASTIX/parameter_map_1.txt

###############
## Download atlas data
###############

# install required packages
RUN apt-get update \
    && apt-get install unzip -y \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /DATA/ATLAS

RUN wget http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09a_nifti.zip \
    && unzip -p mni_icbm152_nlin_sym_09a_nifti.zip COPYING > /LICENSES/MNI_ATLAS_LICENSE \
    && unzip -p mni_icbm152_nlin_sym_09a_nifti.zip mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii > /DATA/ATLAS/T1_atlas.nii \
    && unzip -p mni_icbm152_nlin_sym_09a_nifti.zip mni_icbm152_nlin_sym_09a/mni_icbm152_t2_tal_nlin_sym_09a.nii > /DATA/ATLAS/T2_atlas.nii \
    && gzip /DATA/ATLAS/T1_atlas.nii \
    && gzip /DATA/ATLAS/T2_atlas.nii \
    && rm mni_icbm152_nlin_sym_09a_nifti.zip

###############
## Install python and required packages
###############

## Install python and pip
RUN apt-get update \
    && DEBIAN_FRONTEND="noninteractive" apt-get install --no-install-recommends -y \
    build-essential\
    curl\
    libssl-dev\
    libbz2-dev\
    libffi-dev\
    libncurses5-dev\
    libncursesw5-dev\
    libreadline-dev\
    libsqlite3-dev\
    llvm\
    make\
    openssl\
    python3-distutils \
    tk-dev \
    wget\
    xz-utils\
    zlib1g-dev\
    && rm -rf /var/lib/apt/lists/*

RUN wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz \
    && tar -xzvf Python-3.7.9.tgz \
    && rm Python-3.7.9.tgz \
    && cd Python-3.7.9 \
    && ./configure \
    && make \
    && make install \
    && cd / \
    && rm -rf Python-3.7.9/


# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py \
    && python3 get-pip.py

# Install prognosais

RUN pip3 install --no-cache-dir prognosais==0.2.7

###############
## Set-up from repository
###############

RUN apt-get update \
    && DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    git\
    && rm -rf /var/lib/apt/lists/*


RUN git clone https://github.com/Svdvoort/PrognosAIs_glioma.git \
    && cd PrognosAIs_glioma \
    && mv Data/brain_mask.nii.gz /DATA/ATLAS/ \
    && mkdir -p /DATA/MODEL/ \
    && mv Data/config_prognosais_model.yaml /DATA/MODEL/ \
    && mv Scripts/custom_definitions.py /DATA/MODEL/\
    && cd Trained_models \
    && cat prognosais_model.tar.gz.part* > prognosais_model.tar.gz \
    && tar -xzvf prognosais_model.tar.gz \
    && cp -r prognosais_model/  /DATA/MODEL/prognosais_model/ \
    && rm prognosais_model.tar.gz.part* \
    && rm -r prognosais_model/ \
    && cd .. \
    && mv Docker/run_pipeline.sh / \
    && chmod +x /run_pipeline.sh \
    && mv Scripts/get_predictions.py / \
    && mv Scripts/evaluate_model.py /

ENTRYPOINT [ "/run_pipeline.sh" ]



