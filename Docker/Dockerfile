FROM ubuntu:20.04
###############
# Preparation
###############
# Set up tzdata for non-interactive installation
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata


####################
## Install elastix
####################
# Install required packages
RUN apt-get update && apt-get -qq install wget libgomp1 -y
# Download elastix 5.0.1
RUN wget https://www.github.com/SuperElastix/elastix/releases/download/5.0.1/elastix-5.0.1-linux.tar.bz2
# Extract the files
RUN tar -xjf elastix-5.0.1-linux.tar.bz2
# Copy bins and libraries to general use location
RUN cp elastix-5.0.1-linux/bin/* /usr/local/bin/
RUN cp elastix-5.0.1-linux/lib/* /usr/lib/

# Copy license and notice
RUN cp elastix-5.0.1-linux/LICENSE /
RUN cp elastix-5.0.1-linux/NOTICE /

# Clean-up
RUN rm elastix-5.0.1-linux.tar.bz2
RUN rm -r elastix-5.0.1-linux/

###################
## Set-up python
##################
# We install python using pyenv to get the specific python version

# First install required packages
RUN apt-get update && apt-get install --no-install-recommends -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev git
# Install pyenv
RUN curl https://pyenv.run | bash
# Update the shell
RUN exec $SHELL
ENV PATH="/root/.pyenv/bin:${PATH}"
RUN eval "$(pyenv init -)"
RUN eval "$(pyenv virtualenv-init -)"

# Update pyenv to check for correct installation
RUN pyenv update

# Install latest python 3.7 version
RUN pyenv install -v 3.7.9
RUN pyenv global 3.7.9

# Add new python executable to path.
ENV PATH="/root/.pyenv/versions/3.7.9/bin:${PATH}"

# Install prognosais which has all the required packages
RUN pip3 install prognosais==0.2.5
# Correct the permissions to allow python to be used by all users
RUN chmod -R 777 /root/

###############
## We get the required data
###########

# First get the MNI atlas
RUN apt-get update && apt-get install unzip
RUN mkdir /LICENSES/
RUN wget http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_sym_09a_nifti.zip
RUN unzip -p mni_icbm152_nlin_sym_09a_nifti.zip COPYING > /LICENSES/MNI_ATLAS_LICENSE
RUN mkdir -p /DATA/ATLAS
RUN unzip -p mni_icbm152_nlin_sym_09a_nifti.zip mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii > /DATA/ATLAS/T1_atlas.nii
RUN unzip -p mni_icbm152_nlin_sym_09a_nifti.zip mni_icbm152_nlin_sym_09a/mni_icbm152_t2_tal_nlin_sym_09a.nii > /DATA/ATLAS/T2_atlas.nii

RUN gzip /DATA/ATLAS/T1_atlas.nii
RUN gzip /DATA/ATLAS/T2_atlas.nii

# Remove zip to save some spac
RUN rm mni_icbm152_nlin_sym_09a_nifti.zip

# Get the elastix parameter files
RUN mkdir -p /DATA/ELASTIX
RUN wget https://raw.githubusercontent.com/SuperElastix/ElastixModelZoo/master/models/Par0064/Par0064_rigid.txt -O /DATA/ELASTIX/parameter_map_0.txt
RUN wget https://raw.githubusercontent.com/SuperElastix/ElastixModelZoo/master/models/Par0064/Par0064_affine.txt -O /DATA/ELASTIX/parameter_map_1.txt

# Add the mask for the atlas
ADD brain_mask.nii.gz /DATA/ATLAS/

# Now add the model and configs
RUN mkdir -p /DATA/MODEL/
ADD config_prognosais_model.yaml /DATA/MODEL/
ADD custom_definitions.py /DATA/MODEL/
ADD prognosais_model /DATA/MODEL/prognosais_model/

ADD run_pipeline.sh /
RUN chmod +x /run_pipeline.sh
ADD get_predictions.py /
ADD evaluate_model.py /

# There is a problem with the python
# Because we run the script as the user
# It does not allow access for other people
# Fix this somehow

#RUN chmod 777 /root/.pyenv/versions/3.7.9/bin/python

ENTRYPOINT [ "/run_pipeline.sh" ]



