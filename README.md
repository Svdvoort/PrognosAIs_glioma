# PrognosAIs_glioma

This repository contains the code and trained models required for the running of the PrognosAIs model trained for the prediction of the IDH mutation status, 1p/19q co-deletion status, and grade of glioma based on pre-operative MRI scans. The model also automatically segments the tumor.

## Running the model 


### Docker
The model can be run using the pre-built Docker which contains all code and data needed to apply the model and will carry out all the required pre-processing.
The docker can be run as follows: 

`docker run -u $UID:$GROUPS -v "<local_input_folder>:/input/" -v "<local_output_folder>:/output/" svdvoort/prognosais-glioma:0.1`

Here `<local_input_folder>` needs to be replaced by the path to the folder on the host machine that contains the scans for the different subjects.
`<local_output_folder` needs to be replaced by the folder on the host machine in which the results should be stored.


`<local_input_folder>` should contain one folder per subject, with for each subject the NIFTI files for the pre-constrast T1-weighted scan, post-contrast T1-weighted scan, T2-weighted scan, and T2-weighted FLAIR scan named T1.nii.gz, T1GD.nii.gz, T2.nii.gz and FLAIR.nii.gz respectively.

Thus an example of a structure for two subjects would be: 

```
<local_input_folder>
|
|  Subject-001
|  |  FLAIR.nii.gz
|  |  T1.nii.gz
|  |  T1GD.nii.gz
|  |  T2.nii.gz
|
|  Subject-002
|  |  FLAIR.nii.gz
|  |  T1.nii.gz
|  |  T1GD.nii.gz
|  |  T2.nii.gz
```

### Locally

You can also evaluate the model by locally installing it. This requires you to do your own pre-processing according to the article: XXXX (link to follow later). 
You can also check out the Docker file to check out the pre-processing steps. 

Evaluating the model locally requires the installation of Python 3.7 and the [PrognosAIs framework](https://github.com/Svdvoort/prognosais):

```
python3 -m venv prognosais
source prognosais
pip install prognosais==0.2.5
```

Get the repository locally and extract the model: 

```
git clone https://github.com/Svdvoort/PrognosAIs_glioma
cd PrognosAIs_glioma/Trained_models
cat prognosais_model.tar.gz.part* > prognosais_model.tar.gz
tar -xzvf prognosais_model.tar.gz
```

Create a local folder from which to run the model:

```
mkdir ~/prognosais
cp -r prognosais_model/ ~/prognosais
cp config_prognosais_model.yaml ~/prognosais
cd ../Docker 
cp custom_definitions.py ~/prognosais
cp evaluate_model.py ~/prognosais
cp get_predictions.py ~/prognosais
cd ../Data
cp brain_mask.nii.gz ~/prognosais
```

Now open the `config_prognosais_model.yaml` file and replace all /DATA/\*/ occurences by your home directory.
Replace the /input/ and /output/ occurences by the paths to the input and output folder. 

You can now run the model (on the pre-processed data) using:

`python get_predictions.py ~/prognosais_model/ <input_folder> <output_folder> ~/config_prognosais_model.yaml`

Where `<input_folder>` and `<output_folder>` have to be replaced by the input folder and output folder respectively.




