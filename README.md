# PrognosAIs_glioma

This repository contains the code and trained models required for the running of the PrognosAIs model trained for the prediction of the IDH mutation status, 1p/19q co-deletion status, and grade of glioma based on pre-operative MRI scans. The model also automatically segments the tumor.

## Running the model 


### Docker
The model can be run using the pre-built Docker which contains all code and data needed to apply the model and will carry out all the required pre-processing.
The docker can be run as follows: 

`docker run -u $UID:$GROUPS -v "<local_input_folder>:/input/" -v "<local_output_folder>:/output/" svdvoort/prognosais_glioma:1.0.2`

Here `<local_input_folder>` needs to be replaced by the path to the folder on the host machine that contains the scans for the different subjects.
`<local_output_folder>` needs to be replaced by the folder on the host machine in which the results should be stored.


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

The outputs of the model are then saved in `<local_output_folder>/Results`. A mask is stored for each patient as `<subjec_id>_mask.nii.gz` and the results of the genetic and histological feature predictions are all stored in `genetic_histological_predictions.csv`.

### Locally

You can also evaluate the model by locally installing it. This requires you to do your own pre-processing according to the article: [https://doi.org/10.1016/j.dib.2021.107191](https://doi.org/10.1016/j.dib.2021.107191). 
The easiest way to set-up the pipeline locally is to follow the same steps as provided in [the docker file](https://github.com/Svdvoort/PrognosAIs_glioma/blob/master/Docker/Dockerfile). 
You can then run [the pipeline script](https://github.com/Svdvoort/PrognosAIs_glioma/blob/master/Docker/run_pipeline.sh) to evaluate the model. 


## Model

If you are just interested in the model it is available in the `Trained_models` folder. 

The model is compressed unto a tar archive. To restore the model:

```
cd Trained_models
tar -xzvf prognosais_model.tar.gz
```

The model is now stored, in [TensorFlow SavedModel format](https://www.tensorflow.org/guide/saved_model), in `the prognosais_model` folder.



