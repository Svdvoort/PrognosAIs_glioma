# PrognosAIs_glioma

This repository contains the code and trained models required for the running of the PrognosAIs model trained for the prediction of the IDH mutation status, 1p/19q co-deletion status, and grade of glioma based on pre-operative MRI scans. The model also automatically segments the tumor.

The model and work that went into the development of the model are described in the paper:

Sebastian R van der Voort, Fatih Incekara, Maarten M J Wijnenga, Georgios Kapsas, Renske Gahrmann, Joost W Schouten, Rishi Nandoe Tewarie, Geert J Lycklama, Philip C De Witt Hamer, Roelant S Eijgelaar, Pim J French, Hendrikus J Dubbink, Arnaud J P E Vincent, Wiro J Niessen, Martin J van den Bent, Marion Smits, Stefan Klein, Combined molecular subtyping, grading, and segmentation of glioma using multi-task deep learning, Neuro-Oncology, 2022;, noac166, [https://doi.org/10.1093/neuonc/noac166](https://doi.org/10.1093/neuonc/noac166).

Please cite this paper when you use this code.

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

The model is now stored,in [TensorFlow SavedModel format](https://www.tensorflow.org/guide/saved_model), in `the prognosais_model` folder.

# FAQ

**I get an error like: `mv: cannot stat '/output/Predictions/Results': No such file or directory`**
The end of the log probably looks something like this: 

```text
Traceback (most recent call last):
  File "/usr/local/lib/python3.7/site-packages/pandas/core/indexes/base.py", line 3080, in get_loc
    return self._engine.get_loc(casted_key)
  File "pandas/_libs/index.pyx", line 70, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/index.pyx", line 101, in pandas._libs.index.IndexEngine.get_loc
  File "pandas/_libs/hashtable_class_helper.pxi", line 1625, in pandas._libs.hashtable.Int64HashTable.get_item
  File "pandas/_libs/hashtable_class_helper.pxi", line 1632, in pandas._libs.hashtable.Int64HashTable.get_item
KeyError: 0

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/get_predictions.py", line 33, in <module>
    pipeline.start_local_pipeline()
  File "/usr/local/lib/python3.7/site-packages/PrognosAIs/Pipeline.py", line 65, in start_local_pipeline
    self.input_folder, self.output_folder, self.preprocessing_config
  File "/usr/local/lib/python3.7/site-packages/PrognosAIs/Preprocessing/Preprocessors.py", line 999, in __init__
    make_one_hot=self.labeling_config.make_one_hot,
  File "/usr/local/lib/python3.7/site-packages/PrognosAIs/IO/LabelParser.py", line 45, in __init__
    self.encode_labels_one_hot()
  File "/usr/local/lib/python3.7/site-packages/PrognosAIs/IO/LabelParser.py", line 180, in encode_labels_one_hot
    category_type = self.get_label_category_type(i_label_category)
  File "/usr/local/lib/python3.7/site-packages/PrognosAIs/IO/LabelParser.py", line 147, in get_label_category_type
    category_label_type = type(self.label_data[category_name][0])
  File "/usr/local/lib/python3.7/site-packages/pandas/core/series.py", line 853, in __getitem__
    return self._get_value(key)
  File "/usr/local/lib/python3.7/site-packages/pandas/core/series.py", line 961, in _get_value
    loc = self.index.get_loc(label)
  File "/usr/local/lib/python3.7/site-packages/pandas/core/indexes/base.py", line 3082, in get_loc
    raise KeyError(key) from err
KeyError: 0
mv: cannot stat '/output/Predictions/Results': No such file or directory
rm: cannot remove '/output/Predictions/Samples': No such file or directory
rm: cannot remove '/output/Predictions/config.yml': No such file or directory
```

If this is the case, you should check the names of your patient folders. Due to a bug in the code, patient folder names should be strings, otherwise this error can occur. 
Concretely this means that a patient folder should always have a letter in its name. 
For example do not create a patient folder with the name "808310", but then rename it to "P808310". 
The docker should now run without problems, if not please open an issue. 

**I get en error like: `/run_pipeline.sh: line 51: 276 Illegal instruction` when running the docker**

If you are trying to run the docker on a newer Mac with an M1/M2/M3 etc. chip this might be the cause. See [this issue](https://github.com/Svdvoort/PrognosAIs_glioma/issues/4). The best approach is to run the model locally instead of using Docker in this case. If you run into this error but are not running on a Mac, feel free to open a new issue. 

**How should I interpret the file with the genetic predictions?**

- Prediction_IDH_class_0 is the prediction score for IDH wildtype
- Prediction_IDH_class_1 is the prediction score for IDH mutated
- Prediction_1p19q_class_0 is the prediction score for 1p19q not-co-deleted
- Prediction_1p19q_class_1 is the prediction score for 1p19q co-deleted
- Prediction_grade_class_0 is the prediction score for grade II
- Prediction_grade_class_1 is the prediction score for grade III
- Prediction_grade_class_3 is the prediction score for grade IV







