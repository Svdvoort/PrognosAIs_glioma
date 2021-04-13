from PrognosAIs import Pipeline
import evaluate_model
from PrognosAIs.IO import ConfigLoader
import argparse
from slurmpie import slurmpie
import os
import sys
from glob import glob

model = sys.argv[1]
input_folder = sys.argv[2]
output_folder = sys.argv[3]
config_file = sys.argv[4]

# make label file, stupid requirement
label_file = os.path.join(output_folder, "labels.txt")
print(label_file)
patients = glob(os.path.join(input_folder, "*"))

with open(label_file, "w") as the_label_file:
    the_label_file.write("Patient\tMASK\tIDH\t1p19q\tGrade\n")
    for i_patient in patients:
        patient_name = os.path.basename(i_patient)
        the_label_file.write(patient_name + "\t" + "\t".join(["1"]*4) + "\n")

# Adjust the config

# actually run stuff
pipeline = Pipeline.Pipeline(config_file, True, False, False)
pipeline.input_folder = input_folder
pipeline.output_folder = output_folder

pipeline.start_local_pipeline()

samples_folder = pipeline.samples_folder

evaluator = evaluate_model.Evaluator(model, samples_folder, config_file, output_folder)
evaluator.evaluate()
