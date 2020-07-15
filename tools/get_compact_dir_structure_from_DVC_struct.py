##
import os
import glob
import csv
import shutil

## Define input directories
BASE_DIR =  '/home/petteri/Dropbox/manuscriptDrafts/CThemorr/DATA_DVC/sCROMIS2ICH'
BASE_CT = os.path.join(BASE_DIR, 'CT', 'labeled', 'MNI_1mm_256vx-3D') # isotropic 1mm
IMG_DIR = os.path.join(BASE_CT, 'data', 'BM4D_nonNaN_-100') # low-range clipped to -100 HU
LABEL_BASE = os.path.join(BASE_CT, 'labels', 'voxel') # all the different binary masks
LABEL_DIR = os.path.join(LABEL_BASE, 'hematomaAll')
SPLIT_BASE = os.path.join(BASE_DIR, 'data_splits')
SPLIT_CV_FOLDS = os.path.join(SPLIT_BASE, 'cv_splits_balanced')
WILDCARD = '*.nii*'

if not os.path.exists(IMG_DIR ):
    raise FileNotFoundError('Your IMG_DIR  = "{}" is not found'.format(IMG_DIR ))
if not os.path.exists(LABEL_DIR):
    raise FileNotFoundError('Your LABEL_DIR = "{}" is not found'.format(LABEL_DIR))

# Output directories
OUTPUT_DIR = '/home/petteri/headCT_dataset'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

OUTPUT_IMG_TRAIN = os.path.join(OUTPUT_DIR, 'imagesTr')
OUTPUT_LABEL_TRAIN = os.path.join(OUTPUT_DIR, 'labelsTr')
OUTPUT_IMG_VAL = os.path.join(OUTPUT_DIR, 'imagesVal')
OUTPUT_LABEL_VAL = os.path.join(OUTPUT_DIR, 'labelsVal')
OUTPUT_IMG_TEST = os.path.join(OUTPUT_DIR, 'imagesTs')
OUTPUT_LABEL_TEST = os.path.join(OUTPUT_DIR, 'labelsTs')
if not os.path.exists(OUTPUT_IMG_TRAIN):
    os.makedirs(OUTPUT_IMG_TRAIN)
if not os.path.exists(OUTPUT_LABEL_TRAIN):
    os.makedirs(OUTPUT_LABEL_TRAIN)
if not os.path.exists(OUTPUT_IMG_VAL):
    os.makedirs(OUTPUT_IMG_VAL)
if not os.path.exists(OUTPUT_LABEL_VAL):
    os.makedirs(OUTPUT_LABEL_VAL)
if not os.path.exists(OUTPUT_IMG_TEST):
    os.makedirs(OUTPUT_IMG_TEST)
if not os.path.exists(OUTPUT_LABEL_TEST):
    os.makedirs(OUTPUT_LABEL_TEST)
# TODO dataset.json if you want to make this compatible with prev format
#  see "write_json_nnUnet()" from "sCROMIS2ICH_loaders.py"

##
def get_input_files(IMG_DIR, LABEL_DIR, WILDCARD):
    image_files = glob.glob(os.path.join(IMG_DIR, WILDCARD))
    if len(image_files) == 0:
        raise FileNotFoundError('No image files found with wildcard "{}" from {} '.format(WILDCARD,IMG_DIR))
    label_files = glob.glob(os.path.join(LABEL_DIR, WILDCARD))
    if len(label_files) == 0:
        raise FileNotFoundError('No label files found with wildcard "{}"  from {}'.format(WILDCARD.LABEL_DIR))
    return image_files, label_files

def get_files_from_csv(csv_path):
    with open(csv_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        codes = [tuple(row) for row in reader][0]
    return codes

def get_predefined_splits(SPLIT_BASE, SPLIT_CV_FOLDS, WILDCARD,
                          test_files_csv = 'sCROMIS2ICH_n221_test.csv',
                          train_files_csv = 'sCROMIS2ICH_n221_train_fold-01.csv',
                          val_files_csv = 'sCROMIS2ICH_n221_val_fold-01.csv'):
    # TODO! now hard-coded to fold 1
    test_filecodes = get_files_from_csv(csv_path = os.path.join(SPLIT_BASE, test_files_csv))
    train_filecodes = get_files_from_csv(csv_path = os.path.join(SPLIT_CV_FOLDS, train_files_csv))
    val_filecodes = get_files_from_csv(csv_path = os.path.join(SPLIT_CV_FOLDS, val_files_csv))
    return test_filecodes, train_filecodes, val_filecodes

def copy_files(input_files, filecodes, output_dir):
    files_matching_codes = []
    for code in filecodes:
        fpath = [s for s in input_files if code in s]
        files_matching_codes.append(fpath)
    for fpath_to_copy in files_matching_codes:
        fname = os.path.split(fpath_to_copy[0])[1]
        shutil.copy(fpath_to_copy[0], os.path.join(output_dir, fname))

def copy_wanted_codes_to_output(test_filecodes, train_filecodes, val_filecodes,
                                input_files, output_train, output_val, output_test):
    copy_files(input_files, filecodes = train_filecodes, output_dir = output_train)
    copy_files(input_files, filecodes = val_filecodes, output_dir = output_val)
    copy_files(input_files, filecodes = test_filecodes, output_dir = output_test)

##
def main():

    # get files (images and labels)
    image_files, label_files = get_input_files(IMG_DIR, LABEL_DIR, WILDCARD)

    # get what files need to be in which split
    test_filecodes, train_filecodes, val_filecodes = get_predefined_splits(SPLIT_BASE, SPLIT_CV_FOLDS, WILDCARD)

    # Images
    print('Copying images to output dir')
    copy_wanted_codes_to_output(test_filecodes, train_filecodes, val_filecodes,
                                input_files = image_files,
                                output_train = OUTPUT_IMG_TRAIN,
                                output_val = OUTPUT_IMG_VAL,
                                output_test = OUTPUT_IMG_TEST)

    # Labels
    print('Copying labels to output dir')
    copy_wanted_codes_to_output(test_filecodes, train_filecodes, val_filecodes,
                                input_files = label_files,
                                output_train = OUTPUT_LABEL_TRAIN,
                                output_val = OUTPUT_LABEL_VAL,
                                output_test = OUTPUT_LABEL_TEST)

###
if __name__ == "__main__":
    main()


