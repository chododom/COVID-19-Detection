"""
This file contains a script which transforms the original COVIDx datasets into a version fit for use with Keras generators.
By running the script with the argument x3, the COVIDx3 will be built, argument x7 specifies COVIDx7B and no argument
defaults to COVIDx8B, which is the newest version.

Author: Dominik Chodounský
Institution: Faculty of Information Technology, Czech Technical University in Prague
Last edit: 2021-03-31
"""


import os
import sys
import shutil

# File paths for COVIDx8B (will be used by default)
train_labels_path_x8 = '../../data/labels/train_COVIDx8B.txt'
test_labels_path_x8 = '../../data/labels/test_COVIDx8B.txt'
src_data_folder_x8 = '../../data/original_COVIDx8B'
target_data_folder_x8 = '../../data/COVIDx8B'

# File paths for COVIDx7B (will be used if script is run with argument 'x7')
train_labels_path_x7 = '../../data/labels/train_COVIDx7B.txt'
test_labels_path_x7 = '../../data/labels/test_COVIDx7B.txt'
src_data_folder_x7 = '../../data/original_COVIDx7B'
target_data_folder_x7 = '../../data/COVIDx7B'

# File paths for COVIDx3 (will be used if script is run with argument 'x3')
train_labels_path_x3 = '../../data/labels/train_COVIDx3.txt'
test_labels_path_x3 = '../../data/labels/test_COVIDx3.txt'
src_data_folder_x3 = '../../data/original_COVIDx3'
target_data_folder_x3 = '../../data/COVIDx3'

# Function definitions

def create_fs(data_folder):
    """
    Creates file structure for the target dataset, which includes train and test subdirectories, 
    which both include two subdirectories for the data classes 'positive' and 'negative'.
    
    Parameters
    ----------
    data_folder : str
        Path to target data folder, where the file structure will be created.
    """
    
    if os.path.isdir(data_folder):
        if not os.listdir(data_folder):
            os.makedirs(os.path.join(data_folder, 'train', 'positive'))
            os.makedirs(os.path.join(data_folder, 'train', 'negative'))

            os.makedirs(os.path.join(data_folder, 'test', 'positive'))
            os.makedirs(os.path.join(data_folder, 'test', 'negative'))
        else:
            raise FileExistsError(f'Target directory "{data_folder}" is not empty')
    else:
        raise NotADirectoryError(f'Target directory "{data_folder}" does not exist')
        

def label_data_x8(target_data_folder, src_data_folder, labels_file_path):
    """
    Takes data from one folder and separates it into class subdirectories in the target data folder. 
    This separation is based on class labels specified in a given file.
    
    Note: This function is specific for the COVIDx8B, as it contains some systematic filename errors that must be corrected.
    
    Parameters
    ----------
    target_data_folder : str
        Path to target data folder, where the files will be copied to and sorted by their labels.
    src_data_folder : str
        Path to source folder containing the original data which is to be copied and labeled.
    labels_file_path : str
        Path to text file, which contains labels for each file. The file should have a line entry for each file and syntax is such that parameters of the file are separated
        by spaces and the second from last parameter is the class label and third from last parameter is file name.
    """
    
    labels_file = open(labels_file_path, 'r')
    count = 0

    while True:
        count += 1
        line = labels_file.readline()
        
        # if line is empty -> end of file reached
        if not line:
            break
        
        split = line.split(' ')
        if split[-2] == 'positive' or split[-2] == 'negative':
            if split[-3].startswith('COVID('):
                name = split[-3].replace('(','-')
                name = name.replace(')','')
                shutil.copy(os.path.join(src_data_folder, name), os.path.join(target_data_folder, split[-2], name))
        
            else:
                shutil.copy(os.path.join(src_data_folder, split[-3]), os.path.join(target_data_folder, split[-2], split[-3]))
        else:
            raise ValueError(f'Label must be either positive or negative, it was: {split[-2]}')
            

def label_data_x7(target_data_folder, src_data_folder, labels_file_path):
    """
    Takes data from one folder and separates it into class subdirectories in the target data folder. 
    This separation is based on class labels specified in a given file.
    
    Note: This function is specific for the COVIDx7B, as it is a binary classification task instead of categorical and it differs
    from COVIDx8B.
    
    Parameters
    ----------
    target_data_folder : str
        Path to target data folder, where the files will be copied to and sorted by their labels.
    src_data_folder : str
        Path to source folder containing the original data which is to be copied and labeled.
    labels_file_path : str
        Path to text file, which contains labels for each file. The file should have a line entry for each file and syntax is such that parameters of the file are separated
        by spaces and the second from last parameter is the class label and third from last parameter is file name.
    """
    
    labels_file = open(labels_file_path, 'r')
    count = 0

    while True:
        count += 1
        line = labels_file.readline()
        
        # if line is empty -> end of file reached
        if not line:
            break
        
        split = line.split(' ')
        if split[-2] == 'positive' or split[-2] == 'negative':
            if split[-3].startswith('COVID('):
                name = split[-3].replace('(','-')
                name = name.replace(')','')
                shutil.copy(os.path.join(src_data_folder, name), os.path.join(target_data_folder, split[-2], name))
        
            else:
                shutil.copy(os.path.join(src_data_folder, split[-3]), os.path.join(target_data_folder, split[-2], split[-3]))
        else:
            raise ValueError(f'Label must be either positive or negative, it was: {split[-2]}')


def label_data_x3(target_data_folder, src_data_folder, labels_file_path):
    """
    Takes data from one folder and separates it into class subdirectories in the target data folder. 
    This separation is based on class labels specified in a given file.
    
    Note: This function is specific for the COVIDx3, as it is a categorical classification task.
    
    Parameters
    ----------
    target_data_folder : str
        Path to target data folder, where the files will be copied to and sorted by their labels.
    src_data_folder : str
        Path to source folder containing the original data which is to be copied and labeled.
    labels_file_path : str
        Path to text file, which contains labels for each file. The file should have a line entry for each file and syntax is such that parameters of the file are separated
        by spaces and the second from last parameter is the class label and third from last parameter is file name.
    """
    
    labels_file = open(labels_file_path, 'r')
    
    while True:
        line = labels_file.readline()
        
        # if line is empty -> end of file reached
        if not line:
            break
        
        split = line.split(' ')
        
        if not os.path.exists(os.path.join(src_data_folder, split[1])):
            continue
        
        if split[2].strip() == 'normal':
            shutil.copy(os.path.join(src_data_folder, split[1]), os.path.join(target_data_folder, 'negative', split[1]))
        elif split[2].strip() == 'COVID-19':
            shutil.copy(os.path.join(src_data_folder, split[1]), os.path.join(target_data_folder, 'positive', split[1]))
        elif split[2].strip() == 'pneumonia':
            pass
        else:
            raise ValueError(f"Label must be either normal, COVID-19 or pneumonia, it was: '{split[2]}'")

            
            
def structure_dataset(target_data_folder, src_data_folder, train_labels_path, test_labels_path, dataset='x8'):
    """
    Prepares the file structure for the final dataset and splits train and test data according to class labels.
    
    Parameters
    ----------
    target_data_folder : str
        Path to target data folder, where the files will be copied to and sorted by their labels.
    src_data_folder : str
        Path to source folder containing the original data which is to be copied and labeled.
    train_labels_path : str
        Path to text file, which contains labels for each file in the train set. The file should have a line entry for each file and syntax is such
        that parameters of the file are separated by spaces and the second from last parameter is the class label and third from last parameter is file name.
    test_labels_path : str
        Path to text file, which contains labels for each file in the test set. The file should have a line entry for each file and syntax is such
        that parameters of the file are separated by spaces and the second from last parameter is the class label and third from last parameter is file name.
        
    """
    
    create_fs(target_data_folder)
    if dataset == 'x3':
        label_data_x3(os.path.join(target_data_folder, 'train'), os.path.join(src_data_folder, 'train'), train_labels_path)
        label_data_x3(os.path.join(target_data_folder, 'test'), os.path.join(src_data_folder, 'test'), test_labels_path)
    elif dataset == 'x7':
        label_data_x7(os.path.join(target_data_folder, 'train'), os.path.join(src_data_folder, 'train'), train_labels_path)
        label_data_x7(os.path.join(target_data_folder, 'test'), os.path.join(src_data_folder, 'test'), test_labels_path)
    elif dataset == 'x8':
        label_data_x8(os.path.join(target_data_folder, 'train'), os.path.join(src_data_folder, 'train'), train_labels_path)
        label_data_x8(os.path.join(target_data_folder, 'test'), os.path.join(src_data_folder, 'test'), test_labels_path)

def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'x3':
        structure_dataset(target_data_folder_x3, src_data_folder_x3, train_labels_path_x3, test_labels_path_x3, 'x3')
    elif len(sys.argv) > 1 and sys.argv[1] == 'x7':
        structure_dataset(target_data_folder_x7, src_data_folder_x7, train_labels_path_x7, test_labels_path_x7, 'x7')
    else:
        structure_dataset(target_data_folder_x8, src_data_folder_x8, train_labels_path_x8, test_labels_path_x8, 'x8')
    
if __name__ == '__main__':
    main()