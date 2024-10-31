# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.


import logging
import os
import json
import jsonpickle
import pickle
import numpy as np

import numpy as np
from sklearn import datasets,svm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

from collections import namedtuple

from sklearn.ensemble import RandomForestRegressor

import utils.models
from utils.abstract import AbstractDetector
from utils.models import load_model, load_models_dirpath

import torch
import torchvision
import skimage.io

from scripts.public.evaluate_colorful_memory_model import evaluate

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Detector(AbstractDetector):
    def __init__(self, metaparameter_filepath, learned_parameters_dirpath):
        """Detector initialization function.

        Args:
            metaparameter_filepath: str - File path to the metaparameters file.
            learned_parameters_dirpath: str - Path to the learned parameters directory.
        """
        metaparameters = json.load(open(metaparameter_filepath, "r"))

        self.metaparameter_filepath = metaparameter_filepath
        self.learned_parameters_dirpath = learned_parameters_dirpath
        self.model_filepath = os.path.join(self.learned_parameters_dirpath, "model.bin")
        self.models_padding_dict_filepath = os.path.join(self.learned_parameters_dirpath, "models_padding_dict.bin")
        self.model_layer_map_filepath = os.path.join(self.learned_parameters_dirpath, "model_layer_map.bin")
        self.layer_transform_filepath = os.path.join(self.learned_parameters_dirpath, "layer_transform.bin")

        self.input_features = metaparameters["train_input_features"]
        self.svd_features = 10
        self.weight_params = {
            "rso_seed": metaparameters["train_weight_rso_seed"],
        }

    def write_metaparameters(self):
        metaparameters = {
            "train_input_features": self.input_features,
            "train_svd_features": self.svd_features,
            "train_weight_rso_seed": self.weight_params["rso_seed"],
        }

        with open(os.path.join(self.learned_parameters_dirpath, os.path.basename(self.metaparameter_filepath)), "w") as fp:
            fp.write(jsonpickle.encode(metaparameters, warn=True, indent=2))

    def automatic_configure(self, models_dirpath: str):
        """Configuration of the detector iterating on some of the parameters from the
        metaparameter file, performing a grid search type approach to optimize these
        parameters.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        for random_seed in np.random.randint(1000, 9999, 10):
            self.weight_params["rso_seed"] = random_seed
            self.manual_configure(models_dirpath)

    def generate_features(self, model_dict):
        """generate the features for the model.

        Args:
            models_dict: model dictionary
        """
        model_state = model_dict['model_state']  # Extract the model_state component
        model_feats = None
        # Iterate through the keys in model_state
        for key in model_state.keys():
            if key.startswith('state_emb'):
                weight = model_state[key] 
                if key.endswith('.weight'):
                    # Calculate singular values for the weight matrix
                    if len(weight.shape) == 4:  # Ensure it's a 4D weight matrix
                    # Reshape the weight matrix to 2D (out_channels, in_channels * kernel_size)
                        reshaped_weight = weight.view(weight.shape[0], -1)  # Flatten spatial dimensions
                        u, s, v = torch.svd(reshaped_weight)  # Compute SVD
                    # Flatten the weights and concatenate
                    weight_flat = weight.view(-1)  # Flatten to 1D
                    # Calculate min, max, mean, std of `weight_flat`
                    min_val = weight_flat.min().unsqueeze(0)
                    max_val = weight_flat.max().unsqueeze(0)
                    mean_val = weight_flat.mean().unsqueeze(0)
                    std_val = weight_flat.std().unsqueeze(0)
                    svd_max = s.max().unsqueeze(0)
                    svd_min = s.min().unsqueeze(0)
                    svd_mean = s.mean().unsqueeze(0)

                    if model_feats is None:
                        model_feats = torch.cat((min_val, max_val, mean_val, std_val, s[:self.svd_features]))
                    else:
                        model_feats = torch.cat((model_feats, min_val, max_val, mean_val, std_val, s[:self.svd_features]))

                elif key.endswith('.bias'):
                    # Get the bias tensor, flatten it
                    bias = model_state[key].view(-1)  # Flatten to 1D
                    min_val = bias.min().unsqueeze(0)
                    max_val = bias.max().unsqueeze(0)
                    mean_val = bias.mean().unsqueeze(0)
                    std_val = bias.std().unsqueeze(0)
                    if model_feats is None:
                        model_feats = torch.cat((min_val, max_val, mean_val, std_val))
                    else:
                        model_feats = torch.cat((model_feats, min_val, max_val, mean_val, std_val))
        return model_feats.unsqueeze(0)
    
    def manual_configure(self, models_dirpath: str):
        """Configuration of the detector using the parameters from the metaparameters
        JSON file.

        Args:
            models_dirpath: str - Path to the list of model to use for training
        """
        # Create the learned parameter folder if needed
        if not os.path.exists(self.learned_parameters_dirpath):
            os.makedirs(self.learned_parameters_dirpath)

        # List all available model
        model_path_list_unsorted = [os.path.join(models_dirpath, model) for model in os.listdir(models_dirpath)]
        model_path_list = sorted(model_path_list_unsorted)
        logging.info(f"Loading %d models...", len(model_path_list))

        self.weight_params["rso_seed"] = 44

        model_repr_dict, model_ground_truth_dict = load_models_dirpath(model_path_list)
        X = []
        y = []
        for idx, model_dict in enumerate(model_repr_dict['dict']):
            model_feats= self.generate_features(model_dict)
            y.append(model_ground_truth_dict['dict'][idx])
            X.append(model_feats.numpy())
        X = np.vstack(X)

        #Split the dataset into training and testing sets
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=self.weight_params["rso_seed"])
        X_train = X
        y_train= y
        self.input_features=X_train.shape[1]

        #Create the SVM classifier
        svm_classifier = SVC(kernel='rbf')  #change the kernel as needed (e.g., 'rbf', 'poly', etc.)
        #Train the classifier
        logging.info("Training SVM Classifier.")
        svm_classifier.fit(X_train, y_train)
        #Make predictions on the test set
        #y_pred = svm_classifier.predict(X_test)
        #Evaluate the model
        accuracy = accuracy_score(y_train, y)
        # Output results
        print("Accuracy:", accuracy)

        # Initialize the model
        #mlp = MLPClassifier(hidden_layer_sizes=(10, 10),  
        #            activation='relu',           
        #            solver='adam',               
        #            max_iter=1000,               
        #            random_state=42)             
        
        # Train the MLP classifier
        #mlp.fit(X_train, y_train)
        # Make predictions on the test set
        #y_pred = mlp.predict(X_test)
        # Calculate accuracy
        #accuracy = accuracy_score(y_test, y_pred)
        #print(f"Accuracy: {accuracy:.2f}")

        logging.info("Saving SVM model...")
        with open(self.model_filepath, "wb") as fp:
            pickle.dump(svm_classifier, fp)

        self.write_metaparameters()
        logging.info("Configuration done!")

    def inference_on_example_data(self, model_filepath, examples_dirpath):
        """Method to demonstrate how to inference on a round's example data.

        Args:
            model_filepath: path to the pytorch model file
            examples_dirpath: the directory path for the round example data
        """
        args = namedtuple('args', ['model_dir',
                                   'eipsodes',
                                   'success_rate_episodes',
                                   'procs',
                                   'worst_episodes_to_show',
                                   'argmax',
                                   'gpu',
                                   'grid_size',
                                   'random_length',
                                   'max_steps'])

        args.model_dir = os.path.dirname(model_filepath)
        args.episodes = 5
        args.success_rate_episodes = 5
        args.procs = 10
        args.worst_episodes_to_show = 10
        args.argmax = False
        args.gpu = False
        args.seed = 1

        with open(os.path.join(args.model_dir, "reduced-config.json"), "r") as f:
            config = json.load(f)

        #   grid_size: (int) Size of the environment grid
        args.grid_size = config["grid_size"]
        #   random_length: (bool) If the length of the hallway is randomized (within the allowed size of the grid)
        args.random_length = config["random_length"]
        #   max_steps: (int) The maximum allowed steps for the env (AFFECTS REWARD MAGNITUDE!) - recommend 250
        args.max_steps = config["max_steps"]

        evaluate(args)

    def infer(
            self,
            model_filepath,
            result_filepath,
            scratch_dirpath,
            examples_dirpath,
            round_training_dataset_dirpath,
    ):
        """Method to predict whether a model is poisoned (1) or clean (0).

        Args:
            model_filepath:
            result_filepath:
            scratch_dirpath:
            examples_dirpath:
            round_training_dataset_dirpath:
        """

        # Inferences on examples to demonstrate how it is done for a round
        self.inference_on_example_data(model_filepath, examples_dirpath)

        # build a feature vector for the model, in order to compute its probability of poisoning
        _,model_repr_dict,_ = load_model(model_filepath)
        X = self.generate_features(model_repr_dict)

        # load the SVM from the learned-params location
        with open(self.model_filepath, "rb") as fp:
            regressor: SVC = pickle.load(fp)

        # use the RandomForest to predict the trojan probability based on the feature vector X
        probability = regressor.predict(X)[0]
        # clip the probability to reasonable values
        probability = np.clip(probability, a_min=0.01, a_max=0.99)

        # write the trojan probability to the output file
        with open(result_filepath, "w") as fp:
            fp.write(str(probability))

        logging.info("Trojan probability: {}".format(probability))
