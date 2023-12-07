import yaml
import time
import numpy as np

from classification.nn_utils import NearestNeighbour
from classification.config_utils import load_config
from classification.dataset_utils import Dataset


def main():
    try:
        # Get the configs
        config = load_config()

        # Load the dataset
        dataset = Dataset(config)

        avg_error = 0.0
        accumulative_time = 0.0

        for _ in range(dataset.get_cross_validation_folds()):
            dataset_dict = dataset.get_next_dataset()
            
            # Obtain the train/test data/labels
            train_data = dataset_dict["train_x"]
            train_labels = dataset_dict["train_y"]
            test_data = dataset_dict["test_x"]
            test_labels = dataset_dict["test_y"]

            # Create the model
            start_time = time.time()
            nn = NearestNeighbour(train_data, train_labels, config["preprocess_method"])
            test_predictions = nn.mass_predict(test_data)

            # Predict the whole test set and record times
            end_time = time.time()

            # Report time
            print("Classification time (seconds): ", end_time - start_time)
            accumulative_time += end_time - start_time

            # Calculate and report error
            err_positions = np.not_equal(test_predictions, test_labels)
            error = float(np.sum(err_positions))/len(test_labels)
            print("Error of nearest neighbor classifier: ", error)

            avg_error += error

        print("Average error:", avg_error / dataset.get_cross_validation_folds())
        print("Accumulative time:", accumulative_time)

    except KeyboardInterrupt:
        print("Exiting early...")


if __name__ == "__main__":
    main()