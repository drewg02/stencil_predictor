# Stencil Predictor

This project consists of tools to generate initial conditions for a stencil, run the stencil over a number of iterations, and train a machine learning model to predict the results of the stencil.

## Generators

The following scripts are used to generate initial conditions for the simulation. 
You can either provide a number of rows and columns to create a new array, or provide an input file that will be modified, one of these two things is required. 
The thickness is the thickness of the pattern and defaults to 1.
The output file will be a numpy array saved as a .npy file in all cases.
The output png file is optional and will be a visualization of the array saved as a .png file.

**[center_generator.py](generators/center_generator.py)**

- Generates initial conditions with a pattern starting from the center of the array.

```
center_generator.py [-h] [--rows ROWS] [--cols COLS] [--input_file INPUT_FILE] [--thickness THICKNESS] [--output_png_file OUTPUT_PNG_FILE] <output_file>
```

**[outline_generator.py](generators/outline_generator.py)**

- Generates initial conditions with an outline around the edges of the array.

```
outline_generator.py [-h] [--rows ROWS] [--cols COLS] [--input_file INPUT_FILE] [--thickness THICKNESS] [--output_png_file OUTPUT_PNG_FILE] <output_file>
```

**[plus_generator.py](generators/plus_generator.py)**

- Generates initial conditions with a plus pattern in the center of the array.

```
plus_generator.py [-h] [--rows ROWS] [--cols COLS] [--input_file INPUT_FILE] [--thickness THICKNESS] [--output_png_file OUTPUT_PNG_FILE] <output_file>
```

**[random_generator.py](generators/random_generator.py)**

- Generates initial conditions with random values turned on.

```
random_generator.py [-h] <rows> <cols> <output_file>
```

**[test_generator.py](generators/test_generator.py)**

- Generates initial conditions where the leftmost and rightmost columns are turned on and the rest are turned off.

```
test_generator.py [-h] <rows> <cols> <output_file>
```

# Utilities

**[utilities.py](utilities.py)**

- Stores general-purpose utility functions used throughout the project.

**[training_utilities.py](training_utilities.py)**

- Contains various utilities specifically for training.

# Stenciling

**[stencil2d.py](stencil2d.py)**

- Applies a stenciling algorithm to an array. 
  It will run for num_iterations and save the final result to output_file.
  If all_iterations is set to a file path, it will save a .npy file containing all iterations of the stencil.

   ```
   usage: stencil2d.py [-h] <num_iterations> <input_file> <output_file> [all_iterations]
   ```

# Training

**[train.py](train.py)**

- Used to train a machine learning model. 
- Uses static variables for configuration. 
- Involves preparing data, defining the model, training it, and saving relevant data such as dataloaders and the trained model.

# Inference

**[predict.py](predict.py)**

- Used to predict images with a pre-trained model based on the test_loader saved with the model. 
- Uses static variables for configuration. 
- Performs predictions and saves the results as images.

**[predict_all.py](predict_all.py)**

- Similar to `predict.py`, but instead of using the test_loader, it takes the entire dataset, including validation and training data, and saves the predictions as a .mp4.
- Uses static variables for configuration.

**[difference2d.py](difference2d.py)**

- Takes two array files as input and saves the difference between them as an image to disk.

```
usage: difference2d.py [-h] [--graph-format | --no-graph-format] <input_file_1> <input_file_2> <output_file>
```

# Array Visualization

**[display2d.py](display2d.py)**

- Takes in one .npy array and saves it as an image to disk.

```
usage: display2d.py [-h] [--graph-format | --no-graph-format] <input_file> <output_file>
```

**[print2d.py](print2d.py)**

- Prints a .npy array file to the console.

```
usage: print2d.py [-h] <input_file>
```

**[movie2d.py](movie2d.py)**

- Takes a .npy file containing an array of images and converts them into an .mp4 video in the specified order.

```
usage: movie2d.py [-h] <input_file> <output_file>
```

# Model Visualization

**[print_model.py](print_model.py)**

- Loads a trained model and prints it to the console. 
- Uses static variables for configuration.

**[visualize_model.py](visualize_model.py)**

- Utilizes torchviz/graphviz to generate an image representation of a model and saves it to a file. 
- Uses static variables for configuration.

# Example Usage

We should add something here.
