# STAT 671 Cats/Dogs Classification Demo

This demo is made for the Portland State University (PSU) STAT 671 class. Inspiration was taken from 
[VictorRielly's voronoi repo](https://github.com/VictorRielly). It shows how
a simple kernel based classification algorithm, equipped with an appropriate
kernel, can classify images of cats and dogs.

## Usage

Before running, place images of your cats and dogs in the `user_data` folder.
The notebook will automatically grab these images, classify them, and display
them.

## Installation

Install the python requirements

```
pip install -r requirements.txt
```

Launch the Jupyter notebook
```
jupyter notebook
```

Open the `Cats-And-Dogs-Demo.ipynb` notebook and run it.

## Create Model (optional)

The demo comes with a NN model saved as a `.h5` file, so there is no need to train the NN again. If you need to re-train 
it, follow these steps

1. Go [here](https://www.kaggle.com/c/dogs-vs-cats/data) to download the `dogs-vs-cats.zip` and extract all folders at 
the correct hierarchies. Place it in the current working directory. 
2. Now, run
```
python src/create_model.py
```
It took 5 hours of training on CPU, and ~1.5 hours on GPU. The new `.h5` file will be 
populated in the current working directory.
