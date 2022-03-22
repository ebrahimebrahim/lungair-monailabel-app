# LungAIR MONAILabel App

This project is under development and probably isn't working yet.

This is a MONAILabel app that can hopefully be used to support the lung segmentation component of LungAIR score computation.

For more information, see [MONAILabel](https://github.com/Project-MONAI/MONAILabel/wiki).

To start the server locally:

```sh
monailabel start_server --app . --studies <path_to_xrays>
```

## Structure of the App

- **[./lib/infer.py](./lib/infer.py)** is the script where researchers define the inference class (i.e. type of inferer, pre transforms for inference, etc).
- **[./lib/train.py](./lib/train.py)** is the script to define the pre and post transforms to train the network/model
- **[./lib/strategy.py](./lib/strategy.py)** is the file to define the image selection techniques.
- **[./lib/transforms.py](./lib/transforms.py)** is the file to define customised transformations to be used in the App
- **[main.py](./main.py)** is the script to define network architecture.



