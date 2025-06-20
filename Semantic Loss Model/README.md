## Semantic Loss

Here you can find the implementation of the model chosen: **Semantic Loss**.

The implementation is based on the library [Semantic Loss PyTorch](https://github.com/lucadiliello/semantic-loss-pytorch), hence this README will repeat part of the instructions given in their repository in order to run this model.


#### Running a simple model
The `model.py` file holds the logic of the model. It is also the file called in case the model is ran from the command line.

As mentioned in the main supporting [library](https://github.com/lucadiliello/semantic-loss-pytorch) these are the steps if you would like to write different constraints for the model. As it stands this folder contains the One-Hot constraint for MNIST. Such that the model is penalized for outputting indecisive results where more than one neuron in the output layer is "hot", and favoring outputs with only one peak (i.e.: `[0.0, 0.0, 0.0, 0.0, 0.9, 0.05, 0.05, 0.0, 0.0, 0.0]`)

You can skip the following commands if you don't wish to change this constraint:

- Install this package
```bash
pip install git+https://github.com/lucadiliello/semantic-loss-pytorch.git
```

- Write your constraints respecting the `sympy` sintax, with variables like `X1.2` and operators like `And(X0.2.3, X1.1.1)`. All lines are put in `and` relationship. Convert to `DIMACS` syntax with:
```bash
python -m semantic_loss_pytorch.constraints_to_cnf -i <input_file>.txt -o <dimacs_file>.txt 
```

- Install `PySDD`:
```bash
pip install PySDD
```

- Compile your constraint to a `vtree` and an `sdd` file. To do so, run:
```bash
pysdd -c dimacs.txt -W constraint.vtree -R constraint.sdd
```

`PySDD` is only needed for this step. If you don't need to convert other 
dimacs constraints to `vtree` and `sdd` files, you can uninstall it.


##### Necessary Command:
To run the model simply run:
```bash
python model.py
```
Arguments exist to adjust batch size, epochs, and the name of the vtree and sdd files.

You can also run the model without Semantic Loss (barebones MLP) by adding the `no_semantic_loss` flag


#### Other commands
This script supports the following command-line arguments to control training behavior and experiment configuration:

`--batch_size` (int, default: 64): Batch size used during training.

`--epochs` (int, default: 30): Number of training epochs.

`--constraint_sdd` (str, default: "constraint.sdd"): Path to the .sdd file specifying logical constraints.

`--constraint_vtree` (str, default: "constraint.vtree"): Path to the corresponding .vtree file for the SDD.

`--data_dir` (str, default: "../data"): Root directory of the dataset.

`--no_semantic_loss` (flag): If set, disables semantic loss. Enabled by default.

`--max_train_samples` (int, default: 5000): Maximum number of training samples to use.

`--experiment_name` (str, default: "default_experiment"): Name to tag the experiment for logging and output purposes.

`--max_test_samples` (int, default: 5000): Maximum number of test samples to use.

`--max_classes` (int, default: 20): Maximum number of output classes.

`--wanet_magnitude` (float, default: 0.5): Magnitude of the WaNet backdoor perturbation.

Use these flags to customize experiments and facilitate reproducibility and ablation studies.