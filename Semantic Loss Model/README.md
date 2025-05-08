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