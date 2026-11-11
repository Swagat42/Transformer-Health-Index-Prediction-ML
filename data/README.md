# Dataset

The dataset is **not** stored in this repository. It is *"Data for: Root cause analysis improved with machine learning for failure analysis in power transformers"* by Ricardo Arias and Jennifer Mejia Lara, published on Mendeley Data:

**https://data.mendeley.com/datasets/rz75w3fkxy**

Save the CSV in this folder as:

```
data/Health index1.csv
```

(Or pass any path with `python -m src.train --data <path>`.)

## Schema

470 transformers × 16 columns.

| Group | Columns |
|---|---|
| Dissolved gases (ppm) | `Hydrogen`, `Oxigen`, `Nitrogen`, `Methane`, `CO`, `CO2`, `Ethylene`, `Ethane`, `Acethylene` |
| Oil quality | `DBDS`, `Power factor`, `Interfacial V`, `Dielectric rigidity`, `Water content` |
| Target | `Health index` |
| Dropped | `Life expectation` (excluded to avoid target leakage) |

Column spellings (`Oxigen`, `Acethylene`) follow the original file. Please check the dataset page for its license and cite the original authors if you reuse it.

`sample_input.csv` shows the format expected by `python -m src.predict`. It holds the first five rows of the dataset's feature columns.
