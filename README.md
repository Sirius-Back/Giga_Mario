# GigaMario

Minimal Python package scaffold.

## Install (conda-preferred)

Activate the target conda env (currently **base**, not `caduceus_env`), then:

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate base
conda env update -f environment.yml --prune
```

Equivalent editable install inside the active conda env:

```bash
python -m pip install -e .
```

Verify:

```bash
python -c "import GigaMario; print(GigaMario.__version__)"
```
