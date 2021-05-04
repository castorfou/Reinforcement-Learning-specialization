#!/bin/bash
conda env export --from-history > env_`echo $CONDA_DEFAULT_ENV`.yml
