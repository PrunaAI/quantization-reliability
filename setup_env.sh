#!/usr/bin/env bash
# Creates the quant-rel conda env, then installs the packages that can't go
# in environment.yml's pip: list
set -euo pipefail

conda env create -f environment.yml
conda run -n quant-rel pip install --no-deps \
  optimum==1.22.0 \
  gptqmodel==1.9.0 \
  optimum-quanto==0.2.4
