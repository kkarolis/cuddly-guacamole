#!/bin/bash
set -exuo pipefail

python server/main.py init-db
python server/main.py run-server