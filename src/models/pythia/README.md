# Setting Up
There are a couple of requirements that can make your life easier:
* Install CUDA in relation to your pc's specs and check if your pc can run cuda (makes training WAY faster)
    * Example: without cuda (it took me 16s/datapoint and I had to run over 38k datapoints... so I just gave up and installed cuda) and with cuda (it took me 2s/datapoint) (I have a 3070)
* If not installed already, install venv (with the required dependencies)

# Running Inference Script (./pythia_zerofew.py)
**Run from root**:
```bash
{./venv/Scripts/python.exe or python} src/models/pythia/pythia_zerofew.py --model_size {160m/410m/1.4b} --regime {zeroshot/fewshot} --data_path {data/processed/*.jsonl} --output_dir {results/pythia}

```
**Note**: in data/processed there are 3 files: train.jsonl, dev.jsonl, test.jsonl. You can use any of them as input data.


## Example of running the script for all models and regimes
**Run from root**:
```bash
for model_size in 160m 410m 1.4b; do
    for regime in zeroshot fewshot; do
        {./venv/Scripts/python.exe or just python} src/models/pythia/pythia_zerofew.py --model_size $model_size --regime $regime --data_path data/processed/dev.jsonl --output_dir results/pythia
    done
done
```