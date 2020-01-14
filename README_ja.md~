# langstat_generator

## setup

First, you need to create a cluster on aws parallelcluster.
See this: https://github.com/aws/aws-parallelcluster

After that, login to the cluster:

```sh
$ pcluster ssh yourclustername -i your.pem
```

You need to configure aws in the master instance:

```sh
$ aws configure
```

And then, download goofys binary:

```sh
$ wget https://github.com/kahing/goofys/releases/latest/download/goofys
$ chmod +x goofys
```

You also need to install python3-pip and virtualenv.

```sh
sudo apt install python3-pip
pip3 install virtualenv --user
.local/bin/virtualenv venv
```

And then, clone this project and do some stuff:

```sh
$ git clone https://github.com/sugiyamath/langstat_generator
$ ./venv/bin/pip install -r langstat_generator/requirements.txt
$ cp langstat_generator/*.py .
$ cp langstat_generator/slurm/*.sh .
$ cp langstat_generator/scripts/ -r .
$ chmod +x run.sh
```

Next, you should download wet.path.gz from CommonCrawl: https://commoncrawl.org/the-data/get-started/

```sh
$ wget https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2019-39/wet.paths.gz
```

If you already have s3 bucket which have lm models, skip the next step.

Upload your kenlm language models, language detection model, and sentencepiece model to the s3 bucket as following directory tree:

```
s3://yourbucketname/bin/
- lid.bin
- lm_sp/
-- en.arpa.bin
-- en.sp.model
-- ...and so on
```

lid.bin is facebook's language detection model.

# run

Running the script is simple. Just do this:

```sh
$ sbatch --no-kill run.sh
```

However, you need to set up run.sh itself.

- PY_PATH: python binary path.
- OUT_DIR: final output will saved to this dir.
- WET_PATH: the path of wet.paths.gz
- BIN_DIR: the s3 bucket's bin prefix of language models.
- MAX_SHARD_NUM: the number of tasks each nodes.
- MAX_NODE_NUM: the max node number.(e.g. 6 means 7 nodes)

You also need to check execute.sh.

- ```goofys langstat ./langstat``` means mounting the s3 bucket of language model. Change this from ```langstat``` to your bucket name.
- ```head -n 1120``` is the number of wet files you want to use.



