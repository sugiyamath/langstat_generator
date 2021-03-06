# langstat_generator

## setup

最初に、aws parallelclusterのクラスタを作成する必要があるので以下を参照して作成してください:

https://github.com/aws/aws-parallelcluster

それができたら、クラスタにログインします。

```sh
$ pcluster ssh yourclustername -i your.pem
```

マスターインスタンス内でawsの設定を行います。


```sh
$ aws configure
```

goofysのバイナリを入手します。

```sh
$ wget https://github.com/kahing/goofys/releases/latest/download/goofys
$ chmod +x goofys
```

python3-pipとvirtualenvを入れます。


```sh
sudo apt install python3-pip
pip3 install virtualenv --user
.local/bin/virtualenv venv
```

このプロジェクト自体をクローンして実行ファイルを移動します。


```sh
$ git clone https://github.com/sugiyamath/langstat_generator
$ ./venv/bin/pip install -r langstat_generator/requirements.txt
$ cp langstat_generator/*.py .
$ cp langstat_generator/slurm/*.sh .
$ cp langstat_generator/scripts/ -r .
$ chmod +x run.sh
```

wet.path.gzをCommonCrawlから入手します: https://commoncrawl.org/the-data/get-started/

```sh
$ wget https://commoncrawl.s3.amazonaws.com/crawl-data/CC-MAIN-2019-39/wet.paths.gz
```

もし言語モデル等のバイナリを格納したs3バケットを持っている場合は次のステップは飛ばしてください。



持っていない場合は、kenlmの言語モデル、facebookのfasttextの言語検出モデル、言語モデル用のsentencepieceモデルを以下のディレクトリツリーにアップロードします:


```
s3://yourbucketname/bin/
- lid.bin
- lm_sp/
-- en.arpa.bin
-- en.sp.model
-- ...and so on
```


## run

実行は単純に以下を実行するだけです:

```sh
$ sbatch --no-kill run.sh
```

実行前に、run.shとexecute.shを確認してください:

[run.sh]

- PY_PATH: pythonのバイナリパス。
- OUT_DIR: 最終出力の格納場所。
- WET_PATH: wet.paths.gzのパス。
- BIN_DIR: 言語モデルを保存したs3バケットのbinプレフィックス。
- MAX_SHARD_NUM: 各々のタスクで実行するタスク数。
- MAX_NODE_NUM: ノード番号の最大値。(6は7つのノードを意味します。)


[execute.sh]

- ```goofys langstat ./langstat``` は言語モデルを格納したs3バケットをマウントすることを意味するので、バケット名を変更する場合は変更してください。
- ```head -n 1120``` は使うWETファイル数なので、全てを読み込む場合はこの部分のフィルターをコマンドから除外してください。


もし、各々のノードで利用するCPU数を変更したい場合は、main.pyのパラメータに ```num_cpus``` があるので、それを修正してください。

また、shardingモジュールは、全体のノード数、シャード1あたりのWET数、ノード番号、シャード番号を指定することでファイルを分割しています。以下は、main.py内でshardingモジュールが使われている部分です。

```python
targets = list(
    sharding.sharding(all_targets,
                      node_id,
                      shard_id,
                      total_nodes,
                      wet_per_shard))
```

この中で、```total_nodes``` と```wet_per_shard``` はmain.pyのパラメータに存在します。cc_netの推奨設定は ```wet_per_shard=50``` なので、通常、変更する必要があるのは ```total_nodes``` だけです。

shardingモジュールをテストするには以下のコマンドを実行します:

```sh
$ zcat wet.paths.gz | head -n テストしたいWET総数 | python3 sharding.py テストしたいノード番号 テストしたいノード内シャード番号(0～n)
```

この出力は、当該ノード・シャードで使われるWETパスの一覧が表示されます。

上記のshardingモジュールのテストをする場合、total_nodesの設定は、sharding.pyのデフォルトパラメータを書き換えてください。

## merge

langstat_generatorの出力は二種類あります:

- langstat_<nodeid>_<shardid>.txt : 言語統計
- lmscore_<nodeid>_<shardid>.txt : 言語モデルと言語検出モデルのスコア

この2つの結果をマージしたい場合、```scripts/merge_stat.py``` を使います。上記ファイルが出力された先が ```./out``` の場合は以下です:

```sh
$ python3 scripts/merge_stat.py ./out
```

これにより、```./out``` 内に、```.txt.mgd```というファイルが生成されます。このファイルが結合後のファイルです。