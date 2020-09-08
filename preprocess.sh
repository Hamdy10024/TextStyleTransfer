export EMBED= $1
export CORPUS=$2
mkdir data
cp -r CORPUS ./data/corpus
cp EMBED ./data
cd src
python3 corpus_helper.py --data DATA --vec_dim 300 --embed_fn ../data/glove.6B.300d.txt --data_type yelp

