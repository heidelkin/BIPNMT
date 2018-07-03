#Script to download the data
#or any text data that are lowercased and tokenized

mkdir -p ../data
cd ../data
wget http://www.cl.uni-heidelberg.de/statnlpgroup/bipdata/en-fr_prep.tar.gz
tar -xzf en-fr_prep.tar.gz
cd ..
