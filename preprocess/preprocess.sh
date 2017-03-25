#!/bin/bash

script_dir=`dirname $0`
script_dir=`cd $script_dir; pwd`
root_dir=`cd $script_dir/..; pwd`
corpus_dir=`cd $root_dir/corpus; pwd`

cd $corpus_dir
find . ! -path . -type d | xargs rm -rf
bz2_files=`find . -type f`
for bz2_file in $bz2_files
do
    tar -xvjf $bz2_file
done

cd $root_dir
rm -rf $root_dir/data/preprocess_spam
rm -rf $root_dir/data/preprocess_nonspam
rm -rf $root_dir/data/spam_train
rm -rf $root_dir/data/spam_test
rm -rf $root_dir/data/nonspam_train
rm -rf $root_dir/data/nonspam_test
mkdir -p $root_dir/data/preprocess_spam
mkdir -p $root_dir/data/preprocess_nonspam
mkdir -p $root_dir/data/spam_train
mkdir -p $root_dir/data/spam_test
mkdir -p $root_dir/data/nonspam_train
mkdir -p $root_dir/data/nonspam_test
echo "python $script_dir/email_etl.py"
python $script_dir/email_etl.py
echo "===============all done==============="