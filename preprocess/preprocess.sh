#!/bin/bash

script_dir=`dirname $0`
script_dir=`cd $script_dir; pwd`
root_dir=`cd $script_dir/..; pwd`
corpus_dir=`cd $root_dir/corpus; pwd`

find . ! -path . -type d | xargs rm -rf
bz2_files=`find $corpus_dir -type f`
for bz2_file in $bz2_files
do
    tar -xvjf $bz2_file
done

cd $script_dir
python $script_dir/email_etl.py $corpus_dir
echo "===============all done==============="