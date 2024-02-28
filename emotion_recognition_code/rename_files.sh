#!/bin/bash

root_dir="/home/marta/Desktop/hri/project"
dest_dir1="$root_dir/ravdess1/"
dest_dir2="$root_dir/ravdess2/"
i=0

find "$root_dir/ravdess" -type d | while read -r directory; do
    #actor=$("$directory" | cut -c 7- 8)
    echo "Processing dir: $directory"
    pattern="/03-01-07-01-02-01" #change: 03-01-emo(01/03/04/05/06/07)-01-stat(01/02)-01-(actor)
    find "$directory" -type f | while read -r file; do
        echo "Processing file: $file"
        echo "Actor num: $i"
        matching_files=$(grep "$pattern" )
    if [ "$i" -lt 10 ]; then 
        cp "$file" "$dest_dir1/111"$i"_RAU_DIS_LO.wav"
        i=$((i+1))
    fi
    if [ "$i" -ge 10 ]; then
        cp "$file" "$dest_dir2/11"$i"_RAU_DIS_LO.wav"
        i=$((i+1))
    fi
    done
    i=$((i+1))
done

