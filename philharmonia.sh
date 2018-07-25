#!/bin/bash

mkdir models/
mkdir philharmonia/
cd philharmonia/
mkdir data

#get philharmonia file
wget www.philharmonia.co.uk/assets/audio/samples/all-samples.zip

#unzip samples
unzip all-samples.zip

#no one uses mac anymore
rm -r __MACOSX

#move into directory
cd all-samples

#remove percussion because of weird formatting and it is not that useful anyways
rm percussion.zip

#replace a space with a dash for instrument zip files
find . -type f -name "* *" | while read file; do mv "$file" ${file// /-}; done

#unzip all instrument files into new directories
for zips in *.zip; do
	echo $zips
        newdir=$(basename $zips .zip)
        mkdir $newdir
        mv $zips $newdir
        cd $newdir
        unzip $zips
	rm $zips
        cd ../
done

ls

#convert mp3 files to wav files
for filename in */*.mp3; do
        newfilename=$(echo $filename | cut -d '.' -f 1)$'.wav'
        ffmpeg -i $filename $newfilename
        rm $filename
done

cd ../../

python fund_note.py write

for i in $(seq 1 20);
do
        python fund_note.py read finalize $i
done

