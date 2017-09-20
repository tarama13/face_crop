#!/bin/bash

cmd='python crop_face.py'
echo face crop processing...
for f in $1/*.jpg $1/*.jpeg $1/*.JPG $1/*.JPEG $1/*.png $1/*.PNG
do
	if [ -f $f ]; then
		echo $cmd $f
		$cmd $f
	fi
done

echo done
