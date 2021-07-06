#!/bin/bash

# $1 URL
# $2 resultfile
# $3 startframe
# $4 videolength
# $5 gpuid
# $6 option

#mkfifo /ai/tmpdir/fifo_app.$$
#mkfifo /ai/tmpdir/fifo_model.$$

#python text_detection_recognition.py "$1" $2 $3 $4 $5 $$
python3 ocr_video.py $1 $2 $3 $4 $5 $$

#rm -f /ai/tmpdir/fifo_app.$$
#rm -f /ai/tmpdir/fifo_model.$$
#rm -f /ai/tmpdir/fifo_$$.jpg
#rm -f /ai/tmpdir/fifo_$$.json
