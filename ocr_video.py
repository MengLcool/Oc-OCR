import sys
sys.path.append('..')
from text_detection_recognition.text_craft import TextCraft

if __name__ == '__main__' :
    import os 

    if len(sys.argv) > 5 :
        video_name = sys.argv[1]
        result_file = sys.argv[2]
        start_frame = int(sys.argv[3])
        video_length = int(sys.argv[4])
        gpu_id = int(sys.argv[5])

    else :
        print("five parameters needed for video: videoName, resultFile, startFrame, videoLength, gpuID")
        sys.exit(-1)

    print(sys.argv[1:6])
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    text_craft = TextCraft()
    text_craft.processVideoFile(video_name, result_file, start_frame, video_length)
