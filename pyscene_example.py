# Standard PySceneDetect imports:
from scenedetect import VideoManager
from scenedetect import SceneManager
import pdb
import argparse

# For content-aware scene detection:
from scenedetect.detectors import ContentDetector

def find_scenes(video_path, threshold=30.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold))

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list()

def main(args):
    scenes = find_scenes(args.video, threshold=args.threshold)
    print(scenes)
    pdb.set_trace()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test out PySceneDetect.')
    parser.add_argument('--threshold', default=30, type=int, help='Pyscene threshold')
    parser.add_argument('--video', default='/Users/ericrobinson/screen_counter/originals/004.mp4', type=str, help='video to PySceneDetect')
    args = parser.parse_args()
    main(args)