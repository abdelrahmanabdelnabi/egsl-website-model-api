import av

def get_video_frames(video_path):
  vid = av.open(video_path)
  return [f for f in vid.decode(video=0)]