import av

def get_video_frames(video_path):
  vid = av.open(video_path)
  return [f for f in vid.decode(video=0)]

class StreamWrapper(object):
  def __init__(self, stream):
    self.stream = stream

  def read(self, len):
    return self.stream.read(len)

  def close(self):
    return self.stream.close()
