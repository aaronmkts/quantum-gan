
import tensorflow as tf
import numpy as np
import scipy.misc 
from io import BytesIO         # Python 3.x
from PIL import Image  

class Logger(object):
    
    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = tf.summary.create_file_writer(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        with self.writer.as_default():
            tf.summary.scalar(tag, value, step=step)

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        with self.writer.as_default():
            for i, img in enumerate(images):
                # Convert numpy array to an image
                img = Image.fromarray(img)
                buffer = BytesIO()
                img.save(buffer, format="PNG")
                img_bytes = buffer.getvalue()

                # Write the image summary
                tf.summary.image(f'{tag}/{i}', [tf.image.decode_image(img_bytes)], step=step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        with self.writer.as_default():
            # Create a histogram using numpy
            counts, bin_edges = np.histogram(values, bins=bins)
            
            # Drop the start of the first bin
            bin_edges = bin_edges[1:]

            # Create and write Summary
            tf.summary.histogram(tag, values, step=step)
        self.writer.flush()