import sys, os
import tensorflow as tf

class Visualizer:
    def __init__(self, log_dir, sess = None):
        self.log_dir = log_dir
        self.sess = sess if sess is not None else tf.Session()
        self.embeddings = None

    # Starts a local HTTP server that can be used through tensorboard to visualize the saved embeddings with PCA or tSNE.
    def visualize(embeddings = None):
        if not self.embeddings and not embeddings:
            print("No embeddings loaded")
            sys.exit(1)

        writer = tf.summary.FileWriter(os.path.join(self.log_dir, "projector"), self.sess.graph)

        config = projector.ProjectorConfig()
        embed = config.embeddings.add()
        embed.tensor_name = self.embeddings.name

        projector.visualize_embeddings(writer, config)

        tf.train.Saver().save(self.sess, os.path.join(self.log_dir, "projector", "embeddings.ckpt"))

    def load_embeddings(checkpoints_path, embedding_variable_name):
        tf.train.Saver.restore(self.sess, checkpoints_path)
        self.embeddings = tf.get_default_graph().get_tensor_by_name(embedding_name)

def main():
    log_dir = "./logs"
    projector_dir = os.path.join(log_dir, "projector")

    # Clean up old visualization data
    if tf.gfile.Exists(projector_dir):
        tf.gfile.DeleteRecursively(projector_dir)
        tf.gfile.MkDir(projector_dir)

    # Restore embeddings and visualize - access the TensorBoard HTTP server at localhost:8000
    with tf.Session() as sess:
        model = Model({"load": True, "dir": "../src"}, gpu_fraction = 0.0) # I don't have a GPU
        v = Visualize(log_dir, model.sess)

        with tf.variable_scope("encoder"):
            embeddings = model.sess.run(tf.get_variable("encode_outputs"))[:, -1, :]
            visualize(embeddings)

if __name__ == "__main__":
    tf.app.run(main = main)
