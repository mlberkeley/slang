"""Abstract class inheriting from the base model class. Provides function definitions for models
   that are variants of or are derived from a variational autoencoder."""
class VAEModel(Model):
    @abstractmethod
    def encode(self, x):
        pass

    @abstractmethod
    def decode(self, mu, var):
        pass
