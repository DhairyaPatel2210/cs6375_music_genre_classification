from data_loader import train_x, train_y, test_x, test_y
from Network import Network
from Layers import ActivationLayer, DeepLayer
from Layers.activations import RelU, RelU_prime, sigmoid, sigmoid_prime
from Layers.losses import cross_entropy_loss, cross_entropy_prime_loss
from Configuration.config import EPOCHS, LEARNING_RATE, REGULARIZATION, LAMBD, BATCH_SIZE, B1, B2, EPSILON

def main():
    net = Network()
    input_shape = train_x.shape[0]

    # adding the input layer, 
    net.add(DeepLayer(input_shape, 128))
    net.add(ActivationLayer(RelU, RelU_prime, dropout=True, keep_prob=0.6))

    # hidden layer with total 128 neurons in it
    net.add(DeepLayer(128, 64))
    net.add(ActivationLayer(RelU, RelU_prime, dropout=True, keep_prob=0.7))

    # hidden laer with total 64 neurons
    net.add(DeepLayer(64, 32))
    net.add(ActivationLayer(RelU, RelU_prime, dropout=True, keep_prob=0.8))
    
    # hidden layer with total 32 neurons
    net.add(DeepLayer(32, 10))
    net.add(ActivationLayer(sigmoid, sigmoid_prime))

    # the output layer has 10 categories so it will have 10 neurons
    net.use(cross_entropy_loss, cross_entropy_prime_loss)

    net.fit(
        train_x, 
        train_y, 
        test_x, 
        test_y, 
        epochs=EPOCHS, 
        learning_rate=LEARNING_RATE, 
        regularization=REGULARIZATION, 
        lambd=LAMBD,
        batch_size = BATCH_SIZE,
        beta1 = B1, 
        beta2 = B2,
        epsilon = EPSILON
    )

if __name__ == "__main__":
    main()
    
