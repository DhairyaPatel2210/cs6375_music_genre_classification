import numpy as np
from sklearn.model_selection import train_test_split

# Assume x_train, y_train, x_val, y_val are your data arrays

# Hyperparameters
input_size = x_train.shape[-1]
hidden_size = 64
output_size = 10
learning_rate = 0.001
epochs = 60
batch_size = 32

# LSTM model parameters initialization
W_ih = np.random.randn(4 * hidden_size, input_size)
W_hh = np.random.randn(4 * hidden_size, hidden_size)
W_ho = np.random.randn(output_size, hidden_size)
b_ih = np.zeros((4 * hidden_size, 1))
b_hh = np.zeros((4 * hidden_size, 1))
b_ho = np.zeros((output_size, 1))

# Data splitting
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# One-hot encode the labels
y_train_onehot = np.eye(output_size)[y_train]
y_val_onehot = np.eye(output_size)[y_val]

# Training loop
for epoch in range(epochs):
    # Iterate over batches
    for i in range(0, len(x_train), batch_size):
        x_batch = x_train[i:i + batch_size]
        y_batch = y_train_onehot[i:i + batch_size]

        # Forward pass
        h = np.zeros((hidden_size, 1))
        c = np.zeros((hidden_size, 1))
        output_sequence = []

        for t in range(len(x_batch)):
            x_t = x_batch[t].reshape(-1, 1)

            gates = np.dot(W_ih, x_t) + np.dot(W_hh, h) + b_ih + b_hh
            i_t = sigmoid(gates[:hidden_size])
            f_t = sigmoid(gates[hidden_size:2 * hidden_size])
            o_t = sigmoid(gates[2 * hidden_size:3 * hidden_size])
            c_tilde_t = tanh(gates[3 * hidden_size:])

            c = f_t * c + i_t * c_tilde_t
            h = o_t * tanh(c)

            output_sequence.append(h)

        output_sequence = np.array(output_sequence)
        final_hidden_state = output_sequence[-1].reshape(-1, 1)

        # Loss calculation
        loss = -np.sum(y_batch * np.log(softmax(np.dot(W_ho, final_hidden_state) + b_ho)))

        # Backward pass
        delta_out = softmax(np.dot(W_ho, final_hidden_state) + b_ho) - y_batch
        dW_out = np.outer(delta_out, final_hidden_state)
        db_out = delta_out

        dnext_h = np.zeros_like(h)
        dnext_c = np.zeros_like(c)

        # Backward pass through time
        for t in reversed(range(len(x_batch))):
            x_t = x_batch[t].reshape(-1, 1)
            h_t = output_sequence[t].reshape(-1, 1)
            y_true_t = y_batch[t]

            # Backpropagation through the dense layer
            dW_out += np.outer(delta_out, h_t)
            db_out += delta_out

            # Backpropagation through the LSTM layer
            dx, dprev_h, dprev_c, dW_ih_t, dW_hh_t, db_ih_t, db_hh_t = \
                lstm_backward_step(delta_out, dnext_c, *h_t, x_t, W_ih, W_hh)

            # Accumulate gradients
            dW_ih += dW_ih_t
            dW_hh += dW_hh_t
            db_ih += db_ih_t
            db_hh += db_hh_t

            # Update deltas for the next step
            dnext_h = dprev_h
            dnext_c = dprev_c

        # Update parameters
        W_ih = update_parameters(W_ih, dW_ih, learning_rate)
        W_hh = update_parameters(W_hh, dW_hh, learning_rate)
        W_ho = update_parameters(W_ho, dW_out, learning_rate)
        b_ih = update_parameters(b_ih, db_ih, learning_rate)
        b_hh = update_parameters(b_hh, db_hh, learning_rate)
        b_ho = update_parameters(b_ho, db_out, learning_rate)

    # Validation
    val_loss = 0
    for t in range(len(x_val)):
        x_t = x_val[t].reshape(-1, 1)
        y_true_t = y_val_onehot[t]

        gates = np.dot(W_ih, x_t) + np.dot(W_hh, h) + b_ih + b_hh
        i_t = sigmoid(gates[:hidden_size])
        f_t = sigmoid(gates[hidden_size:2 * hidden_size])
        o_t = sigmoid(gates[2 * hidden_size:3 * hidden_size])
        c_tilde_t = tanh(gates[3 * hidden_size:])

        c = f_t * c + i_t * c_tilde_t
        h = o_t * tanh(c)

        val_loss += -np.sum(y_true_t * np.log(softmax(np.dot(W_ho, h) + b_ho)))

    val_loss /= len(x_val)

    # Print loss for monitoring training progress
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Training Loss: {loss}, Validation Loss: {val_loss}")

# Save the trained model
np.savez("GTZAN_LSTM.npz", W_ih=W_ih, W_hh=W_hh, W_ho=W_ho, b_ih=b_ih, b_hh=b_hh, b_ho=b_ho)

# Test the trained model on a new input sequence
test_input_sequence = np.random.randn(input_size, sequence_length)
h = np.zeros((hidden_size, 1))
c = np.zeros((hidden_size, 1))

for t in range(sequence_length):
    x_t = test_input_sequence[:, t].reshape(-1, 1)

    gates = np.dot(W_ih, x_t) + np.dot(W_hh, h) + b_ih + b_hh
    i_t = sigmoid(gates[:hidden_size])
    f_t = sigmoid(gates[hidden_size:2 * hidden_size])
    o_t = sigmoid(gates[2 * hidden_size:3 * hidden_size])
    c_tilde_t = tanh(gates[3 * hidden_size:])

    c = f_t * c + i_t * c_tilde_t
    h = o_t * tanh(c)

test_output_probabilities = softmax(np.dot(W_ho, h) + b_ho)

# Print the output probabilities for the test sequence
print("Test Output Probabilities:")
print(test_output_probabilities)
