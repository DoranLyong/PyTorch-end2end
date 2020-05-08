import torch
import torch.optim as optim

print("Optimizer: ", dir(optim))

# _Start: design your model
def model(t_u:torch.Tensor, w:torch.Tensor, b:torch.Tensor):
    return w * t_u + b
# _End: design your model


# _Start: define the loss function.
def loss_fn(t_p:torch.Tensor, t_c:torch.Tensor) -> torch.Tensor:
    squared_diff = (t_p - t_c)**2
    return squared_diff.mean()
# _End: define the loss function.


# _Start: define training loop with autograd
def training_loop(n_epochs, optimizer, params, train_t_u, val_t_u, train_t_c, val_t_c):
    for epoch in range(1, n_epochs + 1):
        train_t_p = model(train_t_u, *params)
        train_loss = loss_fn(train_t_p, train_t_c)

        val_t_p = model(val_t_u, *params)
        val_loss = loss_fn(val_t_p, val_t_c)

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if epoch <= 3 or epoch % 500 == 0:
            print('Epoch {}, Training loss {}, Validation loss {}'.format(
                epoch, float(train_loss), float(val_loss)))

    return params
# _End: define training loop with autograd



def main():
    # _Start: create data
    t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]     # currect
    t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] # input
    t_c = torch.tensor(t_c)
    t_u = torch.tensor(t_u)
    t_un = 0.1 * t_u
    # _End: Create data


    # _Start: split data into 'training' and 'validation'
    n_samples = t_u.shape[0]
    n_val = int(0.2 * n_samples)

    shuffled_indices = torch.randperm(n_samples)

    train_indices = shuffled_indices[:-n_val]
    val_indices = shuffled_indices[-n_val:]

    train_t_u = t_u[train_indices]
    train_t_c = t_c[train_indices]

    val_t_u = t_u[val_indices]
    val_t_c = t_c[val_indices]

    train_t_un = 0.1 * train_t_u
    val_t_un = 0.1 * val_t_u
    # _End: split data into 'training' and 'validation'


    # _Start: set a linear model
    w = torch.ones(())              # set broadcasting
    b = torch.zeros(())
    t_p = model(t_u=t_u, w=w, b=b)  # target prediction by my model
    print("Model prediction: ", t_p)
    # _End: set a linear model


    # _Start: return loss function
    loss = loss_fn(t_p=t_p, t_c=t_c)
    print("Loss:", loss)
    # _End: return loss function


    # _Start: set hyperparameters
    params = torch.tensor([1.0, 0.0], requires_grad=True)
    learning_rate = 1e-2
    optimizer = optim.SGD([params], lr=learning_rate)
    # _End: set hyperparameters


    # _Start: weight updates with gradient
    training_loop(
        n_epochs=3000,
        optimizer=optimizer,
        params=params,
        train_t_u=train_t_un,
        val_t_u=val_t_un,
        train_t_c=train_t_c,
        val_t_c=val_t_c)
    # _End: weight updates with gradient
    print("End")


if __name__ == "__main__":
    main()