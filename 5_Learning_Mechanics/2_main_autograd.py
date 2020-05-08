import torch


# _Start: define your model
def model(t_u:torch.Tensor, w:torch.Tensor, b:torch.Tensor):
    return w * t_u + b
# _End: define your model


# _Start: define the loss function
def loss_fn(t_p:torch.Tensor, t_c:torch.Tensor) -> torch.Tensor:
    squared_diff = (t_p - t_c)**2
    return squared_diff.mean()
# _End: define the loss function


# _Start: define training loop with autograd
def training_loop(n_epochs, learning_rate, params, t_u, t_c):
    for epoch in range(1, n_epochs + 1):
        if params.grad is not None:  # <1>
            params.grad.zero_()

        t_p = model(t_u, *params)
        loss = loss_fn(t_p, t_c)
        loss.backward()

        with torch.no_grad():  # <2>
            params -= learning_rate * params.grad

        if epoch % 500 == 0:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))

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


    # _Start: weight updates with gradient
    training_loop(
        n_epochs=5000,
        learning_rate=1e-2,
        params=torch.tensor([1.0, 0.0], requires_grad=True),  # <1>
        t_u=t_un,  # <2>
        t_c=t_c)
    # _End: weight updates with gradient

    print("End...")


if __name__ == "__main__":
    main()