import torch
import numpy as np



def model(t_u:torch.Tensor, w:torch.Tensor, b:torch.Tensor):
    return w * t_u + b

def loss_fn(t_p:torch.Tensor, t_c:torch.Tensor) -> torch.Tensor:
    squared_diff = (t_p - t_c)**2
    return squared_diff.mean()


# _Start: define gradients
def dloss_fn(t_p:torch.Tensor, t_c:torch.Tensor) -> torch.Tensor:
    dsq_diffs = 2 * (t_p - t_c) / t_p.size(0)
    return dsq_diffs

def dmodel_dw(t_u:torch.Tensor, w:torch.Tensor, b:torch.Tensor) -> torch.Tensor:
    return t_u

def dmodel_db(t_u:torch.Tensor, w:torch.Tensor, b:torch.Tensor):
    return 1.0

def grad_fn(t_u, t_c, t_p, w, b):
    dloss_dtp = dloss_fn(t_p, t_c)
    dloss_dw = dloss_dtp * dmodel_dw(t_u, w, b)
    dloss_db = dloss_dtp * dmodel_db(t_u, w, b)
    return torch.stack([dloss_dw.sum(), dloss_db.sum()])
# _End: define gradients


# _Start: define training loop
def training_loop(n_epochs:int, learning_rate:float, params:torch.Tensor, t_u:torch.Tensor, t_c:torch.Tensor):
    for epoch in range(1, n_epochs + 1):
        w, b = params

        t_p = model(t_u, w, b)
        loss = loss_fn(t_p, t_c)
        grad = grad_fn(t_u, t_c, t_p, w, b)

        params = params - learning_rate * grad

        print('Epoch %d, Loss %f' % (epoch, float(loss)))
        print("params: ", params)
        print("grad: ", grad)
    return params
# _End: define training loop



def main():
    # _Start: create data
    t_c = [0.5, 14.0, 15.0, 28.0, 11.0, 8.0, 3.0, -4.0, 6.0, 13.0, 21.0]     # currect
    t_u = [35.7, 55.9, 58.2, 81.9, 56.3, 48.9, 33.9, 21.8, 48.4, 60.4, 68.4] # input
    t_c = torch.tensor(t_c)
    t_u = torch.tensor(t_u)
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
        n_epochs=100, learning_rate=1e-4,
        params= torch.tensor([1.0, 0.0]), t_u=t_u, t_c=t_c )
    # _End: weight updates with gradient



if __name__ == "__main__":
    main()