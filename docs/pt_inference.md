

注意事项：

- model.eval() will notify all your layers that you are in eval mode, that way,
 batchnorm or dropout layers will work in eval mode instead of training mode.

- torch.no_grad() impacts the autograd engine and deactivate it. It will reduce 
 memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script).