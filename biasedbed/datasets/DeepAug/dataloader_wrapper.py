from __future__ import print_function
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import torch

class MyIter(object):
    """An iterator."""

    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        batches = [loader_iter.__next__() for loader_iter in self.loader_iters]
        return self.my_loader.combine_batch(batches)

    # Python 2 compatibility
    next = __next__

    def __len__(self):
        return len(self.my_loader)


class Wrapper(object):
    """This class wraps several pytorch DataLoader objects, allowing each time
    taking a batch from each of them and then combining these several batches
    into one. This class mimics the `for batch in loader:` interface of
    pytorch `DataLoader`.
    Args:
      loaders: a list or tuple of pytorch DataLoader objects
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return MyIter(self)

    def __len__(self):
        bs = len([loader.batch_size for loader in self.loaders])
        return sum([len(loader) for loader in self.loaders])//bs

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        images,labels = zip(*batches)
        return torch.cat(images,0),torch.cat(labels,0)



if __name__ == "__main__":
    from tqdm import tqdm
    mnist = torchvision.datasets.MNIST(
            'data', train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))

    loader1 = DataLoader(
        mnist,
        batch_size=6, shuffle=True)

    loader2 = DataLoader(
        mnist,
        batch_size=6, shuffle=True)

    loader3 = DataLoader(
        mnist,
        batch_size=6, shuffle=True)

    my_loader = Wrapper([loader1, loader2, loader3])

    print(len(mnist))
    print(len(my_loader))
    for x,y in tqdm(my_loader):
        (x.shape)


