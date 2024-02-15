import numpy as array_api
from .autograd import Tensor
import gzip
import struct
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
NDArray = array_api.ndarray

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = array_api.random.rand() < self.p
        ### BEGIN YOUR SOLUTION
        # if not flip_img:
        #     return img 
        # i = 0
        # j = img.shape[1] - 1
        # while i < j:
        #     a = img[:, i, :].copy()
        #     img[:, i, :] = img[:, j, :]
        #     img[:, j, :] = a
        #     i += 1
        #     j -= 1
        # return img
        if flip_img:
            return array_api.flip(img, axis=1)
        return img
        ### END YOUR SOLUTION


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return 
            H x W x C NDArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = array_api.random.randint(low=-self.padding, high=self.padding+1, size=2)
        ### BEGIN YOUR SOLUTION
        res = array_api.zeros(shape=[img.shape[0] + 2 * self.padding,
                                    img.shape[1] + 2 * self.padding,
                                    img.shape[2]])
        start_x = start_y = self.padding
        start_x += shift_x
        start_y += shift_y
        res[self.padding : self.padding + img.shape[0],
            self.padding : self.padding + img.shape[1],
            :] = img
        return res[start_x : start_x + img.shape[0],
                    start_y : start_y + img.shape[1],
                    :]
        ### END YOUR SOLUTION


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        indices = array_api.arange(len(dataset))
        if self.shuffle:
            array_api.random.shuffle(indices)
        self.ordering = array_api.array_split(indices, range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        self.ptr = 0
        ### END YOUR SOLUTION
        return self

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.ptr == len(self.ordering):
            raise StopIteration
        cur_batch = self.ptr
        self.ptr += 1
        res = [Tensor(image) for image in self.dataset[self.ordering[cur_batch]]]
        return tuple(res)
        ### END YOUR SOLUTION


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        self.images: NDArray
        self.labels: NDArray
        
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        # get filename
        new_paths = []
        for path in [image_filename, label_filename]:
            new_paths.append(path[:path.find(".gz")])
            
        # decompressing
        for path in new_paths:        
            with gzip.GzipFile(filename=path+".gz", mode='rb') as uzf:
                with open(file=path, mode = "wb") as wf:
                    wf.write(uzf.read())
                print('decompression done')
                
        # reading X      
        with open(file=new_paths[0], mode='rb') as uzx:
            mg_num = struct.unpack(">i", uzx.read(4))[0]
            num_examples = struct.unpack(">i", uzx.read(4))[0]
            height = struct.unpack(">i", uzx.read(4))[0]
            width = struct.unpack(">i", uzx.read(4))[0]
            input_dim = height * width
            print(mg_num, num_examples, height, width, input_dim)
            
            res_X = array_api.ndarray(shape=(num_examples, input_dim), dtype=array_api.dtype(array_api.float32))
            temp_fmt = ">" + "B" * input_dim
            for i in range(num_examples):
                res_X[i] = struct.unpack(temp_fmt, uzx.read(input_dim))
            # normalizing 
            self.images = res_X / 255.0
            
        # reading y
        with open(file=new_paths[1], mode='rb') as uzy:        
            mg_num = struct.unpack(">i", uzy.read(4))[0]
            num_labels = struct.unpack(">i", uzy.read(4))[0]
            print(mg_num, num_labels)
            
            temp_fmt = ">" + "B" * num_labels
            self.labels = array_api.array(struct.unpack(temp_fmt, uzy.read(num_labels)), dtype=array_api.dtype(array_api.uint8))
        
        
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        (res_X, res_y) = (self.images[index], self.labels[index])
        if self.transforms:
            res_X = self.apply_transforms(res_X.reshape((28, 28, -1))).reshape(-1, 28 * 28)
        return (res_X, res_y)
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return len(self.images)
        ### END YOUR SOLUTION

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
