import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict
from PIL import Image
import os
from util.func import set_random_seed

PART_COLORS = {
    "eye01": (255, 255, 253),
    "eye02": (255, 255, 254),
    "beak": (255, 255, 0),
    "foot01": (255, 0, 1),
    "foot02": (255, 0, 2),
    "wing01": (0, 255, 1),
    "wing02": (0, 255, 2),
    "tail": (0, 0, 255),
}

def get_data(
    dataset: str,
    data_dir: str,
    validation_size: float,
    image_size: int,
    seed: int,
    split_masks_dir: str = None,
):
    """
    Load the proper dataset based on the parsed arguments

    """
    set_random_seed(seed)

    if dataset == 'CUB-200-2011':
        return get_birds(
            True,
            data_dir+'/CUB_200_2011/dataset/train_crop',
            data_dir+'/CUB_200_2011/dataset/train',
            data_dir+'/CUB_200_2011/dataset/test_crop',
            image_size,
            seed,
            validation_size,
            data_dir+'/CUB_200_2011/dataset/train',
            data_dir+'/CUB_200_2011/dataset/test_full',
        )

    if dataset == 'pets':
        return get_pets(
            augment=True,
            train_dir=data_dir+'/PETS/dataset/train',
            project_dir=data_dir+'/PETS/dataset/train',
            test_dir=data_dir+'/PETS/dataset/test',
            img_size=image_size,
            seed=seed,
            validation_size=validation_size,
        )

    # Use --validation_size of 0.2
    if dataset == 'partimagenet':
        return get_part_imagenet(
            augment=True,
            train_dir=data_dir+'/partimagenet/dataset/all',
            project_dir=data_dir+'/partimagenet/dataset/all',
            test_dir=None,
            img_size=image_size,
            seed=seed,
            validation_size=validation_size,
        )

    if dataset == 'CARS':
        return get_cars(
            True,
            data_dir+'/cars/dataset/train',
            data_dir+'/cars/dataset/train',
            data_dir+'/cars/dataset/test',
            image_size,
            seed,
            validation_size,
        )
    if dataset == 'grayscale_example':
        return get_grayscale(
            True,
            data_dir+'/train',
            data_dir+'/train',
            data_dir+'/test',
            image_size,
            seed,
            validation_size,
        )
    if dataset == "FunnyBirds":
        return get_funnybirds(
            augment=True,
            train_dir=data_dir+'/train',
            project_dir=data_dir+'/train',
            test_dir=data_dir+'/test',
            img_size=image_size,
            seed=seed,
            validation_size=validation_size,
            train_dir_pretrain=None,
            test_dir_projection=None,
            online_confound=None,
            split_masks_dir=data_dir+'/train_part_map',
        )


    raise Exception(f'Could not load data set, data set "{dataset}" not found!')


def get_dataloaders(
    dataset: str,
    data_dir: str,
    validation_size: float,
    batch_size_train: int,
    batch_size_pretrain: int,
    image_size: int,
    seed: int,
    num_workers: int,
    disable_cuda: bool,
    weighted_loss: bool,
    split_masks_dir: str = None,
):
    """
    Get data loaders
    """
    # Obtain the dataset
    (
        trainset, 
        trainset_pretraining, 
        trainset_normal, 
        trainset_normal_augment, 
        projectset, 
        testset, 
        testset_projection, 
        classes, 
        num_channels, 
        train_indices, 
        targets,
    ) = get_data(
        dataset=dataset,
        data_dir=data_dir,
        validation_size=validation_size,
        image_size=image_size,
        seed=seed,
        split_masks_dir=split_masks_dir,
    )
    
    # Determine if GPU should be used
    cuda = not disable_cuda and torch.cuda.is_available()
    to_shuffle = True
    sampler = None
    
    num_workers = num_workers
    
    if weighted_loss:
        if targets is None:
            raise ValueError("Weighted loss not implemented for this dataset. Targets should be restructured")
        # https://discuss.pytorch.org/t/dataloader-using-subsetrandomsampler-and-weightedrandomsampler-at-the-same-time/29907
        class_sample_count = torch.tensor([(targets[train_indices] == t).sum() for t in torch.unique(targets, sorted=True)])
        weight = 1. / class_sample_count.float()
        print("Weights for weighted sampler: ", weight, flush=True)
        samples_weight = torch.tensor([weight[t] for t in targets[train_indices]])
        # Create sampler, dataset, loader
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight),replacement=True)
        to_shuffle = False

    trainloader = DataLoader(
        dataset=trainset,
        batch_size=batch_size_train,
        shuffle=to_shuffle,
        sampler=sampler,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(seed),
        drop_last=True,
    )

    if trainset_pretraining is None:
        trainset_pretraining = trainset

    trainloader_pretraining = DataLoader(
        dataset=trainset_pretraining,
        batch_size=batch_size_pretrain,
        shuffle=to_shuffle,
        sampler=sampler,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(seed),
        drop_last=True,
    )

    trainloader_normal = DataLoader(
        dataset=trainset_normal,
        batch_size=batch_size_train,
        shuffle=to_shuffle,
        sampler=sampler,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(seed),
        drop_last=True,
    )

    trainloader_normal_augment = DataLoader(
        dataset=trainset_normal_augment,
        batch_size=batch_size_train,
        shuffle=to_shuffle,
        sampler=sampler,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(seed),
        drop_last=True,
    )

    projectloader = DataLoader(
        dataset=projectset,
        batch_size=batch_size_train,
        shuffle=False,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(seed),
        drop_last=False
    )

    testloader = DataLoader(
        dataset=testset,
        batch_size=batch_size_train,
        shuffle=True,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(seed),
        drop_last=False,
    )

    test_projectloader = DataLoader(
        dataset=testset_projection,
        batch_size=batch_size_train,
        shuffle=False,
        pin_memory=cuda,
        num_workers=num_workers,
        worker_init_fn=np.random.seed(seed),
        drop_last=False,
    )

    print("Num classes (k) = ", len(classes), classes[:5], "etc.", flush=True)
    loaders = {
        "train": trainloader,
        "test": testloader,
        "pretrain": trainloader_pretraining,
        "project": projectloader,
        "test_project": test_projectloader,
        "train_normal": trainloader_normal,
        "train_normal_aug": trainloader_normal_augment,
    }
    return loaders, classes


def create_datasets(transform1, transform2, transform_no_augment, num_channels:int, train_dir:str, project_dir: str, test_dir:str, seed:int, validation_size:float, train_dir_pretrain = None, test_dir_projection = None, transform1p=None, split_masks_dir: str = None):
    
    trainvalset = torchvision.datasets.ImageFolder(train_dir)
    classes = trainvalset.classes
    targets = trainvalset.targets
    indices = list(range(len(trainvalset)))

    train_indices = indices
    
    if test_dir is None:
        if validation_size <= 0.:
            raise ValueError("There is no test set directory, so validation size should be > 0 such that training set can be split.")
        subset_targets = list(np.array(targets)[train_indices])
        train_indices, test_indices = train_test_split(train_indices,test_size=validation_size,stratify=subset_targets, random_state=seed)
        testset = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment), indices=test_indices)
        print("Samples in trainset:", len(indices), "of which",len(train_indices),"for training and ", len(test_indices),"for testing.", flush=True)
    else:
        testset = torchvision.datasets.ImageFolder(test_dir, transform=transform_no_augment)
    
    trainset = torch.utils.data.Subset(
        TwoAugSupervisedDataset(
            trainvalset, 
            transform1=transform1, 
            transform2=transform2,
            split_masks_dir=split_masks_dir,
        ), 
        indices=train_indices
    )
    trainset_normal = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transform_no_augment), indices=train_indices)
    trainset_normal_augment = torch.utils.data.Subset(torchvision.datasets.ImageFolder(train_dir, transform=transforms.Compose([transform1, transform2])), indices=train_indices)
    projectset = torchvision.datasets.ImageFolder(project_dir, transform=transform_no_augment)

    if test_dir_projection is not None:
        testset_projection = torchvision.datasets.ImageFolder(test_dir_projection, transform=transform_no_augment)
    else:
        testset_projection = testset
    if train_dir_pretrain is not None:
        trainvalset_pr = torchvision.datasets.ImageFolder(train_dir_pretrain)
        targets_pr = trainvalset_pr.targets
        indices_pr = list(range(len(trainvalset_pr)))
        train_indices_pr = indices_pr
        if test_dir is None:
            subset_targets_pr = list(np.array(targets_pr)[indices_pr])
            train_indices_pr, test_indices_pr = train_test_split(indices_pr,test_size=validation_size,stratify=subset_targets_pr, random_state=seed)

        trainset_pretraining = torch.utils.data.Subset(
            TwoAugSupervisedDataset(
                trainvalset_pr, 
                transform1=transform1p, 
                transform2=transform2,
                split_masks_dir=split_masks_dir,
            ), 
            indices=train_indices_pr,
        )

    else:
        trainset_pretraining = None

    return trainset, trainset_pretraining, trainset_normal, trainset_normal_augment, projectset, testset, testset_projection, classes, num_channels, train_indices, torch.LongTensor(targets)


def get_funnybirds(
    augment: bool,
    train_dir: str,
    project_dir: str,
    test_dir: str,
    img_size: int,
    seed: int,
    validation_size: float,
    train_dir_pretrain=None,
    test_dir_projection=None,
    online_confound=None,
    split_masks_dir: str = None,
):
    shape = (3, img_size, img_size)
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)
    transform_no_augment = transforms.Compose(
        [
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform1p = None
    if augment:
        transform1 = transforms.Compose(
            [
                transforms.Resize(
                    size=(img_size + 8, img_size + 8)
                ),  # comment out for split
                TrivialAugmentWideNoColor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(img_size + 4, scale=(0.95, 1.0)),
            ]
        )
        transform1p = transforms.Compose(
            [
                transforms.Resize(
                    size=(img_size + 32, img_size + 32)
                ),  # for pretraining, crop can be bigger since it doesn't matter when bird is not fully visible
                TrivialAugmentWideNoColor(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop(img_size + 4, scale=(0.95, 1.0)),
            ]
        )
        transform2 = transforms.Compose(
            [
                TrivialAugmentWideNoShape(),
                transforms.RandomCrop(
                    size=(img_size, img_size)
                ),  # includes crop
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        transform1 = transform_no_augment
        transform2 = transform_no_augment

    return create_datasets(
        transform1,
        transform2,
        transform_no_augment,
        3,
        train_dir,
        project_dir,
        test_dir,
        seed,
        validation_size,
        split_masks_dir=split_masks_dir,
    )


def get_pets(
    augment: bool,
    train_dir: str,
    project_dir: str,
    test_dir: str,
    img_size: int,
    seed: int,
    validation_size: float,
    split_masks_dir: str = None,
):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)

    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    
    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+48, img_size+48)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+8, scale=(0.95, 1.))
        ])
        
        transform2 = transforms.Compose([
            TrivialAugmentWideNoShape(),
            transforms.RandomCrop(size=(img_size, img_size)), #includes crop
            transforms.ToTensor(),
            normalize,
        ])
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size, split_masks_dir=split_masks_dir)


def get_part_imagenet(
    augment: bool,
    train_dir: str,
    project_dir: str,
    test_dir: str,
    img_size: int,
    seed: int,
    validation_size: float,
    split_masks_dir: str = None,
):
    # Validation size was set to 0.2, such that 80% of the data is used for training
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)

    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+48, img_size+48)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+8, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
            TrivialAugmentWideNoShape(),
            transforms.RandomCrop(size=(img_size, img_size)),  # includes crop
            transforms.ToTensor(),
            normalize
        ])
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size, split_masks_dir=split_masks_dir)


def get_birds(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float, train_dir_pretrain = None, test_dir_projection = None, split_masks_dir: str = None): 
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean, std=std)

    transform_no_augment = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+8, img_size+8)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform1p = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), #for pretraining, crop can be bigger since it doesn't matter when bird is not fully visible
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
                            TrivialAugmentWideNoShape(),
                            transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                            transforms.ToTensor(),
                            normalize
                            ])
    else:
        transform1 = transform_no_augment
        transform1p = None
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size, train_dir_pretrain, test_dir_projection, transform1p, split_masks_dir=split_masks_dir)


def get_cars(augment: bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float, split_masks_dir: str = None): 
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.ToTensor(),
                            normalize
                        ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(img_size+4, scale=(0.95, 1.))
        ])
       
        transform2 = transforms.Compose([
                    TrivialAugmentWideNoShapeWithColor(),
                    transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                    transforms.ToTensor(),
                    normalize
                    ])
                            
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size, split_masks_dir=split_masks_dir)


def get_grayscale(augment:bool, train_dir:str, project_dir: str, test_dir:str, img_size: int, seed:int, validation_size:float, train_dir_pretrain = None, split_masks_dir: str = None): 
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,std=std)
    transform_no_augment = transforms.Compose([
                            transforms.Resize(size=(img_size, img_size)),
                            transforms.Grayscale(3), #convert to grayscale with three channels
                            transforms.ToTensor(),
                            normalize
                        ])

    if augment:
        transform1 = transforms.Compose([
            transforms.Resize(size=(img_size+32, img_size+32)), 
            TrivialAugmentWideNoColor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224+8, scale=(0.95, 1.))
        ])
        transform2 = transforms.Compose([
                            TrivialAugmentWideNoShape(),
                            transforms.RandomCrop(size=(img_size, img_size)), #includes crop
                            transforms.Grayscale(3),#convert to grayscale with three channels
                            transforms.ToTensor(),
                            normalize
                            ])
    else:
        transform1 = transform_no_augment    
        transform2 = transform_no_augment           

    return create_datasets(transform1, transform2, transform_no_augment, 3, train_dir, project_dir, test_dir, seed, validation_size, train_dir_pretrain, split_masks_dir=split_masks_dir)


class TwoAugSupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform1, transform2, split_masks_dir=None):
        self.dataset = dataset
        self.classes = dataset.classes
        self.split_masks_dir = split_masks_dir
        if type(dataset) == torchvision.datasets.folder.ImageFolder:
            self.imgs = dataset.imgs
            self.targets = dataset.targets
        else:
            self.targets = dataset._labels
            self.imgs = list(zip(dataset._image_files, dataset._labels))
        self.transform1 = transform1
        self.transform2 = transform2
        self.fmap_size = 26

    def __getitem__(self, index):
        image, target = self.dataset[index]

        # Load corresponding mask if mask directory is provided
        mask = None
        masks = None
        if self.split_masks_dir:
            # Get image filename from the dataset
            img_path = self.imgs[index][0]
            img_filename = os.path.basename(img_path)

            # Construct mask path with same filename
            mask_path = os.path.join(
                self.split_masks_dir,
                str(self.classes[target]),
                img_filename,
            )

            # Load mask
            if os.path.exists(mask_path):
                mask = Image.open(mask_path).convert("RGB")  # Load as RGB
                # Convert mask to tensor
                mask = transforms.ToTensor()(mask)
            else:
                raise FileNotFoundError(
                    f"Mask not found for image {img_filename}"
                )

            def process_mask(mask_tensor, parts):
                mask_gray = torch.zeros_like(
                    mask_tensor[0]
                )  # Use first channel as reference
                for part in parts:
                    part_color = (
                        torch.tensor(PART_COLORS[part]).float() / 255.0
                    )
                    # Compare each channel
                    match = torch.all(
                        torch.abs(mask_tensor - part_color.view(3, 1, 1))
                        < 0.01,
                        dim=0,
                    )
                    mask_gray += match.float()

                mask_gray = mask_gray.unsqueeze(0)  # Add channel dimension

                return mask_gray

            def resize_mask(mask_gray):

                # Resize to 26x26
                mask_gray = torch.nn.functional.interpolate(
                    mask_gray.unsqueeze(0), size=(self.fmap_size, self.fmap_size), mode="bilinear"
                ).squeeze(0)
                return (mask_gray >= 0.5).float()

            mask_gray_wings = process_mask(mask, ("wing01", "wing02"))
            mask_gray_tail = process_mask(mask, ("tail",))
            masks = [mask_gray_wings, mask_gray_tail]

            # Save initial state
            # save_transform_debug_visualizations(
            #     image, mask_gray_wings, index, stage="initial", mask_type="wings"
            # )
            # save_transform_debug_visualizations(
            #     image, mask_gray_tail, index, stage="initial", mask_type="tail"
            # )

        # Set seed for reproducibility
        torch.manual_seed(index)

        # Get the random state for transform1
        state = torch.get_rng_state()
        image = self.transform1(image)

        if masks is not None:
            transformed_masks = []
            for i, m in enumerate(masks):
                # Reset random state to ensure same transformation
                torch.set_rng_state(state)
                mask_transformed = self.transform1(m)

                # save_transform_debug_visualizations(
                #     image,
                #     mask_transformed,
                #     index,
                #     stage="after_transform1",
                #     mask_type="wings" if i == 0 else "tail",
                # )
                mask_26 = resize_mask(mask_transformed)
                transformed_masks.append(mask_26)

        # Continue with rest of transformations
        img1 = self.transform2(image)
        img2 = self.transform2(image)

        if mask is not None:
            return img1, img2, transformed_masks, target
        return img1, img2, target

    def __len__(self):
        return len(self.dataset)


# function copied from:
# https://pytorch.org/vision/stable/_modules/torchvision/transforms/autoaugment.html#TrivialAugmentWide
# (v0.12) and adapted
class TrivialAugmentWideNoColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.5, num_bins), True), 
            "ShearY": (torch.linspace(0.0, 0.5, num_bins), True), 
            "TranslateX": (torch.linspace(0.0, 16.0, num_bins), True), 
            "TranslateY": (torch.linspace(0.0, 16.0, num_bins), True), 
            "Rotate": (torch.linspace(0.0, 60.0, num_bins), True), 
        }


class TrivialAugmentWideNoShapeWithColor(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.5, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }


class TrivialAugmentWideNoShape(transforms.TrivialAugmentWide):
    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[torch.Tensor, bool]]:
        return {
            
            "Identity": (torch.tensor(0.0), False),
            "Brightness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Color": (torch.linspace(0.0, 0.02, num_bins), True), 
            "Contrast": (torch.linspace(0.0, 0.5, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.5, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            "AutoContrast": (torch.tensor(0.0), False),
            "Equalize": (torch.tensor(0.0), False),
        }
