from tqdm import tqdm
import argparse
import torch
import torch.utils.data
import os
from PIL import Image, ImageDraw as D
import torchvision.transforms as transforms
import torchvision
from util.func import get_patch_size

from torch.utils.data import DataLoader

from pipnet.pipnet import PIPNet
from typing import Tuple, List, Dict, Set


class TopKProtoActivations:
    def __init__(
        self,
        k: int,
        device: torch.device,
        relevance_threshold: float = 0.1,
    ):
        self.k = k
        self.top_ks = {}
        self.device = device
        self.relevance_threshold = relevance_threshold

    @property
    def protos_idxs(self):
        """
        Returns all prototypes indices
        present in top K activations record.
        """
        return self.top_ks.keys()

    def add_proto_activations(
        self,
        proto_idx: int,
        proto_scores: torch.Tensor,
        proto_image_coords: Tuple[torch.Tensor, ...],
        images: torch.tensor,
    ):
        """
        Adds new prototype activation data if possible.
        """
        batch_size = images.shape[0]

        # Extract image coords;
        y_min, y_max, x_min, x_max = proto_image_coords
        crop_h = (y_max - y_min)[0].item()
        crop_w = (x_max - x_min)[0].item()

        # Initialize a tensor to hold the crops
        image_crops = torch.zeros((batch_size, 3, crop_h, crop_w))
        image_crops = image_crops.to(device=self.device)

        # Crop patches from the images
        for i in range(batch_size):
            image_crops[i] = images[i, :, y_min[i]:y_max[i], x_min[i]:x_max[i]]

        # Select and store top k activations with visualizations;
        if proto_idx in self.top_ks:
            # Concatenate stored and new activation data;
            proto_scores = torch.cat([proto_scores, self.top_ks[proto_idx][0]], dim=0)
            image_crops = torch.cat([image_crops, self.top_ks[proto_idx][1]], dim=0)

        # Sort proto data by activations;
        proto_scores, sorted_indices = torch.sort(proto_scores, descending=True)
        image_crops = image_crops[sorted_indices]

        # Store top k;
        self.top_ks[proto_idx] = (proto_scores[:self.k], image_crops[:self.k])

    def save_visualizations(self, save_dir: str):

        all_image_crops = []

        # Visualize individual prototypes activations;
        for proto_idx in self.protos_idxs:

            # Convert proto idx to visualized tensor;
            image_crops = self.top_ks[proto_idx][1]

            proto_idx_crop = self.proto_idx_to_crop(
                idx=proto_idx,
                crop_h=image_crops.shape[2],
                crop_w=image_crops.shape[3],
            )

            image_crops = torch.cat([image_crops, proto_idx_crop], dim=0)

            grid = torchvision.utils.make_grid(
                tensor=image_crops,
                nrow=self.k+1,  # <- +1 for proto index visualization;
                padding=1,
            )
            torchvision.utils.save_image(
                tensor=grid,
                fp=os.path.join(save_dir, f"grid_topk_{proto_idx}.png"),
            )

            all_image_crops += image_crops

        # Visualize all prototypes activations;
        if len(all_image_crops) > 0:
            grid = torchvision.utils.make_grid(
                tensor=all_image_crops,
                nrow=self.k+1,
                padding=1,
            )
            torchvision.utils.save_image(
                tensor=grid,
                fp=os.path.join(save_dir, "grid_topk_all.png"),
            )
        else:
            print(
                "Pretrained prototypes not visualized. "
                "Try to pretrain longer.",
                flush=True,
            )

    def discard_irrelevant_protos(self):
        """
        Removes prototypes without any relevant activation score.
        """
        for proto_idx in self.top_ks.keys():
            found = False

            for proto_score in self.top_ks[proto_idx][0]:
                if proto_score > self.relevance_threshold:
                    found = True
                    break

            if not found:
                del self.top_ks[proto_idx]

    def proto_idx_to_crop(self, idx: int, crop_h: int, crop_w: int) -> torch.Tensor:
        """
        Converts index to visualization tensor.
        """
        text = f"P {idx}"
        txtimage = Image.new(
            mode="RGB",
            size=(crop_h, crop_w),
            color=(0, 0, 0),
        )
        draw = D.Draw(txtimage)
        draw.text(
            xy=(crop_h // 2, crop_w // 2),
            text=text,
            anchor='mm',
            fill="white",
        )
        return transforms.ToTensor()(txtimage)


def extract_max_hw_idxs(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert len(x.shape) == 4, "To extract max HW indices pass tensor of shape (B, C, H, W)!"

    _, h_max_idx = torch.max(x, dim=2)
    h_max_idx = torch.max(h_max_idx, dim=2).indices

    _, w_max_idx = torch.max(x, dim=3)
    w_max_idx = torch.max(w_max_idx, dim=2).indices

    return h_max_idx, w_max_idx


def get_irrelevant_class_protos(
    network: PIPNet,
    relevance_thresh: float = 1e-3,
) -> Set[int]:
    """
    Returns indices of prototypes
    that do not have greater classification weight
    that relevance_thresh for each class.
    """
    irrelevant_proto_idxs = set()
    num_prototypes = network.module.get_num_prototypes()
    class_weight = network.module.get_class_weight()

    for proto_idx in range(num_prototypes):
        proto_max_class_weight = torch.max(class_weight[:, proto_idx])
        if proto_max_class_weight <= relevance_thresh:
            irrelevant_proto_idxs.add(proto_idx)

    return irrelevant_proto_idxs


@torch.no_grad()
def visualize_topk(
    network: PIPNet,
    projectloader: DataLoader,
    device: torch.device,
    save_dir: str,
    image_hw_dims: Tuple[int, int],
    k: int = 10,
) -> List[int]:
    """
    Visualizes top k prototypes activations form project_loader.
    returns indices of 'relevant' prototypes
    (those with high enough activation score).
    """
    print("Visualizing prototypes for top K...", flush=True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Make sure the model is in evaluation mode
    network.eval()

    # Find irrelevant prototypes indices
    # (based on classification relevance)
    irrelevant_proto_idxs = get_irrelevant_class_protos(network)

    # Extract shapes;
    num_prototypes = network.module.get_num_prototypes()
    patch_size, skip = get_patch_size(
        image_size=image_hw_dims[0],
        num_prototypes=num_prototypes,
    )

    # Show progress on progress bar;
    img_iter = tqdm(
        enumerate(projectloader),
        total=len(projectloader),
        mininterval=50.0,
        desc=f'Collecting top {k} proto activations',
        ncols=0,
    )

    # Initialize storage for top K activations
    top_k_proto_act = TopKProtoActivations(k=k, device=device)

    # Iterate through the projection set;
    for batch_idx, (x, _) in img_iter:
        x = x.to(device)

        # Pass image through the network;
        model_output = network(x, inference=True)

        # Extract prototype features;
        proto_feature_vec = model_output.proto_feature_vec
        proto_feature_map = model_output.proto_feature_map
        h_max_idxs, w_max_idxs = extract_max_hw_idxs(proto_feature_map)

        for proto_idx in range(num_prototypes):
            # Omit irrelevant protos;
            if proto_idx in irrelevant_proto_idxs:
                continue

            # Convert featuremap coords to image coords;
            proto_h_max_idxs = h_max_idxs[:, proto_idx]
            proto_w_max_idxs = w_max_idxs[:, proto_idx]
            proto_scores = proto_feature_vec[:, proto_idx]

            proto_image_coords = latent_to_image_coords(
                h_idxs=proto_h_max_idxs,
                w_idxs=proto_w_max_idxs,
                latent_hw_dims=(proto_feature_map.shape[2], proto_feature_map.shape[3]),
                image_hw_dims=image_hw_dims,
                rec_field_hw_dims=(patch_size, patch_size),
                rec_field_hw_offset=(skip, skip),
            )

            # Add top K activations;
            top_k_proto_act.add_proto_activations(
                proto_idx=proto_idx,
                proto_scores=proto_scores,
                proto_image_coords=proto_image_coords,
                images=x,
            )

    # Discard irrelevant prototypes;
    top_k_proto_act.discard_irrelevant_protos()

    # Visualize receptive fields;
    top_k_proto_act.save_visualizations(save_dir=save_dir)

    # Return indices of relevant prototypes;
    return list(top_k_proto_act.protos_idxs)


# TODO: move to visualize prediction
def visualize(net, projectloader, num_classes, device, foldername, args: argparse.Namespace):
    print("Visualizing prototypes...", flush=True)
    dir = os.path.join(args.log_dir, foldername)
    if not os.path.exists(dir):
        os.makedirs(dir)

    near_imgs_dirs = dict()
    seen_max = dict()
    saved = dict()
    saved_ys = dict()
    tensors_per_prototype = dict()
    abstainedimgs = set()
    notabstainedimgs = set()
    
    for p in range(net.module._num_prototypes):
        near_imgs_dir = os.path.join(dir, str(p))
        near_imgs_dirs[p]=near_imgs_dir
        seen_max[p]=0.
        saved[p]=0
        saved_ys[p]=[]
        tensors_per_prototype[p]=[]
    
    patchsize, skip = get_patch_size(args)

    imgs = projectloader.dataset.imgs
    
    # skip some images for visualisation to speed up the process
    if len(imgs)/num_classes <10:
        skip_img=10
    elif len(imgs)/num_classes < 50:
        skip_img=5
    else:
        skip_img = 2
    print("Every", skip_img, "is skipped in order to speed up the visualisation process", flush=True)

    # Make sure the model is in evaluation mode
    net.eval()
    classification_weights = net.module._classification.weight
    # Show progress on progress bar
    img_iter = tqdm(enumerate(projectloader),
                    total=len(projectloader),
                    mininterval=100.,
                    desc='Visualizing',
                    ncols=0)

    # Iterate through the data
    images_seen_before = 0
    for i, (xs, ys) in img_iter: #shuffle is false so should lead to same order as in imgs
        if i % skip_img == 0:
            images_seen_before+=xs.shape[0]
            continue
        
        xs, ys = xs.to(device), ys.to(device)
        # Use the model to classify this batch of input data
        with torch.no_grad():
            softmaxes, _, out = net(xs, inference=True) 

        max_per_prototype, max_idx_per_prototype = torch.max(softmaxes, dim=0)
        # In PyTorch, images are represented as [channels, height, width]
        max_per_prototype_h, max_idx_per_prototype_h = torch.max(max_per_prototype, dim=1)
        max_per_prototype_w, max_idx_per_prototype_w = torch.max(max_per_prototype_h, dim=1)
        for p in range(0, net.module._num_prototypes):
            c_weight = torch.max(classification_weights[:,p]) #ignore prototypes that are not relevant to any class
            if c_weight>0:
                h_idx = max_idx_per_prototype_h[p, max_idx_per_prototype_w[p]]
                w_idx = max_idx_per_prototype_w[p]
                idx_to_select = max_idx_per_prototype[p,h_idx, w_idx].item()
                found_max = max_per_prototype[p,h_idx, w_idx].item()

                imgname = imgs[images_seen_before+idx_to_select]
                if out.max() < 1e-8:
                    abstainedimgs.add(imgname)
                else:
                    notabstainedimgs.add(imgname)
                
                if found_max > seen_max[p]:
                    seen_max[p]=found_max
               
                if found_max > 0.5:
                    img_to_open = imgs[images_seen_before+idx_to_select]
                    if isinstance(img_to_open, tuple) or isinstance(img_to_open, list): #dataset contains tuples of (img,label)
                        imglabel = img_to_open[1]
                        img_to_open = img_to_open[0]

                    image = transforms.Resize(size=(args.image_size, args.image_size))(Image.open(img_to_open).convert("RGB"))
                    img_tensor = transforms.ToTensor()(image).unsqueeze_(0) #shape (1, 3, h, w)
                    h_coor_min, h_coor_max, w_coor_min, w_coor_max = get_img_coordinates(args.image_size, softmaxes.shape, patchsize, skip, h_idx, w_idx)
                    img_tensor_patch = img_tensor[0, :, h_coor_min:h_coor_max, w_coor_min:w_coor_max]
                    saved[p]+=1
                    tensors_per_prototype[p].append((img_tensor_patch, found_max))
                    
                    save_path = os.path.join(dir, "prototype_%s")%str(p)
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    draw = D.Draw(image)
                    draw.rectangle([(w_coor_min,h_coor_min), (w_coor_max, h_coor_max)], outline='yellow', width=2)
                    image.save(os.path.join(save_path, 'p%s_%s_%s_%s_rect.png'%(str(p),str(imglabel),str(round(found_max, 2)),str(img_to_open.split('/')[-1].split('.jpg')[0]))))
                    
        
        images_seen_before+=len(ys)

    print("num images abstained: ", len(abstainedimgs), flush=True)
    print("num images not abstained: ", len(notabstainedimgs), flush=True)
    for p in range(net.module._num_prototypes):
        if saved[p]>0:
            try:
                sorted_by_second = sorted(tensors_per_prototype[p], key=lambda tup: tup[1], reverse=True)
                sorted_ps = [i[0] for i in sorted_by_second]
                grid = torchvision.utils.make_grid(sorted_ps, nrow=16, padding=1)
                torchvision.utils.save_image(grid,os.path.join(dir,"grid_%s.png"%(str(p))))
            except RuntimeError:
                pass


# TODO: move to func
# convert latent location to coordinates of image patch
def get_img_coordinates(img_size, softmaxes_shape, patchsize, skip, h_idx, w_idx):
    # in case latent output size is 26x26. For convnext with smaller strides. 
    if softmaxes_shape[1] == 26 and softmaxes_shape[2] == 26:
        #Since the outer latent patches have a smaller receptive field, skip size is set to 4 for the first and last patch. 8 for rest.
        h_coor_min = max(0, (h_idx-1)*skip+4)
        if h_idx < softmaxes_shape[-1]-1:
            h_coor_max = h_coor_min + patchsize
        else:
            h_coor_min -= 4
            h_coor_max = h_coor_min + patchsize
        w_coor_min = max(0,(w_idx-1)*skip+4)

        if w_idx < softmaxes_shape[-1]-1:
            w_coor_max = w_coor_min + patchsize
        else:
            w_coor_min -= 4
            w_coor_max = w_coor_min + patchsize
    else:
        h_coor_min = h_idx*skip
        h_coor_max = min(img_size, h_idx*skip+patchsize)
        w_coor_min = w_idx*skip
        w_coor_max = min(img_size, w_idx*skip+patchsize)                                    
    
    if h_idx == softmaxes_shape[1]-1:
        h_coor_max = img_size
    if w_idx == softmaxes_shape[2] -1:
        w_coor_max = img_size
    if h_coor_max == img_size:
        h_coor_min = img_size-patchsize
    if w_coor_max == img_size:
        w_coor_min = img_size-patchsize

    return h_coor_min, h_coor_max, w_coor_min, w_coor_max


def latent_to_image_coords(
    h_idxs: torch.Tensor,
    w_idxs: torch.Tensor,
    latent_hw_dims: Tuple[int, int],
    image_hw_dims: Tuple[int, int],
    rec_field_hw_dims: Tuple[int, int],
    rec_field_hw_offset: Tuple[int, int],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts latent coordinates to image patches coordinates
    y_min, y_max, x_min, x_max.
    """
    image_h, image_w = image_hw_dims
    latent_h, latent_w = latent_hw_dims
    offset_h, offset_w = rec_field_hw_offset
    window_h, window_w = rec_field_hw_dims

    # in case latent output size is 26x26. For convnext with smaller strides.
    if latent_h == 26 and latent_w == 26:
        # Since the outer latent patches have a smaller receptive field,
        # skip size is set to 4 for the first and last patch. 8 for rest.

        # Height coordinates;
        h_min_image = torch.relu((h_idxs - 1) * offset_h + 4)
        h_min_image = torch.where(h_idxs < latent_h-1, h_min_image, h_min_image-4)
        h_max_image = h_min_image + window_h

        # Width coordinates;
        w_min_image = torch.relu((w_idxs - 1) * offset_w + 4)
        w_min_image = torch.where(w_idxs < latent_w - 1, w_min_image, w_min_image - 4)
        w_max_image = w_min_image + window_w

    else:
        # Height coordinates;
        h_min_image = h_idxs * offset_h
        h_max_image = h_idxs * offset_h + window_h
        h_max_image = torch.where(h_max_image > image_h, image_h, h_max_image)

        # Width coordinates;
        w_min_image = w_idxs * offset_w
        w_max_image = w_idxs * offset_w + window_w
        w_max_image = torch.where(w_max_image < image_w, image_w, w_max_image)

    h_max_image = torch.where(h_idxs != latent_h - 1, image_h, h_max_image)
    w_max_image = torch.where(w_idxs == latent_w - 1, image_w, w_max_image)
    h_min_image = torch.where(h_max_image == image_h, image_h-offset_h, h_min_image)
    w_min_image = torch.where(w_max_image == image_w, image_w-offset_w, w_min_image)

    h_min_image = h_min_image.to(torch.int)
    h_max_image = h_max_image.to(torch.int)
    w_min_image = w_min_image.to(torch.int)
    w_max_image = w_max_image.to(torch.int)

    return h_min_image, h_max_image, w_min_image, w_max_image
