from typing import Mapping, Any, Dict
import skimage.exposure
import numpy as np
import yaml
import math
import cv2
from skimage import measure
import torch
import torch.nn.functional as F
from typing import List
import os


@torch.jit.script
def factor_pad(tensor, factor: int = 16):
    r"""Helper function to pad a tensor such that all dimensions are divisble
    by a particular factor
    """
    h, w = tensor.size()[2:]
    pad_bottom = factor - h % factor if h % factor != 0 else 0
    pad_right = factor - w % factor if w % factor != 0 else 0
    if pad_bottom == 0 and pad_right == 0:
        return tensor
    else:
        return F.pad(tensor, (0, pad_right, 0, pad_bottom))


@torch.jit.script
def find_instance_center(ctr_hmp, threshold: float = 0.1, nms_kernel: int = 7):
    r"""Find the center points from the center heatmap.

    Args:
        ctr_hmp: A Tensor of shape (N, 1, H, W) of raw center heatmap output, where N is the batch size,
        for consistent, we only support N=1.

        threshold: A Float, threshold applied to center heatmap score. Default 0.1.

        nms_kernel: An Integer, NMS max pooling kernel size. Default 7.

    Returns:
        ctr_all: A Tensor of shape (K, 2) where K is the number of center points. The order of second dim is (y, x).

    """
    # thresholding, setting values below threshold to -1
    ctr_hmp = F.threshold(ctr_hmp, threshold, -1.)

    # NMS
    nms_padding = nms_kernel // 2
    ctr_hmp_max_pooled = F.max_pool2d(
        ctr_hmp, kernel_size=nms_kernel, stride=1, padding=nms_padding
    )

    if nms_kernel % 2 == 0:
        # clip last row and column to maintain size
        ctr_hmp_max_pooled = ctr_hmp_max_pooled[..., :-1, :-1]

    ctr_hmp[ctr_hmp != ctr_hmp_max_pooled] = -1.

    # squeeze first two dimensions
    ctr_hmp = ctr_hmp.squeeze()
    assert len(ctr_hmp.size()) == 2, \
        'Something is wrong with center heatmap dimension.'

    # find non-zero elements
    ctr_all = torch.nonzero(ctr_hmp > 0)
    return ctr_all


@torch.jit.script
def chunked_pixel_grouping(ctr, ctr_loc, chunksize: int = 20):
    r"""Gives each pixel in the image an instance id without exceeding memory.

    Args:
        ctr: A Tensor of shape [K, 1, 2] where K is the number of center points.
        The order of third dim is (y, x).

        ctr_loc: A Tensor of shape [1, H*W, 2] of center locations for each pixel
        after applying offsets.

        chunksize: Int. Number of instances to process in a chunk. Default 20.

    Returns:
        instance_ids: A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).

    """
    # initialize a tensor to store nearest instance center ids
    # and a distances placeholder of large floats
    instance_ids = torch.zeros(ctr_loc.size(1), dtype=torch.long, device=ctr.device)  # (H*W,)
    nearest_distances = 1e5 * torch.ones(ctr_loc.size(1), dtype=torch.float, device=ctr.device)  # (H*W,)

    # split the centers into chunks
    ctr_chunks = torch.split(ctr, chunksize, dim=0)
    prev = 1  # starting label for instance ids

    for ctr_chunk in ctr_chunks:
        # chunk of size (chunksize, 1, 2)
        distances = torch.norm(ctr_chunk - ctr_loc, dim=-1)  # (chunksize, H*W)
        min_distances, min_dist_indices = distances.min(dim=0)  # (H*W,)

        # add the instance ids relative to the previous label
        instance_ids[min_distances < nearest_distances] = prev + min_dist_indices[min_distances < nearest_distances]
        nearest_distances = torch.min(nearest_distances, min_distances)

        # increment the instance ids
        prev += ctr_chunk.size(0)

    return instance_ids


@torch.jit.script
def group_pixels(ctr, offsets, chunksize: int = 20, step: float = 1):
    r"""
    Gives each pixel in the image an instance id.

    Args:
        ctr: A Tensor of shape [K, 2] where K is the number of center points. The order of second dim is (y, x).

        offsets: A Tensor of shape [N, 2, H, W] of raw offset output, where N is the batch size of 1.
        The order of second and third dim is (offset_y, offset_x).

        chunksize: Int. Number of instances to process in a chunk. Default 20.

    Returns:
        instance_ids: A Tensor of shape [1, H, W] (to be gathered by distributed data parallel).

    """
    assert ctr.size(0) > 0
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    offsets = offsets.squeeze(0)
    height, width = offsets.size()[1:]

    hend = int(height * step)
    wend = int(width * step)

    # generates a coordinate map, where each location is the coordinate of that loc
    y_coord = torch.arange(0, hend, step=step, dtype=offsets.dtype, device=offsets.device).repeat(1, width,
                                                                                                  1).transpose(1, 2)
    x_coord = torch.arange(0, wend, step=step, dtype=offsets.dtype, device=offsets.device).repeat(1, height, 1)
    coord = torch.cat((y_coord, x_coord), dim=0)

    # multiply the ctrs by step
    ctr = step * ctr

    ctr_loc = coord + offsets
    ctr_loc = ctr_loc.reshape((2, height * width)).transpose(1, 0)

    # ctr: [K, 2] -> [K, 1, 2]
    # ctr_loc = [H*W, 2] -> [1, H*W, 2]
    ctr = ctr.unsqueeze(1)
    ctr_loc = ctr_loc.unsqueeze(0)

    if ctr.size(0) <= chunksize:
        distance = torch.norm(ctr - ctr_loc, dim=-1)  # (K, H*W)
        instance_id = 1 + torch.argmin(distance, dim=0)  # (H*W)
    else:
        instance_id = chunked_pixel_grouping(ctr, ctr_loc, chunksize)

    instance_id = instance_id.reshape((1, height, width))

    return instance_id


@torch.jit.script
def get_instance_segmentation(
        sem_seg,
        ctr_hmp,
        offsets,
        thing_list: List[int],
        threshold: float = 0.1,
        nms_kernel: int = 7
):
    r"""Post-processing for instance segmentation, gets class agnostic instance id map.

    Args:
        sem_seg: A Tensor of shape (N, 1, H, W), predicted semantic label.

        ctr_hmp: A Tensor of shape (N, 1, H, W) of raw center heatmap output,
        where N is the batch size of 1.

        offsets: A Tensor of shape (N, 2, H, W) of raw offset output.
        The order of second dim is (offset_y, offset_x).

        thing_list: A List of instance class ids.

        threshold: A Float, threshold applied to center heatmap score. Default 0.1.

        nms_kernel: An Integer, NMS max pooling kernel size. Default 7.

    Returns:
        thing_seg: A Tensor of shape (1, H, W).

        ctr: A Tensor of shape (1, K, 2) where K is the number of center points.
        The order of second dim is (y, x).

    """
    assert sem_seg.size(0) == 1, \
        f'Only batch size of 1 is supported!'

    sem_seg = sem_seg[0]

    # keep only label for instance classes
    instance_seg = torch.zeros_like(sem_seg)
    for thing_class in thing_list:
        instance_seg[sem_seg == thing_class] = 1

    ctr = find_instance_center(ctr_hmp, threshold=threshold, nms_kernel=nms_kernel)

    # no objects, return zeros
    if ctr.size(0) == 0:
        return torch.zeros_like(sem_seg), ctr.unsqueeze(0)

    instance_id = group_pixels(ctr, offsets)
    return instance_seg * instance_id, ctr.unsqueeze(0)


@torch.jit.script
def merge_semantic_and_instance(
        sem_seg,
        ins_seg,
        label_divisor: int,
        thing_list: List[int],
        stuff_area: int,
        void_label: int
):
    r"""Post-processing for panoptic segmentation, by merging semantic
    segmentation label and class agnostic instance segmentation label.

    Args:
        sem_seg: A Tensor of shape (1, H, W), predicted semantic label.

        ins_seg: A Tensor of shape (1, H, W), predicted instance label.

        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.

        thing_list: A List of thing class ids.

        stuff_area: An Integer, remove stuff whose area is less than stuff_area.

        void_label: An Integer, indicates the region has no confident prediction.

    Returns:
        merged_seg: A Tensor of shape (1, H, W).

    """
    # In case thing mask does not align with semantic prediction
    pan_seg = torch.zeros_like(sem_seg) + void_label
    thing_seg = ins_seg > 0
    semantic_thing_seg = torch.zeros_like(sem_seg)
    for thing_class in thing_list:
        semantic_thing_seg[sem_seg == thing_class] = 1

    # keep track of instance id for each class
    class_id_tracker: Dict[int, int] = {}

    # paste thing by majority voting
    instance_ids = torch.unique(ins_seg)
    for ins_id in instance_ids:
        if ins_id == 0:
            continue

        # Make sure only do majority voting within semantic_thing_seg
        thing_mask = (ins_seg == ins_id) & (semantic_thing_seg == 1)
        if torch.count_nonzero(thing_mask) == 0:
            continue

        class_id, _ = torch.mode(sem_seg[thing_mask].view(-1, ))
        if class_id.item() in class_id_tracker:
            new_ins_id = class_id_tracker[class_id.item()]
        else:
            class_id_tracker[class_id.item()] = 1
            new_ins_id = 1

        class_id_tracker[class_id.item()] += 1
        pan_seg[thing_mask] = class_id * label_divisor + new_ins_id

    # paste stuff to unoccupied area
    class_ids = torch.unique(sem_seg)
    for class_id in class_ids:
        if class_id.item() in thing_list:
            continue

        # calculate stuff area
        stuff_mask = (sem_seg == class_id) & (~thing_seg)
        area = torch.nonzero(stuff_mask).size(0)

        if area >= stuff_area:
            pan_seg[stuff_mask] = class_id * label_divisor

    return pan_seg


@torch.jit.script
def get_panoptic_segmentation(
        sem,
        ctr_hmp,
        offsets,
        thing_list: List[int],
        label_divisor: int,
        stuff_area: int,
        void_label: int,
        threshold: float = 0.1,
        nms_kernel: int = 7
):
    r"""Post-processing for panoptic segmentation.

    Args:
        sem_seg: A Tensor of shape (N, 1, H, W), predicted semantic labels.

        ctr_hmp: A Tensor of shape (N, 1, H, W) of raw center heatmap output,
        where N is the batch size of 1.

        offsets: A Tensor of shape (N, 2, H, W) of raw offset output.
        The order of second dim is (offset_y, offset_x).

        thing_list: A List of thing class ids.

        label_divisor: An Integer, used to convert panoptic id = semantic id * label_divisor + instance_id.

        stuff_area: An Integer, remove stuff whose area is less than stuff_area.

        void_label: An Integer, indicates the region has no confident prediction.

        threshold: A Float, threshold applied to center heatmap score. Default 0.1.

        nms_kernel: An Integer, NMS max pooling kernel size. Default 7.

    Returns:
        pan_seg: A Tensor of shape (1, H, W) of type torch.long.

    """

    if sem.size(1) != 1:
        raise ValueError('Expect single channel semantic segmentation. Softmax/argmax first!')
    if sem.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if ctr_hmp.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')
    if offsets.size(0) != 1:
        raise ValueError('Only supports inference for batch size = 1')

    # instance segmentation with thing centers
    instance, center = get_instance_segmentation(
        sem, ctr_hmp, offsets, thing_list, threshold=threshold, nms_kernel=nms_kernel
    )

    panoptic = merge_semantic_and_instance(
        sem, instance, label_divisor, thing_list, stuff_area, void_label
    )

    return panoptic, center


def resize_by_factor(image, scale_factor=1):
    # do nothing
    if scale_factor == 1:
        return image

    # cv2 expects (w, h) for image size
    h, w = image.shape
    dh = math.ceil(h / scale_factor)
    dw = math.ceil(w / scale_factor)

    image = cv2.resize(image, (dw, dh), cv2.INTER_LINEAR)

    return image


def connected_components(seg):
    seg = measure.label(seg)
    return seg


@torch.no_grad()
def logits_to_prob(logits):
    # multiclass or binary
    if logits.size(1) > 1:
        logits = F.softmax(logits, dim=1)
    else:
        logits = torch.sigmoid(logits)

    return logits



def read_yaml(url):
    r"""
    Loads a yaml config file from the given path/url.
    """
    with open(url, mode='r') as handle:
        config = yaml.load(handle, Loader=yaml.FullLoader)

    return config


def normalize(img, mean, std, max_pixel_value=255.0):
    mean = np.array(mean, dtype=np.float32)
    mean *= max_pixel_value

    std = np.array(std, dtype=np.float32)
    std *= max_pixel_value

    denominator = np.reciprocal(std, dtype=np.float32)

    img = img.astype(np.float32)
    img -= mean
    img *= denominator
    return img


def to_tensor(img):
    # move channel dim from last to first
    tensor = torch.from_numpy(img[None])
    return tensor


def load_model_to_device(fpath_or_url, device):
    # check whether local file or url
    model = torch.jit.load(fpath_or_url, map_location=device)
    return model


class Preprocessor:
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image=None):
        assert image is not None
        if np.issubdtype(image.dtype, np.floating):
            raise Exception('Input image cannot be float type!')

        max_value = np.iinfo(image.dtype).max
        image = normalize(image, self.mean, self.std, max_pixel_value=max_value)
        return {'image': to_tensor(image)}


class PanopticDeepLabEngine:
    def __init__(
            self,
            model,
            thing_list,
            label_divisor=1000,
            stuff_area=64,
            void_label=0,
            nms_threshold=0.1,
            nms_kernel=7,
            confidence_thr=0.5,
            padding_factor=16,
            coarse_boundaries=True,
    ):
        self.model = model.eval()
        self.thing_list = thing_list
        self.label_divisor = label_divisor
        self.stuff_area = stuff_area
        self.void_label = void_label
        self.nms_threshold = nms_threshold
        self.nms_kernel = nms_kernel
        self.confidence_thr = confidence_thr
        self.padding_factor = padding_factor
        self.coarse_boundaries = coarse_boundaries

    def to_model_device(self, tensor):
        # move tensor to the model device
        device = next(self.model.parameters()).device
        return tensor.to(device, non_blocking=True)

    @torch.no_grad()
    def _harden_seg(self, sem):
        if sem.size(1) > 1:  # multiclass segmentation
            sem = torch.argmax(sem, dim=1, keepdim=True)
        else:
            sem = (sem >= self.confidence_thr).long()

        return sem

    @torch.no_grad()
    def postprocess(self, sem, ctr_hmp, offsets):
        pan_seg, _ = get_panoptic_segmentation(
            sem, ctr_hmp, offsets, self.thing_list,
            self.label_divisor, self.stuff_area,
            self.void_label, self.nms_threshold, self.nms_kernel
        )
        return pan_seg

    @torch.no_grad()
    def infer(self, image, render_steps=2):
        model_out = self.model(image, render_steps, interpolate_ins=not self.coarse_boundaries)

        # notice that sem is NOT sem_logits
        model_out['sem'] = logits_to_prob(model_out['sem_logits'])

        return model_out

    @torch.no_grad()
    def get_instance_cells(self, ctr_hmp, offsets, upsampling=1):
        # first find the object centers
        ctr = find_instance_center(ctr_hmp, self.nms_threshold, self.nms_kernel)

        # grid step size for pixel grouping
        step = 4 if self.coarse_boundaries else 1

        # no objects, return zeros
        if ctr.size(0) == 0:
            instance_cells = torch.zeros_like(ctr_hmp)
        else:
            # grouped pixels should be integers,
            # but we need them in float type for upsampling
            instance_cells = group_pixels(ctr, offsets, step=step).float()[None]  # (1, 1, H, W)

        # scale again by the upsampling factor times step
        instance_cells = F.interpolate(instance_cells, scale_factor=int(upsampling * step), mode='nearest')
        return instance_cells

    @torch.no_grad()
    def get_panoptic_seg(self, sem, instance_cells):
        # keep only label for instance classes
        instance_seg = torch.zeros_like(sem)
        for thing_class in self.thing_list:
            instance_seg[sem == thing_class] = 1

        # map object ids
        instance_seg = (instance_seg * instance_cells[0]).long()

        pan_seg = merge_semantic_and_instance(
            sem, instance_seg, self.label_divisor, self.thing_list,
            self.stuff_area, self.void_label
        )

        return pan_seg

    @torch.no_grad()
    def postprocess(self, sem, instance_cells):
        # harden the segmentation
        sem = self._harden_seg(sem)[0]
        return self.get_panoptic_seg(sem, instance_cells)

    def __call__(self, image, size, upsampling=1):
        assert math.log(upsampling, 2).is_integer(), \
            "Upsampling factor not log base 2!"

        # check that image is 4d (N, C, H, W) and has a
        # batch dim of 1, larger batch size raises exception
        assert image.ndim == 4 and image.size(0) == 1

        # move image to same device as the model
        h, w = size
        image = factor_pad(image, self.padding_factor)
        image = self.to_model_device(image)

        # infer labels
        model_out = self.infer(image, int(2 + math.log(upsampling, 2)))

        # calculate the instance cells
        instance_cells = self.get_instance_cells(
            model_out['ctr_hmp'], model_out['offsets'], upsampling
        )
        pan_seg = self.postprocess(model_out['sem'], instance_cells)

        # remove padding from the pan_seg
        pan_seg = pan_seg[..., :h, :w]

        return pan_seg


class Empanada2DInference(torch.nn.Module):
    def __init__(self, model, inference_scale=1, label_divisor=100000, nms_threshold=0.1, nms_kernel=3,
                 confidence_thr=0.3, fine_boundaries=True, tile_size=0):
        super().__init__()
        model = os.path.join(os.path.dirname(os.path.abspath(__file__)), model)
        self.panoptic_model = torch.jit.load(model, map_location=torch.device('cpu'))

        thing_list = [1]
        labels = [1]
        class_names = {1: 'mito'}
        self.label_divisor = label_divisor
        padding_factor = 16
        self.inference_scale = inference_scale
        fine_boundaries = fine_boundaries
        self.tile_size = tile_size

        self.engine = PanopticDeepLabEngine(
            self.panoptic_model, thing_list=thing_list,
            label_divisor=label_divisor,
            nms_threshold=nms_threshold,
            nms_kernel=nms_kernel,
            confidence_thr=confidence_thr,
            padding_factor=padding_factor,
            coarse_boundaries=not fine_boundaries,
        )

        self.preprocessor = Preprocessor(mean=0.57571, std=0.12765)

    def force_connected(self, pan_seg):
        for label in self.engine.thing_list:
            # convert from pan_seg to instance_seg
            min_id = label * self.label_divisor
            max_id = min_id + self.label_divisor

            # zero all objects/semantic segs outside of instance_id range
            instance_seg = pan_seg.copy()
            outside_mask = np.logical_or(pan_seg < min_id, pan_seg >= max_id)
            instance_seg[outside_mask] = 0

            # relabel connected components
            instance_seg = connected_components(instance_seg).astype(np.int32)
            instance_seg[instance_seg > 0] += min_id
            pan_seg[instance_seg > 0] = instance_seg[instance_seg > 0]

        return pan_seg

    def infer(self, image):
        image = skimage.exposure.rescale_intensity(image.cpu().numpy(), out_range=(0, 255)).astype(np.uint8)
        # engine handles upsampling and padding
        size = image.shape
        # resize image to correct scale
        image = resize_by_factor(image, self.inference_scale)
        image = self.preprocessor(image)['image'].unsqueeze(0)
        pan_seg = self.engine(image, size, upsampling=self.inference_scale)
        pan_seg = self.force_connected(pan_seg.squeeze().cpu().numpy().astype(np.int32))
        return pan_seg.astype(np.int16)

    def forward(self, image):
        return self.infer(image)

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True, assign: bool = False):
        self.panoptic_model.load_state_dict(state_dict, strict=strict, assign=assign)


