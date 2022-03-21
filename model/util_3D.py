import torch
import torch.nn.functional as F

def divide_safe(num, den):
    eps = 1e-8
    den[torch.where(den==0)]=den[torch.where(den==0)]+eps
    return num/den

def inv_depths(start_depth, end_depth, num_depths):
    """Sample reversed, sorted inverse depths between a near and far plane.

    Args:
        start_depth: The first depth (i.e. near plane distance).
        end_depth: The last depth (i.e. far plane distance).
        num_depths: The total number of depths to create. start_depth and
            end_depth are always included and other depths are sampled
            between them uniformly according to inverse depth.
    Returns:
        The depths sorted in descending order (so furthest first). This order is
        useful for back to front compositing.
    """
    inv_start_depth = 1.0 / start_depth
    inv_end_depth = 1.0 / end_depth
    depths = [start_depth, end_depth]
    for i in range(1, num_depths - 1):
        fraction = float(i) / float(num_depths - 1)
        inv_depth = inv_start_depth + (inv_end_depth - inv_start_depth) * fraction
        depths.append(1.0 / inv_depth)
    depths = sorted(depths)
    return depths[::-1]

def meshgrid_abs(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid in the absolute coordinates.

    Args:
        batch: batch size
        height: height of the grid
        width: width of the grid
        is_homogeneous: whether to return in homogeneous coordinates
    Returns:
        x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    xs = torch.linspace(0.0, width-1, width)
    ys = torch.linspace(0.0, height-1, height)
    ys, xs = torch.meshgrid(ys, xs)

    if is_homogeneous:
        ones = torch.ones_like(xs)
        coords = torch.stack([xs, ys, ones], dim=0)
    else:
        coords = torch.stack([xs, ys], dim=0)
    coords = coords.unsqueeze(0).repeat([batch, 1, 1, 1])
    return coords

def inv_homography(k_s, k_t, rot, t, n_hat, a):
    """Computes inverse homography matrix between two cameras via a plane.

    Args:
        k_s: intrinsics for source cameras, [..., 3, 3] matrices
        k_t: intrinsics for target cameras, [..., 3, 3] matrices
        rot: relative rotations between source and target, [..., 3, 3] matrices
        t: [..., 3, 1], translations from source to target camera. Mapping a 3D
        point p from source to target is accomplished via rot * p + t.
        n_hat: [..., 1, 3], plane normal w.r.t source camera frame
        a: [..., 1, 1], plane equation displacement
    Returns:
        homography: [..., 3, 3] inverse homography matrices (homographies mapping
        pixel coordinates from target to source).
    """
    rot_t=torch.transpose(rot,-2,-1)
    k_t_inv =torch.inverse(k_t)
    denom = a - torch.matmul(torch.matmul(n_hat, rot_t), t)
    numerator = torch.matmul(torch.matmul(torch.matmul(rot_t, t), n_hat), rot_t)
    inv_hom = torch.matmul(
        torch.matmul(k_s, rot_t + divide_safe(numerator, denom)),k_t_inv)
    return inv_hom

def normalize_homogeneous(points):
    """Converts homogeneous coordinates to regular coordinates.

    Args:
        points: [..., n_dims_coords+1]; points in homogeneous coordinates.
    Returns:
        points_uv_norm: [..., n_dims_coords];
            points in standard coordinates after dividing by the last entry.
    """
    uv = points[..., :-1]
    w = points[..., -1].unsqueeze(-1)
    return divide_safe(uv, w)

def transform_points(points,homography):
    """Transforms input points according to homography.

    Args:
        points: [..., H, W, 3]; pixel (u,v,1) coordinates.
        homography: [..., 3, 3]; desired matrix transformation
    Returns:
        output_points: [..., H, W, 3]; transformed (u,v,w) coordinates.
    """

    # Because the points have two additional dimensions as they vary across the
    # width and height of an image, we need to reshape to multiply by the
    # per-image homographies.
    points_orig_shape = list(points.size())
    points_reshaped_shape = list(homography.size())
    points_reshaped_shape[-2] = -1

    points_reshaped = points.view(points_reshaped_shape)
    transformed_points = torch.matmul(points_reshaped, torch.transpose(homography,-2,-1))
    transformed_points = transformed_points.view(points_orig_shape)
    return transformed_points

def grid_sample(imgs, coords):
    # todo
    _,coords_h,coords_w,_=coords.size()

    coords[:,:,:,0]=coords[:,:,:,0]/(coords_w-1)*2-1
    coords[:,:,:,1]=coords[:,:,:,1]/(coords_h-1)*2-1

    imgs=imgs.permute(0,3,1,2)
    imgs_sampled=F.grid_sample(imgs, coords)
    imgs_sampled=imgs_sampled.permute(0,2,3,1)

    return imgs_sampled

def bilinear_wrapper(imgs, coords):
    """Wrapper around bilinear sampling function, handles arbitrary input sizes.

    Args:
    imgs: [..., H_s, W_s, C] images to resample
    coords: [..., H_t, W_t, 2], source pixel locations from which to copy
    Returns:
    [..., H_t, W_t, C] images after bilinear sampling from input.
    """
    # The bilinear sampling code only handles 4D input, so we'll need to reshape.
    init_dims = list(imgs.size())[:-3:]
    end_dims_img = list(imgs.size())[-3::]
    end_dims_coords = list(coords.size())[-3::]
    prod_init_dims = init_dims[0]
    for ix in range(1, len(init_dims)):
        prod_init_dims *= init_dims[ix]
    imgs=imgs.contiguous()
    imgs = imgs.view([prod_init_dims] + end_dims_img)
    coords = coords.view([prod_init_dims] + end_dims_coords)
    imgs_sampled = grid_sample(imgs, coords)
    imgs_sampled =imgs_sampled.view(init_dims + list(imgs_sampled.size())[-3::])
    return imgs_sampled


def transform_plane_imgs(imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a):
    """Transforms input imgs via homographies for corresponding planes.

    Args:
    imgs: are [..., H_s, W_s, C]
    pixel_coords_trg: [..., H_t, W_t, 3]; pixel (u,v,1) coordinates.
    k_s: intrinsics for source cameras, [..., 3, 3] matrices
    k_t: intrinsics for target cameras, [..., 3, 3] matrices
    rot: relative rotation, [..., 3, 3] matrices
    t: [..., 3, 1], translations from source to target camera
    n_hat: [..., 1, 3], plane normal w.r.t source camera frame
    a: [..., 1, 1], plane equation displacement
    Returns:
    [..., H_t, W_t, C] images after bilinear sampling from input.
        Coordinates outside the image are sampled as 0.
    """

    hom_t2s_planes = inv_homography(k_s, k_t, rot, t, n_hat, a)
    pixel_coords_t2s = transform_points(pixel_coords_trg, hom_t2s_planes)
    pixel_coords_t2s = normalize_homogeneous(pixel_coords_t2s)
    imgs_s2t = bilinear_wrapper(imgs, pixel_coords_t2s)

    return imgs_s2t

def planar_transform(imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a):
    """Transforms imgs, masks and computes dmaps according to planar transform.

    Args:
    imgs: are [L, B, H, W, C], typically RGB images per layer
    pixel_coords_trg: tensors with shape [B, H_t, W_t, 3];
        pixel (u,v,1) coordinates of target image pixels. (typically meshgrid)
    k_s: intrinsics for source cameras, [B, 3, 3] matrices
    k_t: intrinsics for target cameras, [B, 3, 3] matrices
    rot: relative rotation, [B, 3, 3] matrices
    t: [B, 3, 1] matrices, translations from source to target camera
        (R*p_src + t = p_tgt)
    n_hat: [L, B, 1, 3] matrices, plane normal w.r.t source camera frame
        (typically [0 0 1])
    a: [L, B, 1, 1] matrices, plane equation displacement
        (n_hat * p_src + a = 0)
    Returns:
    imgs_transformed: [L, ..., C] images in trg frame
    Assumes the first dimension corresponds to layers.
    """
    n_layers = list(imgs.size())[0]
    rot_rep_dims = [n_layers]
    rot_rep_dims += [1 for _ in range(len(k_s.size()))]

    cds_rep_dims = [n_layers]
    cds_rep_dims += [1 for _ in range(len(pixel_coords_trg.size()))]

    k_s = k_s.unsqueeze(0).repeat(rot_rep_dims)
    k_t = k_t.unsqueeze(0).repeat(rot_rep_dims)
    t = t.unsqueeze(0).repeat(rot_rep_dims)
    rot = rot.unsqueeze(0).repeat(rot_rep_dims)
    pixel_coords_trg = pixel_coords_trg.unsqueeze(0).repeat(cds_rep_dims)

    imgs_trg = transform_plane_imgs(imgs, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a)
    return imgs_trg

def projective_forward_homography(src_images, intrinsics, pose, depths):
    """Use homography for forward warping.

    Args:
    src_images: [layers, batch, height, width, channels]
    intrinsics: [batch, 3, 3]
    pose: [batch, 4, 4]
    depths: [layers, batch]
    Returns:
    proj_src_images: [layers, batch, height, width, channels]
    """
    n_layers, n_batch, height, width, _ = list(src_images.size())
    # Format for planar_transform code:
    # rot: relative rotation, [..., 3, 3] matrices
    # t: [B, 3, 1], translations from source to target camera (R*p_s + t = p_t)
    # n_hat: [L, B, 1, 3], plane normal w.r.t source camera frame [0,0,1]
    #        in our case
    # a: [L, B, 1, 1], plane equation displacement (n_hat * p_src + a = 0)
    rot = pose[:, :3, :3]
    t = pose[:, :3, 3:]
    n_hat = torch.tensor([0,0,1]).view(1,1,1,3).to(src_images.device)
    n_hat = n_hat.repeat([n_layers, n_batch, 1, 1]).float()
    a = -depths.view([n_layers, n_batch, 1, 1])
    k_s = intrinsics
    k_t = intrinsics
    pixel_coords_trg = meshgrid_abs(n_batch, height, width).permute([0, 2, 3, 1]).to(src_images.device)
    proj_src_images = planar_transform(src_images, pixel_coords_trg, k_s, k_t, rot, t, n_hat, a)
    return proj_src_images