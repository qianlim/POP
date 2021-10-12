from genericpath import isdir
import os
from os.path import join, basename, dirname, realpath
import glob

import numpy as np
import open3d as o3d
from tqdm import tqdm


def render_pcl_front_view(vis, cam_params=None, fn=None, img_save_fn=None, pt_size=3):
    mesh = o3d.io.read_point_cloud(fn)  
    vis.add_geometry(mesh)
    opt = vis.get_render_option()
    
    opt.point_size = pt_size

    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(cam_params)

    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(img_save_fn, True)

    vis.clear_geometries()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', default='SCALE_demo_00000_simuskirt', help='Name of the model (experiment), the same as the --name flag in the main experiment')
    parser.add_argument('-r', '--img_res', type=int ,default=1024, help='Resolution of rendered image')
    parser.add_argument('--query_resolution', type=int, default=256, help='resolution of the query UV positional map that was used to generate the\
                                                                           point cloud results')
    parser.add_argument('--case', type=str, choices=['seen', 'unseen'], default='seen', help='wheter to render the test_seen (outfit) or test_unseen scenario')                                                                  
    args = parser.parse_args()

    img_res = args.img_res

    # path for saving the rendered images
    SCRIPT_DIR = dirname(realpath(__file__))
    target_root = join(SCRIPT_DIR, '..', 'results', 'rendered_imgs')
    os.makedirs(target_root, exist_ok=True)
    
    # set up camera
    focal_length = 900 * (args.img_res / 1024.) # 900 is a hand-set focal length when the img resolution=1024. 
    x0, y0 = (img_res-1)/2, (img_res-1)/2
    INTRINSIC = np.array([
        [focal_length, 0.,           x0], 
        [0.,           focal_length, y0],
        [0.,           0.,            1]
    ])

    EXTRINSIC = np.load(join(SCRIPT_DIR, 'cam_front_extrinsic.npy'))


    cam_intrinsics = o3d.camera.PinholeCameraIntrinsic()
    cam_intrinsics.intrinsic_matrix = INTRINSIC
    cam_intrinsics.width = img_res
    cam_intrinsics.height = img_res

    cam_params_front = o3d.camera.PinholeCameraParameters()
    cam_params_front.intrinsic = cam_intrinsics
    cam_params_front.extrinsic = EXTRINSIC

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_res, height=img_res)

    # render
    color_mode, pt_size, ext = ['normal_colored', 3.0, '_pred.ply']
    
    render_savedir_root = join(target_root, args.name, 'test_{}'.format(args.case),\
                     'query_resolution{}'.format(args.query_resolution))

    pcl_result_root = join(target_root, '../saved_samples', args.name, 'test_{}'.format(args.case),\
                     'query_resolution{}'.format(args.query_resolution))
    outfit_names = [x for x in os.listdir(pcl_result_root)]

    for outfit in outfit_names:
        ply_folder = join(pcl_result_root, outfit)
    
        print('parsing pcl files at {}..'.format(ply_folder))
        flist = sorted(glob.glob(join(ply_folder, '*{}'.format(ext))))

        render_savedir = join(render_savedir_root, outfit)
        os.makedirs(render_savedir, exist_ok=True)
        
        for fn in tqdm(flist):
            bn = basename(fn)
            img_save_fn = join(render_savedir, bn.replace('{}'.format(ext), '.png'))
            render_pcl_front_view(vis, cam_params_front, fn, img_save_fn, pt_size=pt_size)


if __name__ == '__main__':
    main()