import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend for headless environment
import matplotlib.pyplot as plt
import argparse
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    AmbientLights,
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
)
from pytorch3d.renderer.blending import BlendParams
from sugar_scene.gs_model import GaussianSplattingWrapper
from sugar_scene.sugar_model import SuGaR, load_refined_model

def parse_args():
    parser = argparse.ArgumentParser(description='SuGaR Viewer: Render scenes using SuGaR models')
    
    # Basic parameters
    parser.add_argument('--source_path', type=str, required=True,
                        help='Path to the scene data')
    parser.add_argument('--gs_checkpoint_path', type=str, required=True,
                        help='Path to the vanilla Gaussian Splatting checkpoint')
    parser.add_argument('--refined_sugar_folder', type=str, required=True,
                        help='Path to the refined SuGaR folder')
    parser.add_argument('--refined_iteration', type=int, default=15000,
                        help='Refinement iteration to load')
    
    # Camera & rendering options
    parser.add_argument('--camera_index', type=int, default=-1,
                        help='Camera index to render (-1 for random)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device index')
    parser.add_argument('--plot_ratio', type=float, default=2.0,
                        help='Size ratio for output images')
    parser.add_argument('--render_mesh', action='store_true',
                        help='Render textured mesh view')
    parser.add_argument('--render_vanilla', action='store_true',
                        help='Render vanilla 3DGS view')
    parser.add_argument('--output_dir', type=str, default='./renders',
                        help='Directory to save rendered images')
    parser.add_argument('--eval_split', action='store_true',
                        help='Use evaluation split')
    parser.add_argument('--skip_images', type=int, default=8,
                        help='Number of images to skip for eval split')
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set GPU
    torch.cuda.set_device(args.gpu)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data and vanilla Gaussian Splatting model
    print(f"\nLoading config {args.gs_checkpoint_path}...")
    nerfmodel = GaussianSplattingWrapper(
        source_path=args.source_path,
        output_path=args.gs_checkpoint_path,
        iteration_to_load=7000,  # Default iteration for vanilla GS
        load_gt_images=False,
        eval_split=args.eval_split,
        eval_split_interval=args.skip_images,
    )
    
    print(f'{len(nerfmodel.training_cameras)} training images detected.')
    print(f'The model has been trained for 7000 steps.')
    print(f'{len(nerfmodel.gaussians._xyz) / 1e6:.6f} M gaussians detected.')
    
    # Load the refined SuGaR checkpoint
    refined_sugar_path = os.path.join(args.refined_sugar_folder, f"{args.refined_iteration}.pt")
    print(f"\nLoading config {refined_sugar_path}...")
    
    refined_sugar = load_refined_model(refined_sugar_path, nerfmodel)
    
    # Select camera for rendering
    cameras_to_use = nerfmodel.training_cameras
    
    if args.camera_index == -1:
        cam_idx = np.random.randint(0, len(cameras_to_use.gs_cameras))
    else:
        cam_idx = args.camera_index
    
    print(f"Rendering image with index {cam_idx}.")
    print("Image name:", cameras_to_use.gs_cameras[cam_idx].image_name)
    
    # Render images
    refined_sugar.eval()
    refined_sugar.adapt_to_cameras(cameras_to_use)
    
    with torch.no_grad():
        # Render vanilla 3DGS
        if args.render_vanilla:
            gs_image = nerfmodel.render_image(
                nerf_cameras=cameras_to_use,
                camera_indices=cam_idx).clamp(min=0, max=1)
            
            plt.figure(figsize=(10 * args.plot_ratio, 10 * args.plot_ratio))
            plt.axis("off")
            plt.title("Vanilla 3DGS render")
            plt.imshow(gs_image.cpu().numpy())
            
            # Save the image
            vanilla_output_path = os.path.join(args.output_dir, f"vanilla_{cam_idx}.png")
            plt.savefig(vanilla_output_path, bbox_inches='tight')
            plt.close()
            print(f"Vanilla render saved to {vanilla_output_path}")
        
        # Render refined SuGaR
        sugar_image = refined_sugar.render_image_gaussian_rasterizer(
            nerf_cameras=cameras_to_use, 
            camera_indices=cam_idx,
            sh_deg=nerfmodel.gaussians.active_sh_degree,
            compute_color_in_rasterizer=True,
        ).nan_to_num().clamp(min=0, max=1)
        
        plt.figure(figsize=(10 * args.plot_ratio, 10 * args.plot_ratio))
        plt.axis("off")
        plt.title("Refined SuGaR render")
        plt.imshow(sugar_image.cpu().numpy())
        
        # Save the image
        sugar_output_path = os.path.join(args.output_dir, f"sugar_{cam_idx}.png")
        plt.savefig(sugar_output_path, bbox_inches='tight')
        plt.close()
        print(f"SuGaR render saved to {sugar_output_path}")
        
    # Free up memory
    torch.cuda.empty_cache()
    
    # Render with textured mesh if requested
    if args.render_mesh:
        # Find mesh path
        scene_name = args.refined_sugar_folder.rstrip('/').split('/')[-2] # in case of a path like 'output/refined_sugar/scene_name/
        model_name = os.path.basename(args.refined_sugar_folder.rstrip('/'))
        refined_mesh_dir = './output/refined_mesh'
        refined_mesh_path = os.path.join(refined_mesh_dir, scene_name, f"{model_name}.obj")
        
        print(f"Loading refined mesh from {refined_mesh_path}, this could take a minute...")
        try:
            textured_mesh = load_objs_as_meshes([refined_mesh_path]).to(nerfmodel.device)
            print(f"Loaded textured mesh with {len(textured_mesh.verts_list()[0])} vertices and {len(textured_mesh.faces_list()[0])} faces.")
            
            # Render the mesh
            mesh_raster_settings = RasterizationSettings(
                image_size=(refined_sugar.image_height, refined_sugar.image_width),
                blur_radius=0.0, 
                faces_per_pixel=1
            )
            lights = AmbientLights(device=nerfmodel.device)
            rasterizer = MeshRasterizer(
                cameras=cameras_to_use.p3d_cameras[cam_idx], 
                raster_settings=mesh_raster_settings,
            )
            renderer = MeshRenderer(
                rasterizer=rasterizer,
                shader=SoftPhongShader(
                    device=refined_sugar.device, 
                    cameras=cameras_to_use.p3d_cameras[cam_idx],
                    lights=lights,
                    blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
                )
            )
            
            with torch.no_grad():
                print(f"Rendering mesh with camera {cam_idx}")
                p3d_cameras = cameras_to_use.p3d_cameras[cam_idx]
                rgb_img = renderer(textured_mesh, cameras=p3d_cameras)[0, ..., :3]
            
            plt.figure(figsize=(10 * args.plot_ratio, 10 * args.plot_ratio))
            plt.axis("off")
            plt.title("Refined SuGaR mesh with UV texture")
            plt.imshow(rgb_img.cpu().numpy())
            
            # Save the image
            mesh_output_path = os.path.join(args.output_dir, f"mesh_{cam_idx}.png")
            plt.savefig(mesh_output_path, bbox_inches='tight')
            plt.close()
            print(f"Mesh render saved to {mesh_output_path}")
        except Exception as e:
            print(f"Error loading or rendering mesh: {e}")

if __name__ == "__main__":
    main()