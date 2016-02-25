import os
import sys
sys.path.append("phantom_scripts")

# This is the main phantom script which creates all phantoms.
# Must be run from the folder which contains this file.

class Args:
    pass

out_dir = "generated_phantoms"
    
def verify_correct_path():
    dirs = [entry for entry in os.listdir('.') if os.path.isdir(entry)]
    if "phantom_scripts" in dirs and "phantom_data" in dirs: return
        
    print "This script must be run from the project root directory."
    exit()

def ensure_output_folder_exists():
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    
def create_artery_phantom():
    """
    Stack cross-sectional splines and interpolate shape
    in between.
    """
    from realistic_artery import create_phantom
    
    args = Args()
    args.x_min = -0.06
    args.x_max = 0.06
    args.h5_out = os.path.join(out_dir, "realistic_artery.h5")
    args.spline_files = ["phantom_data/artery_crossection_splines/spline_000.txt",
                         "phantom_data/artery_crossection_splines/spline_001.txt",
                         "phantom_data/artery_crossection_splines/spline_002.txt",
                         "phantom_data/artery_crossection_splines/spline_003.txt",
                         "phantom_data/artery_crossection_splines/spline_004.txt",
                         "phantom_data/artery_crossection_splines/spline_005.txt",
                         "phantom_data/artery_crossection_splines/spline_006.txt"]
    args.scale = 3e-3
    args.num_scatterers = 2000000
    args.inside_factor = 0.1
    args.outside_factor = 1.0
    args.space_factor = 0.5
                         
    create_phantom(args)

def create_carotid_bifurcation_phantoms():
    """
    Create carotid bifurcation phantom (1) without plaque and (2) with plaque
    """
    from carotid_bifurcation import create_phantom
    args = Args()
    args.z0 = 0.025
    args.x_min = -0.08
    args.x_max = 0.08
    args.y_min = -0.03
    args.y_max = 0.03
    args.z_min = 0.0
    args.z_max = 0.05
    args.num_scatterers = 5000000
    args.small_r = 5e-3
    args.large_r = 8.2e-3
    args.common_x_max = 13e-3
    args.theta = 3.141592*10/180.0
    args.visualize = False
    args.lumen_ampl = 0.1

    args.enable_plaque = False
    args.h5_file = os.path.join(out_dir, "carotid_no_plaque.h5")
    create_phantom(args)

    args.enable_plaque = True
    args.h5_file = os.path.join(out_dir, "carotid_plaque.h5")
    create_phantom(args)

def create_contracting_cylinder():
    """
    A cylinder which contracts according to a scaling signal
    which is a function of time.
    """
    from contracting_cylinder_spline import create_phantom
    args = Args()
    args.h5_out = os.path.join(out_dir, "contracting_cylinder_spline.h5")
    args.h5_scale = "phantom_data/real_left_ventricle_contraction.h5"
    args.r0 = 1e-2
    args.z0 = 0.12
    args.num_scatterers = 20000
    args.num_control_points = 10
    args.spline_degree = 3
    create_phantom(args)

def create_rotating_plaque_phantoms():
    """
    A cross-sectional slice through e.g. an artery with a
    lump of scatterers moving along the outer perimenter to
    simulate a plaque which is moving.
    """
    from rotating_plaque import create_phantom
    args = Args()
    args.x_min = -0.015
    args.x_max = 0.015
    args.z_min = 0.0
    args.z_max = 0.03
    args.num_scatterers = 100000
    args.z0 = 0.015
    args.radius = 5e-3
    args.num_cs = 10
    args.spline_degree = 3
    args.num_plaque_scatterers = 1000
    args.inside_ampl = 0.02

    args.plaque_radius = 1.6e-3
    args.h5_file = os.path.join(out_dir, "rotating_plaque_small.h5")
    create_phantom(args)

    args.plaque_radius = 2.9e-3
    args.h5_file = os.path.join(out_dir, "rotating_plaque_large.h5")
    create_phantom(args)

def create_rotating_cube_phantom():
    """
    A rotating 3D cube of scatterers, which is a good example
    of complex scatterer tracjectories in 3D.
    """
    from rotating_cube import create_phantom
    args = Args()
    args.h5_file = os.path.join(out_dir, "rot_cube.h5")
    args.x_min = -0.03
    args.x_max = 0.03
    args.y_min = -0.03
    args.y_max = 0.03
    args.z_min = -0.0
    args.z_max = 0.03
    args.z0 = 0.06
    args.num_cs = 20
    args.spline_degree = 3
    args.t0 = 0.0
    args.t1 = 1.0
    args.num_scatterers = 100000
    # velocities are chosen so that the motion has a period of one second
    args.x_angular_velocity = 3.14159*2
    args.y_angular_velocity = 3.14159*4
    args.z_angular_velocity = 3.14159*8
    create_phantom(args)

def create_random_spline_noise_phantom():
    """
    Scatterers moving along random 3D trajectories.
    """
    from spline_noise import create_phantom
    args = Args()
    args.h5_file = os.path.join(out_dir, "random_spline_noise.h5")
    args.num_scatterers = 100000
    args.x_min = -0.04
    args.x_max = 0.04
    args.y_min = -0.04 
    args.y_max = 0.04
    args.z_min = 0.01
    args.z_max = 0.08
    args.spline_degree = 3
    args.num_cs = 20
    create_phantom(args)

def create_harmonic_box_phantom():
    """
    A cube of scatterers moving harmonically up and down along
    the Z-axis.
    """
    from harmonic_box import create_phantom
    args = Args()
    args.h5_file = os.path.join(out_dir, "harmonic_box.h5")
    args.x_min = -0.025
    args.x_max = 0.025
    args.y_min = -0.025
    args.y_max = 0.025
    args.thickness = 0.05
    args.z0 = 8e-2
    args.ampl = 1e-2
    args.freq = 1.3
    args.num_scatterers = 100000
    args.num_control_points = 10
    args.t_start = 0.0
    args.t_end = 1.0
    args.spline_degree =3
    create_phantom(args)

def create_lv_spline_phantom():
    """
    3D left ventricle phantom which contracts according to a realistic
    contraction function.
    """
    from lv_spline_model import create_phantom
    args = Args()
    args.h5_file = os.path.join(out_dir, "lv_spline_model.h5")
    args.thickness = 8e-3
    args.z_ratio = 0.7
    args.x_min = -0.02
    args.x_max = 0.02
    args.y_min = -0.02
    args.y_max = 0.02
    args.z_min = 0.008
    args.z_max = 0.09
    args.num_scatterers_in_box = 400000
    args.motion_ampl = 0.25
    args.t0 = 0.0
    args.t1 = 1.0
    args.spline_degree = 2
    args.num_cs = 10
    args.scale_h5_file = "phantom_data/real_left_ventricle_contraction.h5"
    args.lv_max_amplitude = 1.0
    create_phantom(args)

def create_2d_cyst_phantom():
    """
    Create a 2D cyst phantom.
    """
    from cyst_phantom_2d import create_phantom
    args = Args()
    args.h5_file = os.path.join(out_dir, "cyst_2d.h5")
    args.density = 500.0
    args.cyst_scale = 0.3
    create_phantom(args)

def create_simple_phantom():
    """
    A few scatterers along the positive z-axis.
    """
    from simple import create_phantom
    args = Args()
    args.h5_file = os.path.join(out_dir, "simple.h5")
    args.num_scatterers=12
    args.z0 = 0.005
    args.z1 = 0.12
    create_phantom(args)

def create_tissue_flow_phantom():
    from tissue_with_flow import create_phantom
    args = Args()
    args.h5_file = os.path.join(out_dir, "tissue_with_constant_flow.h5")
    args.num_tissue_scatterers = 1000000
    args.num_flow_scatterers = 500000
    args.box_dim = 0.03
    args.radius = 0.008
    args.tissue_length = 8e-2
    args.flow_ampl_factor = 0.2
    args.peak_velocity = 15e-2
    args.end_time = 1.0
    args.exponent = 20 # approximate constant flow
    create_phantom(args)
    
    args.exponent = 2
    args.h5_file = os.path.join(out_dir, "tissue_with_parabolic_flow.h5")
    create_phantom(args)
    
def create_spinning_disk_phantom():
    from spinning_disc import create_phantom
    args = Args()
    args.h5_file = os.path.join(out_dir, "spinning_disc.h5")
    args.degree = 2
    args.num_cs = 10
    args.period = 1.0
    args.num_scatterers = 20000
    args.z0 = 0.025
    args.radius = 2e-2
    create_phantom(args)
    
if __name__ == '__main__':
    verify_correct_path()
    ensure_output_folder_exists()
    create_lv_spline_phantom()
    create_carotid_bifurcation_phantoms()
    create_contracting_cylinder()
    create_rotating_plaque_phantoms()
    create_rotating_cube_phantom()
    create_random_spline_noise_phantom()
    create_harmonic_box_phantom()
    create_2d_cyst_phantom()
    create_simple_phantom()
    create_tissue_flow_phantom()
    create_spinning_disk_phantom()
    print 'NOTE: This is the last script and may take a while to finish...'
    create_artery_phantom()
    