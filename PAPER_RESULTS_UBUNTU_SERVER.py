import os
import shutil
# This script is for benchmarking on Linux


# Prerequisites:
# - an "install" folder in the same folder as this script
# - a folder "PAPER_PHANTOMS" with the phantoms
# - PYTHONPATH and LD_LIBRARY_PATH correctly set up
# - a valid beam profile LUT
res_dir = "Server"
python_cmd = "python"
num_cpu_cores = 8
lut_file = "lut_cardiac_phased_array_71_elements.h5"

if not os.path.exists(res_dir): os.makedirs(res_dir)

print "Recording Git hash..."
os.system("git rev-parse HEAD > %s" % (os.path.join(res_dir, "git_HEAD_hash.txt")))

print "Measuring memory transfer speed..."
os.system("install/bin/gpu_memcpy_speedtest > %s" % os.path.join(res_dir, "memcpy_speedtest.txt"))

print "Performing spline evaluation experiment..."
os.system("install/bin/gpu_render_spline_comparison > %s" % os.path.join(res_dir, "gpu_spline_render.txt"))
os.chdir("python")

def DeleteFiles(file_list):
    for f in file_list:
        if os.path.exists(f): os.remove(f)

def CopyFiles(file_list, dst_dir):
    for f in file_list:
        if os.path.exists(f): shutil.copyfile(f, os.path.join(dst_dir, f))
    
print "Benchmarking simulator performance on all phantoms"
img_files = ["benchmark_carotid_plaque_cpu.png",
             "benchmark_carotid_plaque_gpu.png",
             "benchmark_contracting_cylinder_spline_cpu.png",
             "benchmark_contracting_cylinder_spline_gpu.png",
             "benchmark_lv_spline_model_cpu.png",
             "benchmark_lv_spline_model_gpu.png",
             "benchmark_rot_cube_cpu.png",
             "benchmark_rot_cube_gpu.png",
             "benchmark_simple_cpu.png",
             "benchmark_simple_gpu.png",
             "benchmark_tissue_with_parabolic_flow_cpu.png",
             "benchmark_tissue_with_parabolic_flow_gpu.png"]
             
res_file = os.path.join(os.path.join("..", res_dir), "all_benchmark.txt")
DeleteFiles(img_files)
os.system("%s benchmark_paper.py --save_pdf --enable_cpu --num_cpu_cores %d --phantom_folder ../PAPER_PHANTOMS %s" % (python_cmd, num_cpu_cores, res_file))
CopyFiles(img_files, os.path.join("..", res_dir))

print "Benchmarking combinations of phasedelay and LUT"
res_file = os.path.join(os.path.join("..", res_dir), "benchmark_phasedelay_ON_beamprofile_ANALYTICAL.txt")
os.system("%s benchmark_paper.py --phase_delay on --phantom_folder ../PAPER_PHANTOMS %s" % (python_cmd, res_file))

res_file = os.path.join(os.path.join("..", res_dir), "benchmark_phasedelay_OFF_beamprofile_ANALYTICAL.txt")
os.system("%s benchmark_paper.py --phase_delay off --phantom_folder ../PAPER_PHANTOMS %s" % (python_cmd, res_file))

res_file = os.path.join(os.path.join("..", res_dir), "benchmark_phasedelay_ON_beamprofile_LUT.txt")
os.system("%s benchmark_paper.py --phase_delay on --lut_file %s --phantom_folder ../PAPER_PHANTOMS %s" % (python_cmd, os.path.join("..", lut_file), res_file))

res_file = os.path.join(os.path.join("..", res_dir), "benchmark_phasedelay_OFF_beamprofile_LUT.txt")
os.system("%s benchmark_paper.py --phase_delay off --lut_file %s --phantom_folder ../PAPER_PHANTOMS %s" % (python_cmd, os.path.join("..", lut_file), res_file))

print "Performing spline-vs-fixed M-mode experiment..."
img_files = ["mmode_fixed_alg.png", "mmode_spline_alg.png"]
DeleteFiles(img_files)
out_file = os.path.join(os.path.join("..", res_dir), "spline_vs_fixed_console.txt")
os.system("%s demo_mmode_gpu_spline_vs_fixed.py --only_save_png --scatterer_file ../PAPER_PHANTOMS/lv_spline_model.h5 > %s" % (python_cmd, out_file))
CopyFiles(img_files, os.path.join("..", res_dir))

os.chdir("..")

os.system("7za a -tzip UbuntuResults.zip %s" % res_dir)
 
