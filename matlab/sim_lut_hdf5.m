function sim_lut_hdf5(h5_file, xmit_aperture, receive_aperture, geo, job_info)
    % SIM_LUT_HDF5
    % Simulate beam profile in 3D array and write OpenBCSim lookup-table
    % to a hdf5 file. Uses Field II calc_hhp() internally.
    %
    % h5_file: name of output hdf5 file
    % xmit_aperture: transmit aperture
    % receive_aperture: receive aperture
    % geo: struct containing geometrical extent
    % job_info: (optional) if specified, simulate only a range of the x indices
    %         (useful for multi-CPU simulations)
    if nargin < 4
        error('the first four arguments are mandatory');
    end
    num_jobs = 1;
    cur_job = 1;
    if nargin == 5
        if length(job_info) ~= 2
            error('job_info must have length 2');
        end
        num_jobs = job_info(1);
        cur_job = job_info(2);
    end
    % map number of jobs and cur. job no to limits for x index
    interval_limits = int32(linspace(1, geo.num_x, num_jobs+1));
    xi_start = interval_limits(cur_job);
    xi_end   = interval_limits(cur_job+1)-1;
    if cur_job == num_jobs
        xi_end = xi_end + 1; % make last interval closed 
    end
    fprintf('Simulation job %d of %d : x interval is [%d, %d]\n', cur_job, num_jobs, xi_start, xi_end);
    
    % create all simulation points
    xs_ = linspace(geo.x_min, geo.x_max, geo.num_x);
    ys_ = linspace(geo.y_min, geo.y_max, geo.num_y);
    zs_ = linspace(geo.z_min, geo.z_max, geo.num_z);
    
    % nested loops instead of simulating all points in one batch due to
    % observed out-of-memory issues observed with the batch-solution..
    %
    % dimensions are in reverse order since Matlab uses column-major and
    % the HDF5 library uses row-major..
    % (the simulator expects: dim0~radial, dim1~lateral, dim2~elevational)
    %
    % using zeros to support multi-CPU simulation with different Matlab
    % instances; combination is done by simple summation with a Python script
    % afterwards.
    intensities = zeros(geo.num_y, geo.num_x, geo.num_z);
    for ix = xi_start:xi_end
        fprintf('ix = %d of %d\n', ix, xi_end)
        for iy = 1:geo.num_y
            for iz = 1:geo.num_z
                [p, start_time] = calc_hhp(xmit_aperture, receive_aperture, [xs_(ix) ys_(iy) zs_(iz)]);
                intensities(iy, ix, iz) = rssq(p); % root-sum-of-squares
            end
        end
    end

    % map intensities to [0, 1] - ONLY IF ONE JOB
    if num_jobs == 1
        min_value = min(intensities(:));
        max_value = max(intensities(:));
        intensities = (intensities - min_value)/(max_value - min_value);
    end
    % store lookup-table and geometry to hdf5 file [using 32-bit floats]
    h5create(h5_file, '/beam_profile', size(intensities), 'DataType', 'single');
    h5write(h5_file, '/beam_profile', single(intensities));
    h5create(h5_file, '/ele_extent', 2, 'DataType', 'single');
    h5write(h5_file, '/ele_extent', single([geo.y_min, geo.y_max]));
    h5create(h5_file, '/lat_extent', 2, 'DataType', 'single');
    h5write(h5_file, '/lat_extent', single([geo.x_min, geo.x_max]));
    h5create(h5_file, '/rad_extent', 2, 'DataType', 'single');
    h5write(h5_file, '/rad_extent', single([geo.z_min, geo.z_max]));
