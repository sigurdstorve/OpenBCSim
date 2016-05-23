function sim_lut_hdf5(h5_file, xmit_aperture, receive_aperture, geo)
    % SIM_LUT_HDF5
    % Simulate beam profile in 3D array and write OpenBCSim lookup-table
    % to a hdf5 file. Uses Field II calc_hhp() internally.
    %
    % h5_file: name of output hdf5 file
    % xmit_aperture: transmit aperture
    % receive_aperture: receive aperture
    % geo: struct containing geometrical extent
    
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
    intensities = zeros(geo.num_y, geo.num_x, geo.num_z);
    for ix = 1:geo.num_x
        fprintf('ix = %d of %d\n', ix, geo.num_x)
        for iy = 1:geo.num_y
            for iz = 1:geo.num_z
                [p, start_time] = calc_hhp(xmit_aperture, receive_aperture, [xs_(ix) ys_(iy) zs_(iz)]);
                % TODO: use sum of squares instead of just max value in time dimension?
                intensities(iy, ix, iz) = max(p);
            end
        end
    end

    % map intensities to [0, 1]
    min_value = min(intensities(:));
    max_value = max(intensities(:));
    intensities = (intensities - min_value)/(max_value - min_value);

    % store lookup-table and geometry to hdf5 file [using 32-bit floats]
    h5create(h5_file, '/beam_profile', size(intensities), 'DataType', 'single');
    h5write(h5_file, '/beam_profile', intensities);
    h5create(h5_file, '/ele_extent', 2, 'DataType', 'single');
    h5write(h5_file, '/ele_extent', [geo.y_min, geo.y_max]);
    h5create(h5_file, '/lat_extent', 2, 'DataType', 'single');
    h5write(h5_file, '/lat_extent', [geo.x_min, geo.x_max]);
    h5create(h5_file, '/rad_extent', 2, 'DataType', 'single');
    h5write(h5_file, '/rad_extent', [geo.z_min, geo.z_max]);
