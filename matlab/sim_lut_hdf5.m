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
    [xx, yy, zz] = ndgrid(xs_, ys_, zs_); 
    all_points = [xx(:) yy(:) zz(:)];

    [p, start_time] = calc_hhp(xmit_aperture, receive_aperture, all_points);

    % Map to scalar intensities
    % dim1: time
    % dim2: point index
    % TODO: use sum of squares instead of just max value in time dimension?
    intensities = max(p, [], 1);

    % map intensities to [0, 1]
    min_value = min(intensities);
    max_value = max(intensities);
    intensities = (intensities - min_value)/(max_value - min_value);

    % reshape
    intensities = reshape(intensities, [geo.num_x geo.num_y geo.num_z]);

    % reorder dimensions to expected: [z, x, y] ~ [rad, lat, ele]
    % ==> not needed size matlab uses column-major...
    %intensities = permute(intensities, [3, 2, 1]);

    % store lookup-table and geometry to hdf5 file
    h5create(h5_file, '/beam_profile', size(intensities), 'DataType', 'single');
    h5write(h5_file, '/beam_profile', intensities);
    h5create(h5_file, '/ele_extent', 2, 'DataType', 'single');
    h5write(h5_file, '/ele_extent', [geo.y_min, geo.y_max]);
    h5create(h5_file, '/lat_extent', 2, 'DataType', 'single');
    h5write(h5_file, '/lat_extent', [geo.x_min, geo.x_max]);
    h5create(h5_file, '/rad_extent', 2, 'DataType', 'single');
    h5write(h5_file, '/rad_extent', [geo.z_min, geo.z_max]);
