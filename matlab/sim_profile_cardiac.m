function sim_profile_cardiac(geo, h5_file_out, job_info)
    % SIM_PROFILE_CARDIAC
    %
    % geo: geometry struct with following fields: x_min, x_max, num_x, y_min, y_max, num_y, z_min, z_max, num_z
    %      Axis interpretation is z~radial, x~lateral, y~elevational.
    % h5_file_out : name of output file with lookup-table. WARNING: Will be overwritten!
    % job_info (optional) : Used to divide work across multiple CPUs [num_jobs, cur. job no.]
    %
    if nargin < 3
        % default to only one job
        jon_info = [1, 1];
    end

    field_init(0);

    % Script for simulating two-way pulse-echo profile for
    % a phased array cardiac probe.
    % Based on the example http://field-ii.dk/?examples/kidney_example/kidney_example.html

    center_freq = 3.0e6;
    fs = 100e6;
    c0 = 1540.0;
    lambda = c0/center_freq;
    el_width = 0.5*lambda;
    el_height = 15e-3;
    kerf = lambda/20;
    num_elements = 71;
    show_plots = false;

    probe_width = num_elements*(el_width+kerf);
    fprintf('Probe width is %2.1f mm\n', 1000*probe_width);

    % fixed transmission focus
    tx_focus = 70e-3;

    % fixed elevation focus (fixed acoustic lens)
    elevation_focus = 70e-3;

    % duration of impulse response
    imp_resp_num_cycles = 2;

    % duration of excitation signal
    excitation_num_cycles = 2;

    % number of subdivisions into mathematical elements in x and y
    num_sub_x = 5;
    num_sub_y = 25;

    set_sampling(fs);

    % frequency dependent dB/[MHz cm]
    freq_att = 0.5; 

    % compute frequency independent
    freq_att_temp = freq_att*100/1e6;
    att = freq_att_temp*center_freq;
    set_field('att', att);
    set_field('Freq_att', freq_att_temp);
    set_field('att_f0', center_freq);

    % configure attenuation
    set_field('use_att', 1);

    % transmission aperture w/fixed elevation focus
    tx_aperture = xdc_focused_array(num_elements, el_width, el_height, kerf, elevation_focus,...
                                    num_sub_x, num_sub_y, [0.0 0.0 tx_focus]);

    % impulse response and excitation for the transmission aperture
    imp_resp_times = 0:(1.0/fs):(imp_resp_num_cycles/center_freq);
    imp_resp = sin(2*pi*center_freq*imp_resp_times);
    if show_plots
        figure(1);
        plot(imp_resp_times, imp_resp);
        title('Impulse response');
        xlabel('Time [s]');
    end
    
    imp_resp = imp_resp.*hanning(max(size(imp_resp)))';
    xdc_impulse(tx_aperture, imp_resp);

    excitation_times = 0:(1.0/fs):(excitation_num_cycles/center_freq);
    excitation = sin(2*pi*center_freq*excitation_times);
    
    if show_plots
        figure(2);
        plot(excitation_times, excitation);
        title('Excitation signal');
        xlabel('Time [s]');
    end
    xdc_excitation(tx_aperture, excitation);

    % reception aperture
    rx_aperture = xdc_focused_array(num_elements, el_width, el_height, kerf, elevation_focus,...
                                    num_sub_x, num_sub_y, [0.0 0.0 tx_focus]);
    xdc_impulse(rx_aperture, imp_resp);

    % set apodization
    apodization = hanning(num_elements)';
    xdc_apodization(tx_aperture, 0, apodization);
    xdc_apodization(rx_aperture, 0, apodization);

    % focus at this angle - dynamic focus on receive
    theta = 0.0;
    xdc_focus(tx_aperture, 0, [tx_focus*sin(theta) 0.0 tx_focus*cos(theta)]);
    xdc_dynamic_focus(rx_aperture, 0.0, theta, 0.0);

    delete(h5_file_out);
    tic;
    sim_lut_hdf5(h5_file_out, tx_aperture, rx_aperture, geo, job_info);
    elapsed_time = toc;
    fprintf('Simulation time was %f seconds\n', elapsed_time);

    xdc_free(tx_aperture);
    xdc_free(rx_aperture);
    
