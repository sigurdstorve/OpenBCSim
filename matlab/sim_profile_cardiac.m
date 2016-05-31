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

probe_width = num_elements*(el_width+kerf);
fprintf('Probe width is %2.1f mm\n', 1000*probe_width);

% fixed transmission focus
tx_focus = 70e-3;

% fixed elevation focus (acoustic lens)
elevation_focus = 70e-3;

% duration of impulse response
imp_resp_num_cycles = 2;

% duration of excitation signal
excitation_num_cycles = 2;

% number of subdivisions into mathematical elements in x and y
num_sub_x = 5;
num_sub_y = 25;

% name of output file with lookup-table. WARNING: Will be overwritten!
h5_file_out = 'beam_profile_cardiac.h5';

set_sampling(fs);
set_field('show_times', 5);

% transmission aperture w/fixed elevation focus
tx_aperture = xdc_focused_array(num_elements, el_width, el_height, kerf, elevation_focus,...
                                num_sub_x, num_sub_y, [0.0 0.0 tx_focus]);

% impulse response and excitation for the transmission aperture
imp_resp_times = 0:(1.0/fs):(imp_resp_num_cycles/center_freq);
imp_resp = sin(2*pi*center_freq*imp_resp_times);
figure(1);
plot(imp_resp_times, imp_resp);
title('Impulse response');
xlabel('Time [s]');

imp_resp = imp_resp.*hanning(max(size(imp_resp)))';
xdc_impulse(tx_aperture, imp_resp);

excitation_times = 0:(1.0/fs):(excitation_num_cycles/center_freq);
excitation = sin(2*pi*center_freq*excitation_times);
figure(2);
plot(excitation_times, excitation);
title('Excitation signal');
xlabel('Time [s]');

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

% create geometry struct for volume in which to simulate two-way sensitivity
geo = struct();
geo.z_min = 1e-3;
geo.z_max = 160e-3;
geo.num_z = 1024;
geo.x_min = -2e-2;
geo.x_max = 2e-2;
geo.num_x = 128;
geo.y_min = -2e-2;
geo.y_max = 2e-2;
geo.num_y = 128;

delete(h5_file_out);
sim_lut_hdf5(h5_file_out, tx_aperture, rx_aperture, geo);

xdc_free(tx_aperture);
xdc_free(rx_aperture);