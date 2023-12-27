// Filename: metadata.hpp

#ifndef METADATA_HPP
#define METADATA_HPP

namespace metadata {
    constexpr unsigned int num_anchors = 19125;
    constexpr int apply_exp_scaling = 1;
    constexpr float x_scale = 1.0;
    constexpr float y_scale = 1.0;
    constexpr float w_scale = 1.0;
    constexpr float h_scale = 1.0;
    constexpr unsigned int num_sectors = 4;
    constexpr unsigned int num_anchors_per_coord = 9;
    constexpr unsigned int reset_idxs[] = {
        0, 14400, 18000, 18900
    };
    constexpr unsigned int num_xs_per_y[] = {
        360, 180, 90, 45
    };
    constexpr float x_strides[] = {
        0.025, 0.05, 0.1, 0.2
    };
    constexpr float y_strides[] = {
        0.025, 0.05, 0.1, 0.2
    };
    constexpr float widths[4][9] = {
        {0.05303, 0.075, 0.10607, 0.06682, 0.09449, 0.13364, 0.08418, 0.11905, 0.16837},
        {0.10607, 0.15, 0.21213, 0.13364, 0.18899, 0.26727, 0.16837, 0.23811, 0.33674},
        {0.21213, 0.3, 0.42426, 0.26727, 0.37798, 0.53454, 0.33674, 0.47622, 0.67348},
        {0.42426, 0.6, 0.84853, 0.53454, 0.75595, 1.06908, 0.67348, 0.95244, 1.34695}
    };
    constexpr float heights[4][9] = {
        {0.10607, 0.075, 0.05303, 0.13364, 0.09449, 0.06682, 0.16837, 0.11905, 0.08418},
        {0.21213, 0.15, 0.10607, 0.26727, 0.18899, 0.13364, 0.33674, 0.23811, 0.16837},
        {0.42426, 0.3, 0.21213, 0.53454, 0.37798, 0.26727, 0.67348, 0.47622, 0.33674},
        {0.84853, 0.6, 0.42426, 1.06908, 0.75595, 0.53454, 1.34695, 0.95244, 0.67348}
    };
}

#endif // METADATA_HPP
