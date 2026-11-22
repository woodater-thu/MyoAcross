"""Read-only reduction of MuJoCo-Warp world-frame solver contact forces."""

import warp as wp


@wp.kernel
def accumulate_foot_forces(
    nacon: wp.array[int], geoms: wp.array[wp.vec2i], worlds: wp.array[int],
    forces: wp.array[wp.spatial_vector], sides: wp.array[int], ground: wp.array[int],
    output: wp.array2d[wp.vec3],
):
    i = wp.tid()
    if i < nacon[0]:
        pair = geoms[i]
        first, second = pair[0], pair[1]
        if ground[first] != 0 and sides[second] >= 0:
            wp.atomic_add(output, worlds[i], sides[second], wp.spatial_top(forces[i]))
        elif ground[second] != 0 and sides[first] >= 0:
            wp.atomic_add(output, worlds[i], sides[first], -wp.spatial_top(forces[i]))
