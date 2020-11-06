__kernel void dot_product(__global int* x, __global int* y, __global int* result_vec)
{
    int gid = get_global_id(0);

    result_vec[gid] = x[gid]*y[gid];
}
