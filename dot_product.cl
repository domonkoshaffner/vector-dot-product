__kernel void dot_product(__global float* x, __global float* y, __global float* result_vec)
{
    int gid = get_global_id(0);

    result_vec[gid] = x[gid]*y[gid];
}
