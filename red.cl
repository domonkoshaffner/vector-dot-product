__kernel void reduction(__global int* source, __global int* result, __local int* temporary, int index) 
{
    // each thread loads one element from global to shared mem
    unsigned int gid = get_local_id(0);
    unsigned int i   = get_global_id(0);
    
    temporary[gid] = source[i];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // do reduction in shared mem
    for(unsigned int s=1; s < get_local_size(0); s *= 2)
    {
        if (gid % (2*s) == 0)
        {
            temporary[gid] += temporary[gid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // write result for this block to global mem
    if(gid == 0){ result[index] = temporary[0]; }
}