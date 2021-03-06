__kernel void reduction(__global float* x, __global float* y, __global float* result, __local float* localData, int ind, int total_size, int zeros_number) 
{
    int gid = get_global_id(0);        // id of the work item amongs every work items 
    int lid = get_local_id(0);         // id of the work item in the workgroup
    int localSize = get_local_size(0); // size of the workgroup


    if(ind==0)
    {
        result[gid] = x[gid]*y[gid];
    }


    else
    {
    // transfer from global to local memory

        localData[lid] = 0.0f;

        if(ind % 2 == 0)
        {
            if (gid < total_size-zeros_number)
            {
                localData[lid] = x[gid];
            }
        }

        else
        {
            if (gid < total_size-zeros_number)
            {
                localData[lid] = result[gid];
            }
        }

        // make sure everything up to this point in the workgroup finished executing
        barrier(CLK_LOCAL_MEM_FENCE);

        // perform reduction in local memory
        for(unsigned int s = 1; s < localSize; s *= 2)
        {
            // sum every adjacent pairs
            if (lid % (2 * s) == 0)
            {
                localData[lid] += localData[lid + s];
            }

            // make sure everything is synchronized properly before we go into the next iteration
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if(lid == 0)
        {
            if(ind % 2 == 0)
            {
                // id of the workgroup = get_group_id()
                result[get_group_id(0)] =  localData[0];
            }
            else
            {
                 // id of the workgroup = get_group_id()
                x[get_group_id(0)] =  localData[0];               
            }
        }       
    }

}
