#include <CL/cl2.hpp>

#include <chrono>
#include <numeric>
#include <iterator>

#include <vector>       // std::vector
#include <exception>    // std::runtime_error, std::exception
#include <iostream>     // std::cout
#include <fstream>      // std::ifstream
#include <random>       // std::default_random_engine, std::uniform_real_distribution
#include <algorithm>    // std::transform
#include <cstdlib>      // EXIT_FAILURE
#include <iomanip>
#include <math.h>

int main()
{
    try
    {
        // Checking the device info
        cl::CommandQueue queue = cl::CommandQueue::getDefault();
        cl::Device device = queue.getInfo<CL_QUEUE_DEVICE>();

        // Load program source
        std::ifstream file1{ "C:/Users/haffn/Desktop/MSc-III/GPU-II/Projects/first project/dot_product.cl" };
        std::ifstream file2{ "C:/Users/haffn/Desktop/MSc-III/GPU-II/Projects/first project/red.cl" };


        if (!file1.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "dot_product.cl" };
        if (!file2.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "red.cl" };

        // Creating the program
        cl::Program dot_pr_kernel{ std::string{ std::istreambuf_iterator<char>{ file1 }, std::istreambuf_iterator<char>{} } };
        cl::Program reduction_kernel{ std::string{ std::istreambuf_iterator<char>{ file2 }, std::istreambuf_iterator<char>{} } };

        //Building the program
        dot_pr_kernel.build({ device });
        reduction_kernel.build({ device });

        // Creating the kernel
        auto dot_product = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>(dot_pr_kernel, "dot_product");
        auto reduction = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg>(reduction_kernel, "reduction");

        //auto workgroupsize = kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(device);


// ############################################################################ 
// Creating the vectors and variables

        constexpr int size = 1;
        constexpr int workgroupsize = 256;
        auto cpu_result = 0;
        int zeros_to_fill = workgroupsize - (size % workgroupsize);

        std::vector<int> vec_zeros(zeros_to_fill, 0);
        std::vector<int> vec1(size);
        std::vector<int> vec2(size);
        std::vector<int> final_res(workgroupsize);

        // Filling up the vectors
        for (int i = 0; i < vec1.size(); ++i)
        {
            if ( i % 2 == 0)
            {
                vec1[i] = 1;
                vec2[i] = 1;
            }
            else
            {
                vec1[i] = 1;
                vec2[i] = 1;            
            }
        }

        // Checking the length of the vectors and filling them up with zeros
        // So their length is n*256 (workgroup size)

        vec1.insert( vec1.end(), vec_zeros.begin(), vec_zeros.end() );
        vec2.insert( vec2.end(), vec_zeros.begin(), vec_zeros.end() );

        std::vector<int> result_vec(vec1.size());
        std::vector<int> result2(workgroupsize, 0);

// ############################################################################   
// Creating buffers, copying to the GPU, measuring time 

        // Creating buffers from the vectors
        cl::Buffer buf_vec1{ std::begin(vec1), std::end(vec1), true };
        cl::Buffer buf_vec2{ std::begin(vec2), std::end(vec2), true };
        cl::Buffer buf_result_vec {std::begin(result_vec), std::end(result_vec), false };

        // Explicit (blocking) dispatch of data before launch
        cl::copy(queue, std::begin(vec1), std::end(vec1), buf_vec1);
        cl::copy(queue, std::begin(vec2), std::end(vec2), buf_vec2);
        cl::copy(queue, std::begin(result_vec), std::end(result_vec), buf_result_vec);

        // Launch kernels
        dot_product(cl::EnqueueArgs{queue, cl::NDRange{vec1.size()}}, buf_vec1, buf_vec2, buf_result_vec);

        // Finishing kernels
        cl::finish();

        // (Blocking) fetch of results
        cl::copy(queue, buf_result_vec, std::begin(result_vec), std::end(result_vec));
        cl::copy(queue, std::begin(result_vec), std::end(result_vec), buf_result_vec);

// ############################################################################ 
// Launching the kernels

        int num_of_it =  floor( log(size)/log(256) );

        std::cout << num_of_it << std::endl;

        for (int i=0; i < num_of_it + 1; ++i)
        {
            std::cout << i << std::endl;

            if(result_vec.size() % workgroupsize != 0)
            {
                int zeros_to_fill2 = workgroupsize - (result_vec.size() % workgroupsize);
                std::vector<int> vec_zeros2(zeros_to_fill2, 0);
                result_vec.insert( result_vec.end(), vec_zeros2.begin(), vec_zeros2.end() );

            }

            std::vector<int> result3(result_vec.size(), 0);

            cl::Buffer buf_result {std::begin(result_vec), std::end(result_vec), true };
            cl::Buffer buf_result3 {std::begin(result3), std::end(result3), false };

            cl::copy(queue, std::begin(result_vec), std::end(result_vec), buf_result_vec);
            cl::copy(queue, std::begin(result3), std::end(result3), buf_result3);

            
            reduction(cl::EnqueueArgs{queue, cl::NDRange{vec1.size()}, cl::NDRange{workgroupsize}}, buf_result, buf_result3, cl::Local(workgroupsize*sizeof(int)));
            
            cl::finish();

            // (Blocking) fetch of results
            cl::copy(queue, buf_result3, std::begin(result3), std::end(result3));

            result_vec = result3;       

            if(i == num_of_it)
            {
                cl::copy(queue, buf_result3, std::begin(final_res), std::end(final_res)); 
            }
            
        }

// ############################################################################ 
// Doing the same calculation on the CPU

        for (int i = 0; i < vec1.size(); ++i)
        {
            cpu_result += vec1[i]*vec2[i];
        }

// ############################################################################ 
// Printing the results

        std::cout << std::endl << "The CPU result is: " << cpu_result << ".";
        std::cout << std::endl << "The GPU result is: " << final_res[0] << ".";
        std::cout << std::endl;

    }

// ############################################################################
// If kernel failed to build

    
    catch (cl::BuildError& error) 
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;

        for (const auto& log : error.getBuildLog())
        {
            std::cerr <<
                "\tBuild log for device: " <<
                log.first.getInfo<CL_DEVICE_NAME>() <<
                std::endl << std::endl <<
                log.second <<
                std::endl << std::endl;
        }

        std::exit(error.err());
    }
    catch (cl::Error& error) // If any OpenCL error occurs
    {
        std::cerr << error.what() << "(" << error.err() << ")" << std::endl;
        std::exit(error.err());
    }
    catch (std::exception& error) // If STL/CRT error occurs
    {
        std::cerr << error.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}



/*

        int zeros_to_fill = workgroupsize - (size % workgroupsize);
        std::vector<int> vec_zeros(zeros_to_fill, 0);
        result_vec.insert( result_vec.end(), vec_zeros.begin(), vec_zeros.end() );

        int num_of_it =  floor( log(size)/log(256) );

        std::cout << std::endl << num_of_it << std::endl;

        int num_of_iterations = (size + zeros_to_fill)/256;

        int new_len = result_vec.size()/256;
        int zeros_to_fill2 = workgroupsize - (new_len % workgroupsize);
        int new_size = new_len + zeros_to_fill2;

        std::cout << std::endl << new_size << std::endl;

        std::vector<int> result3(new_size, 0);


*/