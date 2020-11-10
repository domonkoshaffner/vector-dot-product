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
// Creating vectors and variables

        constexpr int size = 1000000;
        constexpr int workgroupsize = 256;
        float cpu_result = 0;
        int zeros_to_fill = workgroupsize - (size % workgroupsize);

        std::vector<float> vec_zeros(zeros_to_fill, 0.0f);
        std::vector<float> vec1(size);
        std::vector<float> vec2(size);
        std::vector<float> final_res(workgroupsize);


        // Algorithm for uniform distribution between -1 and 1
        std::random_device rnd_device;
        std::mt19937 mersenne_engine(rnd_device());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        auto gen = [&]() { return dist(mersenne_engine); };

        // Filling up the vectors with random uniform distribution
        std::generate(vec1.begin(), vec1.end(), gen); 
        std::generate(vec2.begin(), vec2.end(), gen);

        // Checking the length of the vectors and filling them up with zeros
        // So their length is n*256 (workgroup size)
        vec1.insert( vec1.end(), vec_zeros.begin(), vec_zeros.end() );
        vec2.insert( vec2.end(), vec_zeros.begin(), vec_zeros.end() );

        // Creating the first result vector
        std::vector<float> result_vec(vec1.size());

// ############################################################################   
// Creating buffers, copying to the GPU, measuring time 
// First kernel: multiplication: vec1[n]*vec2[n]

        // Creating buffers from the vectors
        cl::Buffer buf_vec1{ std::begin(vec1), std::end(vec1), true };
        cl::Buffer buf_vec2{ std::begin(vec2), std::end(vec2), true };
        cl::Buffer buf_result_vec {std::begin(result_vec), std::end(result_vec), false };

        // Starting the clock
        auto time_gpu_start = std::chrono::high_resolution_clock::now();

        // Explicit (blocking) dispatch of data before launch
        cl::copy(queue, std::begin(vec1), std::end(vec1), buf_vec1);
        cl::copy(queue, std::begin(vec2), std::end(vec2), buf_vec2);
        cl::copy(queue, std::begin(result_vec), std::end(result_vec), buf_result_vec);

        // Launching the kernels
        dot_product(cl::EnqueueArgs{queue, cl::NDRange{vec1.size()}}, buf_vec1, buf_vec2, buf_result_vec);

        // Finishing the kernels
        cl::finish();

        // (Blocking) fetch of results
        cl::copy(queue, buf_result_vec, std::begin(result_vec), std::end(result_vec));
        cl::copy(queue, std::begin(result_vec), std::end(result_vec), buf_result_vec);

// ############################################################################ 
// Launching the kernels for summing up the result_vec

        // Calculating the number of iterations
        int num_of_it =  floor( log(size)/log(256) );

        // Every loop reduces the data to size()/256 till the final result is a scalar
        for (int i=0; i < num_of_it + 1; ++i)
        {
            if(result_vec.size() % workgroupsize != 0)
            {
                // First we need to fill up the vectors with zeros, so the size() is divisible by 256
                int zeros_to_fill2 = workgroupsize - (result_vec.size() % workgroupsize);
                std::vector<float> vec_zeros2(zeros_to_fill2, 0.0f);
                result_vec.insert( result_vec.end(), vec_zeros2.begin(), vec_zeros2.end() );
            }

            // Creating a temporary vector to store the temporary results
            std::vector<float> result_temp(result_vec.size(), 0.0f);

            // Creating the buffers and copying the to the GPU
            cl::Buffer buf_result {std::begin(result_vec), std::end(result_vec), true };
            cl::Buffer buf_result_temp {std::begin(result_temp), std::end(result_temp), false };

            cl::copy(queue, std::begin(result_vec), std::end(result_vec), buf_result_vec);
            cl::copy(queue, std::begin(result_temp), std::end(result_temp), buf_result_temp);

            // Launching the kernels
            reduction(cl::EnqueueArgs{queue, cl::NDRange{vec1.size()}, cl::NDRange{workgroupsize}}, buf_result, buf_result_temp, cl::Local(workgroupsize*sizeof(float)));
            
            // Finishing the kernels
            cl::finish();

            // (Blocking) fetch of results
            cl::copy(queue, buf_result_temp, std::begin(result_temp), std::end(result_temp));

            result_vec = result_temp;       

            if(i == num_of_it)
            {
                cl::copy(queue, buf_result_temp, std::begin(final_res), std::end(final_res)); 
            }
            
        }

        auto time_gpu_end = std::chrono::high_resolution_clock::now();

// ############################################################################ 
// Dot product calculation on the CPU

        auto time_cpu_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < vec1.size(); ++i)
        {
            cpu_result += vec1[i]*vec2[i];
        }

        auto time_cpu_end = std::chrono::high_resolution_clock::now();

// ############################################################################ 
// Calculating the time differences and printing the results

        auto time_difference_cpu = std::chrono::duration_cast<std::chrono::microseconds>(time_gpu_end-time_gpu_start).count()/1000.0;  
        auto time_difference_gpu = std::chrono::duration_cast<std::chrono::microseconds>(time_cpu_end-time_cpu_start).count()/1000.0;  

        std::cout << std::fixed << std::setprecision(10);

        std::cout << std::endl << "The CPU result is: " << cpu_result << ".";
        std::cout << std::endl << "The GPU result is: " << final_res[0] << ".";

        std::cout << std::fixed << std::setprecision(0) << std::endl;

        std::cout << std::endl << "The computational time for a " << size << " long vector dot product on the CPU: " << time_difference_cpu  << " milisec.";
        std::cout << std::endl << "The computational time for a " << size << " long vector dot product on the GPU: " << time_difference_gpu  << " milisec." << std::endl;

        float max_acc_diff = 1e-4;
        
        if (abs(final_res[0] - cpu_result) < max_acc_diff)
        {
            std::cout << std::endl << "The results are the same with respect to the given tolerance!" << std::endl;
        }

        else
        {
            std::cout << std::endl << "The results are not quite the same." << std::endl;
        }
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

