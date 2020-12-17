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
        std::ifstream file2{ "C:/Users/haffn/Desktop/MSc-III/GPU-II/Projects/first project/first project - temp/red.cl" };

        // if program failed to open
        if (!file2.is_open())
            throw std::runtime_error{ std::string{ "Cannot open kernel source: " } + "red.cl" };

        // Creating the program
        cl::Program reduction_kernel{ std::string{ std::istreambuf_iterator<char>{ file2 }, std::istreambuf_iterator<char>{} } };

        //Building the program
        reduction_kernel.build({ device });

        // Creating the kernel
        auto reduction = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::LocalSpaceArg, int, int, int>(reduction_kernel, "reduction");

// ############################################################################ 
// Creating the vectors, constants and buffers for the calcualtions

        // Creating the required variables
        constexpr int size = 256*256*256;
        constexpr int workgroupsize = 256;
        float cpu_result = 0;
        int zeros_to_fill = workgroupsize - (size % workgroupsize);
        size_t temp_zeros;
        size_t temp_size;
        size_t newsize;

        // Creating the vectors
        std::vector<float> vec_zeros(zeros_to_fill, 0.0f);
        std::vector<float> vec1(size);
        std::vector<float> vec2(size);
        std::vector<float> vec3(size+zeros_to_fill);
        std::vector<int> zeros;
        std::vector<size_t> size_vec;

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

        // Creating the buffers
        cl::Buffer buf_vec1 {std::begin(vec1), std::end(vec1), true };
        cl::Buffer buf_vec2 {std::begin(vec2), std::end(vec2), true };

// ############################################################################
//Calculating the iterations and the number of zeros we have to append

        int num_of_it =  (int)ceil( log(size)/log(workgroupsize) );

        zeros.push_back(0);
        zeros.push_back(0);
        size_vec.push_back(vec1.size());
        size_vec.push_back(vec1.size());

        for(int i=2; i<=num_of_it; ++i)
        {
            newsize = size_vec[i-1]/256;
            temp_zeros = workgroupsize - (newsize % workgroupsize);
            temp_size = newsize + temp_zeros;

            zeros.push_back((int)temp_zeros);
            size_vec.push_back(temp_size);
        }


// ############################################################################
// Calling the kernels

        // Starting the clock
        auto time_gpu_start = std::chrono::high_resolution_clock::now();

        // Copying the buffers to the GPU
        cl::copy(queue, std::begin(vec1), std::end(vec1), buf_vec1);
        cl::copy(queue, std::begin(vec2), std::end(vec2), buf_vec2);

        // Every loop reduces the data to size()/256 till the final result is a scalar
        for (int i=0; i <= num_of_it; ++i)
        {
            reduction(cl::EnqueueArgs{queue, cl::NDRange{size_vec[i]}, cl::NDRange{workgroupsize}}, buf_vec1, buf_vec2, cl::Local(workgroupsize*sizeof(float)), i, (int)size_vec[i], (int)zeros[i]);
        }

        // Copying back the results
        if(num_of_it % 2 )
        {
            cl::copy(queue, buf_vec1, std::begin(vec3), std::begin(vec3)+1);
        }

        else
        {
            cl::copy(queue, buf_vec2, std::begin(vec3), std::begin(vec3)+1);
        }

        // Stopping the clock
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

        auto time_difference_cpu = std::chrono::duration_cast<std::chrono::microseconds>(time_cpu_end-time_cpu_start).count()/1000.0;  
        auto time_difference_gpu = std::chrono::duration_cast<std::chrono::microseconds>(time_gpu_end-time_gpu_start).count()/1000.0;  

        std::cout << std::fixed << std::setprecision(10);

        std::cout << std::endl << "The CPU result is: " << cpu_result << ".";
        std::cout << std::endl << "The GPU result is: " << vec3[0] << ".";

        std::cout << std::fixed << std::setprecision(0) << std::endl;

        std::cout << std::endl << "The computational time for a " << size << " long vector dot product on the CPU: " << time_difference_cpu  << " milisec.";
        std::cout << std::endl << "The computational time for a " << size << " long vector dot product on the GPU: " << time_difference_gpu  << " milisec." << std::endl;

        float max_acc_diff = 0.01f;
        
        if (abs(vec3[0] - cpu_result) < max_acc_diff)
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
    catch (std::exception& error) // If STL/CRT error occursk
    {
        std::cerr << error.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

