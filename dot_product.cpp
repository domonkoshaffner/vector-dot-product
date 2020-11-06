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


template<typename T>
std::vector<T> slice(std::vector<T> const &v, int m, int n)
{
	auto first = v.cbegin() + m;
	auto last = v.cbegin() + n;

	std::vector<T> vec(first, last);
	return vec;
}


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

        constexpr int size = 256;
        auto cpu_result = 0;
        int workgroupsize = 256;
        int zeros_to_fill;

        std::vector<int> vec1(size);
        std::vector<int> vec2(size);
        std::vector<int> result_vec(size);
        std::vector<int> result_vec2(size);
        std::vector<int> temp(size);

        // Filling up the vectors
        for (int i = 0; i < vec1.size(); ++i)
        {
            vec1[i] = i+1;
            vec2[i] = i+1;
        }

        // Checking the length of the vectors and filling them up with zeros
        // So their length is n*256 (workgroup size)

        zeros_to_fill = workgroupsize - (size % workgroupsize);
        std::vector<int> vec_zeros(zeros_to_fill, 0);

        vec1.insert( vec1.end(), vec_zeros.begin(), vec_zeros.end() );
        vec2.insert( vec2.end(), vec_zeros.begin(), vec_zeros.end() );

        int number_of_iterations = vec1.size()/workgroupsize;

// ############################################################################   
// Creating buffers, copying to the GPU, measuring time 

        // Creating buffers from the vectors
        cl::Buffer buf_vec1{ std::begin(vec1), std::end(vec1), true };
        cl::Buffer buf_vec2{ std::begin(vec2), std::end(vec2), true };
        cl::Buffer buf_result_vec {std::begin(result_vec), std::end(result_vec), false };
        cl::Buffer buf_result_vec2 {std::begin(result_vec2), std::end(result_vec2), false };
        cl::Buffer buf_temp {std::begin(temp), std::end(temp), false };

        // Explicit (blocking) dispatch of data before launch
        cl::copy(queue, std::begin(vec1), std::end(vec1), buf_vec1);
        cl::copy(queue, std::begin(vec2), std::end(vec2), buf_vec2);
        cl::copy(queue, std::begin(result_vec), std::end(result_vec), buf_result_vec);

        // Launch kernels
        dot_product(cl::EnqueueArgs{queue, cl::NDRange{size}}, buf_vec1, buf_vec2, buf_result_vec);

        cl::finish();

        // (Blocking) fetch of results
        cl::copy(queue, buf_result_vec, std::begin(result_vec), std::end(result_vec));

        cl::copy(queue, std::begin(result_vec), std::end(result_vec), buf_result_vec);
        //cl::copy(queue, std::begin(result_vec2), std::end(result_vec2), buf_result_vec2);
        //cl::copy(queue, std::begin(temp), std::end(temp), buf_temp);


// ############################################################################ 
// Launch kernels
/*
        for (int i = 0; i < workgroupsize; ++i)
        {
            std::vector<int> sub_vec = slice(result_vec, i*256, 256*(i+1));



        }
*/
        reduction(cl::EnqueueArgs{queue, cl::NDRange{size}, cl::NDRange{size}}, buf_result_vec, buf_result_vec2, cl::Local(size*sizeof(float)));

        cl::finish();

        // (Blocking) fetch of results
        cl::copy(queue, buf_result_vec2, std::begin(temp), std::end(temp));       



// ############################################################################ 
// Doing the same calculation on the CPU

        for (int i = 0; i < vec1.size(); ++i)
        {
            cpu_result += vec1[i]*vec2[i];
        }

// ############################################################################ 
// Printing the result vector

        std::cout << std::endl << "The CPU result is: " << cpu_result << ".";
        std::cout << std::endl << "The GPU result is: " << temp[0] << "." << std::endl; 
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
