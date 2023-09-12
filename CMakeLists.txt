cmake_minimum_required(VERSION 3.10)

project(mnist_cnn)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -march=native")

find_package(CUDA REQUIRED)
include_directories("${CUDA_INCLUDE_DIRS}")

include(CheckLanguage)
check_language(CUDA)

if (CMAKE_CUDA_COMPILER)
	message("[ACS LAB] CUDA is supported. Enabling CUDA sources.")
	enable_language(CUDA)
	add_definitions(-DUSE_CUDA)
	set(CMAKE_CUDA_STANDARD 11)

	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")
	set(CUDA_SOURCES
			src/cnn/global_ptr_cuda.cu
			src/Train.cu
			src/Infer.cu
			# ADD YOUR OWN CUDA SOURCES HERE
			)
else ()
	message("[ACS LAB] Could not find CUDA support. Disabling CUDA sources.")
endif ()

add_executable(${PROJECT_NAME}
		src/cnn/optimization_method.h
		src/cnn/point_t.h
		src/cnn/tensor_t.h
		src/utils/Timer.h
		src/main.cpp
		src/Infer.cu src/Infer.cuh src/utils/io.cpp src/utils/io.h
		${CUDA_SOURCES}
		)