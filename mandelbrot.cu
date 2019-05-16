#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cuda.h>
#include <sys/time.h>

#include "colors.h"

#define MAX_ITERATIONS 65536
#define THREADS_PER_BLOCK 256

__device__ float autoPow(float x, float y)
{
	return powf(x, y);
}

__device__ double autoPow(double x, double y)
{
	return pow(x, y);
}

template<typename T>
__global__ void mandelbrotKernel(const int depthStart,
				const int depthEnd,
				const T zoomFactor,
				const int width,
				const int height,
				unsigned* const intensities)
{
	// Calculate the global thread index and exit if there is no work to do
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	const int pz = depthStart + idx / (height * width);
	const int py = (idx / width) % height;
	const int px = idx % width;

	if (idx >= (depthEnd - depthStart) * height * width) return;

	// Parameters for a really cool part that has great depth
	const T xCenter = static_cast<T>(-0.235125001);
	const T yCenter = static_cast<T>(0.827215);
	//const T xCenter = static_cast<T>(-0.598274455069517539539);
	//const T yCenter = static_cast<T>(0.663825928894102918143);
	const T baseVerticalRadius = static_cast<T>(1.0);//(0.00004);

	// This controls how much each depth zooms in, bounded by (0,1]. Lower values zoom faster.
	//const T zoomFactor = static_cast<T>(0.9);
	//const T zoomFactor = static_cast<T>(0.95);

	// Compute mandelbrot set
	const T yRadius = baseVerticalRadius * autoPow(zoomFactor, pz);
	const T xRadius = yRadius * width / height;

	const T y0 = yCenter + yRadius - static_cast<T>(2.0) * yRadius * py / (height - 1);
	const T x0 = xCenter - xRadius + static_cast<T>(2.0) * xRadius * px / (width - 1);

	T x = x0;
	T y = y0;
	T x2 = x * x;
	T y2 = y * y;
	unsigned iteration = 0;

	while (x2 + y2 <= static_cast<T>(4.0) && iteration < MAX_ITERATIONS)
	{
		y = static_cast<T>(2.0) * x * y + y0;
		x = x2 - y2 + x0;
		x2 = x * x;
		y2 = y * y;
		++iteration;
	}

	// Store intensity value for determining color later
	intensities[idx] = iteration;
}

template __global__ void mandelbrotKernel<float>(const int, const int, const float, const int, const int, unsigned* const);
template __global__ void mandelbrotKernel<double>(const int, const int, const double, const int, const int, unsigned* const);



void savePpmImage(const char* const filename, const int width, const int height, const unsigned* const intensities)
{
	// Open the file
	std::ofstream outfile(filename);

	// Write the header
	//outfile << "P3 " << width << " " << height << " " << MAX_ITERATIONS / 2 << "\n";
	outfile << "P3 " << width << " " << height << " " << 255 << "\n";

	// Write pixel information
	for (int y = 0; y < height; ++y)
	{
		const int yidx = y * width;

		for (int x = 0; x < width; ++x)
		{
			//const std::array<unsigned, 3> color = mapIntensityToColor(intensities[yidx + x]);
			//outfile << color[0] << " " << color[1] << " " << color[2] << " ";
			const unsigned i = intensities[yidx + x] % 256;
			outfile << djl70::paletteRed[i] << " " << djl70::paletteGreen[i] << " " << djl70::paletteBlue[i] << " ";
		}

		outfile << "\n";
	}

	outfile.close();
}



void printUsage(const char* const programName)
{
	std::cerr << "Usage: " << programName << " <depths> <width> [save_output [depth_start [zoom_factor]]]\n"
		<< "Description: Computes the Mandelbrot set using CUDA\n"
		<< "Argument combinations worth trying:\n"
		<< "    " << programName << " 256 120 1\n"
		<< "    " << programName << " 16 600 1\n"
		<< "    " << programName << " 64 2400 0\n"
		<< "    " << programName << " 64 2400 0 64\n"
		<< "    " << programName << " 512 600 0 0 0.99\n"
		<< "\n"
		<< "depths:      The number of 'layers' to process (each layer 'zooms in' to the set)\n"
		<< "width:       The width of the images to process (height is set automatically)\n"
		<< "save_output: (optional, default 0) Set to 1 to save the processed images (note: not recommended beyond depths=64 and width=600, because otherwise saving the images may require a lot of time and disk space)\n"
		<< "depth_start: (optional, default 0) The zero-based 'layer' to begin processing at, inclusive\n"
		<< "zoom_factor: (optional, default 0.9) A value (0 < zoom_factor <= 1) deciding how much to zoom for each 'layer'. Higher values produce slower zooms"
		<< std::endl;
}

int main(int argc, char* argv[])
{
	// Verify command line arguments
	if (argc < 3)
	{
		printUsage(argv[0]);
		return -1;
	}
	const int depths = atoi(argv[1]);
	if (depths < 1)
	{
		std::cerr << "Error: arg 'depths' must be at least 1" << std::endl;
		return -1;
	}
	const int width = atoi(argv[2]);
	if (width < 2)
	{
		std::cerr << "Error: arg 'width' must be at least 2" << std::endl;
		return -1;
	}
	const int height = (float)width * 2.0f / 3.0f;
	const bool doSaveImages = (argc >= 4) && (atoi(argv[3]) == 1);
	const int depthStart = (argc >= 5) ? atoi(argv[4]) : 0;
	if (depthStart < 0)
	{
		std::cerr << "Error: arg 'depth_start' must be at least 0" << std::endl;
		return -1;
	}
	const int depthEnd = depthStart + depths;
	const double zoomFactor = (argc >= 6) ? atof(argv[5]) : 0.9;
	std::cout << "'depths' = " << depths
		<< "\n'width' = " << width
		<< "\n'height' = " << height
		<< "\n'save_output' = " << doSaveImages
		<< "\n'depth_start' = " << depthStart << " (inclusive)"
		<< "\n'depth_end' = " << depthEnd << " (exclusive)"
		<< "\n'zoom_factor' = " << zoomFactor
		<< std::endl;

	// Allocate host memory
	const int n = depths * height * width;
	unsigned* h_intensities = new unsigned[n];

	// Allocate device memory
	unsigned* d_intensities;
	if (cudaSuccess != cudaMalloc((void**)&d_intensities, sizeof(unsigned) * n))
	{
		delete[] h_intensities;
		std::cerr << "Error: failed to allocate device memory" << std::endl;
		return -1;
	}

	// Launch GPU kernel and begin timing
	timeval start, end;
	if (doSaveImages || depthEnd > 128)
	{
		std::cout << "'depth_end' > 128 or 'save_output' = 1, launching kernel with double precision to ensure high quality output" << std::endl;
		gettimeofday(&start, NULL);
		mandelbrotKernel<double><<<(n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(depthStart, depthEnd, zoomFactor, width, height, d_intensities);
	}
	else
	{
		std::cout << "'depth_end' <= 128 and 'save_output' = 0, launching kernel with single precision to ensure the best performance" << std::endl;
		gettimeofday(&start, NULL);
		mandelbrotKernel<float><<<(n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(depthStart, depthEnd, zoomFactor, width, height, d_intensities);
	}
	cudaDeviceSynchronize();

	// End timing
	gettimeofday(&end, NULL);
	const double runtime = end.tv_sec - start.tv_sec + (end.tv_usec - start.tv_usec) / 1000000.0;
	std::cout << "Kernel runtime: " << std::fixed << std::setprecision(4) << runtime << " s" << std::endl;

	// Check for errors from the kernel
	cudaError_t e = cudaGetLastError();
	if (cudaSuccess != e)
	{
		delete[] h_intensities;
		cudaFree(d_intensities);
		std::cerr << "CUDA error " << e << ": " << cudaGetErrorString(e) << std::endl;
		return -1;
	}

	// Save images if desired
	if (doSaveImages)
	{
		// Copy results to the host
		if (cudaSuccess != cudaMemcpy(h_intensities, d_intensities, sizeof(unsigned) * n, cudaMemcpyDeviceToHost))
		{
			delete[] h_intensities;
			cudaFree(d_intensities);
			std::cerr << "Error: failed to copy from device to host" << std::endl;
			return -1;
		}

		std::cout << "Saving output images, please wait..." << std::endl;
		for (int i = depthStart; i < depthEnd; ++i)
		{
			std::stringstream filename;
			filename << "mandelbrot" << std::setw(3) << std::setfill('0') << i << std::setfill(' ') << ".ppm";
			savePpmImage(filename.str().c_str(), width, height, &h_intensities[(i - depthStart) * height * width]);
		}
		std::cout << "Output images successfully saved as .ppm files\n"
			<< "Try running 'convert -delay 10 mandelbrot*ppm mandelbrot.gif' to create an animation"
			<< std::endl;
	}

	// Free memory
	delete[] h_intensities;
	cudaFree(d_intensities);

	return 0;
}
