
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "easyBMP/EasyBMP.h"
#include "vec3.h"
#include "ray.h"
#include "object.h"
#include "sphere.h"
#include "camera.h"

#include <stdio.h>
#include <string>
#include <vector>

int gcd(int a, int b) {
  return b ? gcd(b, a % b) : a;
}
cudaError_t ray_tracing_gpu(vec3* pixels, sphere** sphere, int x, int y, int number_of_objects, camera cam);
//void ray_tracing_cpu(vec3* pixels, sphere* sph, int x, int y, int number_of_objects, camera cam);

__device__ __host__ vec3 color(ray r, sphere** sph, int number_of_objects) {
  float t = 0;
  for (int k = 0; k < number_of_objects; k++) {
    if (sph[k]->intersection(r, t))
      return vec3(0, 0, 1);
  }
  return vec3(1, 1, 0);
}

__global__ void render(vec3* pixels, sphere** sph, int size_x, int size_y, int number_of_objects, camera* cam)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i > size_x || j > size_y) return;
  float h = i / static_cast<float>(size_x);
  float v = j / static_cast<float>(size_y);
  
  //pixels[size_x * j + i] = vec3(0, 1, 0);
  //ray r = cam->getRay(i, j);
  pixels[size_x * j + i] = color(cam->getRay(h, v), sph, number_of_objects);
}

int main(int argc, char** argv)
{
  if (argc < 4) return -1;
  std::string number = "0123456789";
  std::string resolution = argv[3];

  int number_of_objects = std::atoi(argv[1]);
  int number_of_juches = std::atoi(argv[2]);
  int x_size = std::atoi(resolution.c_str());
  int index_x = resolution.find("x");
  int y_size = std::atoi(&resolution.c_str()[index_x + 1]);
  std::string filename = argv[4];

  //std::cout << number_of_spheres << std::endl;
  //std::cout << number_of_juches << std::endl;
  //std::cout << x_size << std::endl;
  //std::cout << y_size << std::endl;
  //std::cout << filename << std::endl;

  BMP outImage;
  outImage.SetSize(x_size, y_size);

  int coef = gcd(x_size, y_size);
  std::vector<vec3> colors(x_size * y_size);
  camera cam(vec3(0,0,0), x_size/coef, y_size/coef);
  std::vector<sphere> spheres;
  spheres.push_back(sphere({ 0,0,-2 }, 0.5));

  sphere** spheres1 = new sphere*[number_of_objects];
  spheres1[0] = new sphere({ 0,0,-2 }, 0.5);

  //ray_tracing_cpu(colors.data(), &sph, x_size, y_size, number_of_objects, cam);

  std::cout << "correct" << std::endl;

  cudaError_t cudaStatus = ray_tracing_gpu(colors.data(), spheres1, x_size, y_size, number_of_objects, cam);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "Ray Tracing failed!");
    return 1;
  }

  for (int i = 0; i < x_size; i++) {
    for (int j = 0; j < y_size; j++) {
      outImage(i, j)->Red = colors.at(j * x_size + i).x() * 255;
      outImage(i, j)->Green = colors.at(j * x_size + i).y() * 255;
      outImage(i, j)->Blue = colors.at(j * x_size + i).z() * 255;
    }
  }

  outImage.WriteToFile(filename.c_str());

  // cudaDeviceReset must be called before exiting in order for profiling and
  // tracing tools such as Nsight and Visual Profiler to show complete traces.
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
    return 1;
  }

  return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t ray_tracing_gpu(vec3* pixels, sphere** sph, int x, int y, int number_of_objects, camera cam)
{
  vec3* dev_pixels;
  sphere** dev_sph = new sphere*[number_of_objects];
  camera* dev_cam;
  cudaError_t cudaStatus;

  // Choose which GPU to run on, change this on a multi-GPU system.
  cudaStatus = cudaSetDevice(0);
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
      goto Error;
  }

  cudaStatus = cudaMalloc((void**)&dev_pixels, x * y * sizeof(vec3));
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed!");
      goto Error;
  }

  for (int i = 0; i < number_of_objects; i++) {
    cudaStatus = cudaMalloc((void**)&(dev_sph[i]),sizeof(sphere));
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed!");
      goto Error;
    }
    cudaStatus = cudaMemcpy(dev_sph[i], sph[i],sizeof(sphere), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed!");
      goto Error;
    }
  }
    
  cudaStatus = cudaMalloc((void**)&dev_cam, sizeof(camera));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  
  cudaStatus = cudaMemcpy(dev_cam, &cam, sizeof(camera), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error;
  }

  dim3 blocks(x / 8 + 1, y / 8 + 1);
  dim3 threads(8, 8);
  render<<<blocks, threads>>>(dev_pixels, dev_sph, x, y, number_of_objects, dev_cam);

  // Check for any errors launching the kernel
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
      goto Error;
  }
    
  // cudaDeviceSynchronize waits for the kernel to finish, and returns
  // any errors encountered during the launch.
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
      goto Error;
  }

  // Copy output vector from GPU buffer to host memory.
  cudaStatus = cudaMemcpy(pixels, dev_pixels, x * y * sizeof(vec3), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed!");
      goto Error;
  }

Error:
  cudaFree(dev_pixels);
  cudaFree(dev_sph);
  cudaFree(dev_cam);

  return cudaStatus;
}

//void ray_tracing_cpu(vec3* pixels, sphere* sph, int x, int y, int number_of_objects, camera cam){
//
//  for (int i = 0; i < x; i++){
//    for (int j = 0; j < y; j++) {
//      pixels[x * j + i] = color(cam.getRay(i, j), sph, number_of_objects);
//    }
//  }
//}
