#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "easyBMP/EasyBMP.h"
#include "vec3.h"
#include "ray.h"
#include "object.h"
#include "sphere.h"
#include "camera.h"
#include "light_source.h"
#include "material.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <random>
#include <ctime>
#include <chrono>


int gcd(int a, int b);
float rand_float(float min, float max);
cudaError_t ray_tracing_gpu(vec3* pixels, sphere* sphere,light_source* src,int x, int y, int number_of_objects,int number_of_sources,camera cam);
void ray_tracing_cpu(vec3* pixels,sphere* sphere,light_source* src,int x, int y,int number_of_objects,int number_of_sources,camera cam);
__device__ __host__ vec3 color(ray r,sphere* sph,light_source* src,int number_of_objects,int number_of_sources,int depth); 
__global__ void render(vec3* pixels,sphere* sph,light_source* src,int size_x, int size_y,int number_of_objects,int number_of_sources,camera* cam);
__global__ void create_scene_gpu(sphere* sph, light_source* src, int number_of_objects, int number_of_sources);
void create_scene_cpu(sphere* sph, light_source* src, int number_of_objects, int number_of_sources);

int main(int argc, char** argv)
{
  if (argc < 4) return -1;
  
  int number_of_objects = std::atoi(argv[1]);
  int number_of_juches = std::atoi(argv[2]);
  std::string resolution = argv[3];
  int x_size = std::atoi(resolution.c_str());
  int index_x = resolution.find("x");
  int y_size = std::atoi(&resolution.c_str()[index_x + 1]);
  std::string filename = argv[4];

  BMP outImage;
  BMP cpu_out_image;
  outImage.SetSize(x_size, y_size);
  cpu_out_image.SetSize(x_size, y_size);

  float coef = gcd(x_size, y_size);
  std::vector<vec3> colors(x_size * y_size);
  std::vector<vec3> cpu_colors(x_size * y_size);
  camera cam(vec3(0,0,1), x_size/coef, y_size/coef);
  sphere* spheres = new sphere[number_of_objects]; 
  light_source* sources = new light_source[number_of_juches];

  create_scene_cpu(spheres, sources, number_of_objects, number_of_juches);


  auto start = std::chrono::steady_clock::now();
  cudaError_t cudaStatus = ray_tracing_gpu(colors.data(), spheres, sources, x_size, y_size, number_of_objects, number_of_juches, cam);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "Ray Tracing failed!");
    return 1;
  }
  auto end = std::chrono::steady_clock::now();
  auto time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  
  start = std::chrono::steady_clock::now();
  ray_tracing_cpu(cpu_colors.data(), spheres, sources, x_size, y_size, number_of_objects, number_of_juches, cam);
  end = std::chrono::steady_clock::now();
  auto time_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "Time CPU: " << time_cpu.count() << " ms" << std::endl;
  std::cout << "Time GPU: " << time_gpu.count() << " ms" << std::endl;

  for (int i = 0; i < x_size; i++) {
    for (int j = 0; j < y_size; j++) {

      vec3 c = colors.at(j * x_size + i);
      if (c.x() > 1 || c.y() > 1 || c.z() > 1) {
        c = c / std::max(c.x(), std::max(c.y(), c.z()));
      }

      outImage(i, j)->Red = c.x() * 255;
      outImage(i, j)->Green = c.y() * 255;
      outImage(i, j)->Blue = c.z() * 255;
    }
  }

  for (int i =  0; i < x_size ; i++) {
    for (int j = 0; j < y_size; j++) {
      
      vec3 c = cpu_colors.at(j * x_size + i);
      if (c.x() > 1 || c.y() > 1 || c.z() > 1) {
        c = c / std::max(c.x(), std::max(c.y(), c.z()));
      }

      cpu_out_image(i, j)->Red = c.x() * 255;
      cpu_out_image(i, j)->Green = c.y() * 255;
      cpu_out_image(i, j)->Blue = c.z() * 255;
    }
  }

  outImage.WriteToFile((filename + ".BMP").c_str());
  cpu_out_image.WriteToFile((filename + "_cpu" + ".BMP").c_str());

 /*  cudaDeviceReset must be called before exiting in order for profiling and
   tracing tools such as Nsight and Visual Profiler to show complete traces.
 */ 
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaDeviceReset failed!");
    return 1;
  }

  return 0;
}

cudaError_t ray_tracing_gpu(vec3* pixels,sphere* sph,light_source* src,int x, int y,int number_of_objects,int number_of_sources,camera cam)
{
  vec3* dev_pixels;
  sphere* dev_sph;
  light_source* dev_src;
  camera* dev_cam;
  cudaError_t cudaStatus;

  cudaStatus = cudaMalloc((void**)&dev_pixels, x * y * sizeof(vec3));
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed!");
      goto Error;
  }

  cudaStatus = cudaMalloc((void**)&dev_sph, number_of_objects * sizeof(sphere));
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMalloc failed!");
      goto Error;
  }
  cudaStatus = cudaMemcpy(dev_sph, sph, number_of_objects * sizeof(sphere), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed!");
      goto Error;
  }
 
  cudaStatus = cudaMalloc((void**)&dev_src, number_of_sources * sizeof(light_source));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMalloc failed!");
    goto Error;
  }
  cudaStatus = cudaMemcpy(dev_src, src, number_of_sources * sizeof(light_source), cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "cudaMemcpy failed!");
    goto Error;
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
  render<<<blocks, threads>>>(dev_pixels, dev_sph, dev_src, x, y, number_of_objects, number_of_sources, dev_cam);

  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
      goto Error;
  }
    
  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
      goto Error;
  }

  cudaStatus = cudaMemcpy(pixels, dev_pixels, x * y * sizeof(vec3), cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaMemcpy failed!");
      goto Error;
  }

Error:
  cudaFree(dev_pixels);
  cudaFree(dev_sph);
  cudaFree(dev_cam);
  cudaFree(dev_src);

  return cudaStatus;
}

int gcd(int a, int b) {
  return b ? gcd(b, a % b) : a;
}

float rand_float(float min, float max) {
  float fraction = 1.f / (static_cast<float>(RAND_MAX) + 1.f);
  return static_cast<float>(rand() * fraction * (max - min + 1) + min);
}

void ray_tracing_cpu(vec3* pixels, sphere* sphere, light_source* src, int x, int y, int number_of_objects, int number_of_sources, camera cam)
{
    for (int i = 0; i < x; i++) {
        for (int j = 0; j < y; j++) {
            float h = (i + 0.5f) / static_cast<float>(x);
            float v = (j + 0.5f) / static_cast<float>(y);
            pixels[x * j + i] = color(cam.getRay(h, v), sphere, src, number_of_objects, number_of_sources, 5);
        }
    }
}

__device__ __host__ vec3 color(ray r,sphere* sph,light_source* src,int number_of_objects,int number_of_sources,int depth) {
  ray current_ray = r;
  vec3 result = { 0,0,0 };
  float frac = 1.0f;

  for (int d = 0; d <= depth; d++) {

    float min_t = std::numeric_limits<float>::max();
    float t = 0;
    sphere* s = nullptr;
    vec3 point;

    for (int i = 0; i < number_of_objects; i++) {
      if (sph[i].intersection(current_ray, t)) {
        if (t < min_t) {
          min_t = t;
          s = &(sph[i]);
          current_ray.set_t(t);
          point = current_ray.r(t);
        }
      }
    }

    if (s == nullptr || d == depth) {
      float coef = 0.5f * (r.dir().y() + 1.0f);
      result = result + frac * ((1.0f - coef) * vec3(1.0, 1.0, 1.0) + coef * vec3(0.5, 0.7, 1.0));
      break;
    }

    float diffuse_light_intensity = 0.f;
    float specular_light_intensity = 0.f;
    for (size_t i = 0; i < number_of_sources; i++) {
      vec3 normal = s->normal(point);
      vec3 light_dir = (src[i].pos() - point).normalized();
     
      float light_distance = (src[i].pos() - point).length();
      vec3 shadow_pos = (vec3::dot(light_dir, normal) < 0) ? point - 1e-3 * normal : point + 1e-3 * normal;
      ray shadow_ray = ray(shadow_pos, light_dir, 0);
      bool intersect = false;
      for (int j = 0; j < number_of_objects; j++) {
        if (sph[j].intersection(shadow_ray, t) && (shadow_ray.r(t) - shadow_pos).length() < light_distance) {
          intersect = true;
          break;
        }
      }
      if (intersect == true) continue;

      float dot = vec3::dot(light_dir, s->normal(point));
      float max = (dot > 0) ? dot : 0;
      diffuse_light_intensity += src[i].i() * max;

      vec3 raflection_light_dir = s->reflectedDir(light_dir, point);
      dot = vec3::dot(raflection_light_dir, current_ray.dir());
      max = (dot > 0) ? dot : 0;
      specular_light_intensity += src[i].i() * pow(max, static_cast<float>(s->material().get_spec()));
    }

    result = result + frac * (s->material().get_diffuse() * diffuse_light_intensity * s->color()
      + s->material().get_specular() * specular_light_intensity * vec3(1, 1, 1));

    frac *= s->material().get_reflection();
    current_ray = s->reflectedRay(current_ray);
  }

  return result;
}

__global__ void render(vec3* pixels,sphere* sph,light_source* src,int size_x, int size_y,int number_of_objects,int number_of_sources,camera* cam)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i > size_x || j > size_y) return;
  float h = (i + 0.5f) / static_cast<float>(size_x);
  float v = (j + 0.5f) / static_cast<float>(size_y);

  pixels[size_x * j + i] = color(cam->getRay(h, v), sph, src, number_of_objects, number_of_sources, 5);
}

void create_scene_cpu(sphere* sph, light_source* src, int number_of_objects, int number_of_sources) {
  
  srand(static_cast<unsigned int>(time(0)));

  float min_z{-9}, max_z{-20};
  float z{}, x{}, y{}, radius{};
  float r{}, g{}, b{};
  float intensity{};
  vec3 pos{};

  for (int i = 0; i < number_of_objects; i++) {
    z = rand_float(min_z, max_z);
    y = rand_float(z * tan(45. / 2. / 180. * 3.14), -z * tan(45. / 2. / 180. * 3.14));
    x = rand_float(z * tan(45. / 2. / 180. * 3.14), -z * tan(45. / 2. / 180. * 3.14));
    radius = rand_float(0.5, 1);
    r = rand_float(0, 1);
    g = rand_float(0, 1);
    b = rand_float(0, 1);

    sph[i] = sphere({ x,y,z }, radius, { r,g,b }, 0, 32);
  }

  for (int i = 0; i < number_of_sources; i++) {
    while (true) {
      z = rand_float(min_z, max_z);
      y = rand_float(z * tan(45. / 2. / 180. * 3.14), -z * tan(45. / 2. / 180. * 3.14));
      x = rand_float(z * tan(45. / 2. / 180. * 3.14), -z * tan(45. / 2. / 180. * 3.14));
      intensity = rand_float(0, 1);
      pos = { x,y,z };

      bool t = false;
      for (int j = 0; j < number_of_objects; j++) {
        float l = (pos - sph[j].pos()).length();
        if (l <= sph[i].r()) {
          break;
        }
        t = true;
      }
      if (t) break;
    }

    src[i] = light_source({ x,y,z }, intensity);
  }
}

__global__ void create_scene_gpu(sphere* sph, light_source* src, int number_of_objects, int number_of_sources) {

}