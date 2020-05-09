#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vec3.h"
#include "ray.h"
class camera {
protected:
	vec3 position;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;	
public:

	__host__ __device__ camera(vec3 position, int size_x, int size_y)
		: position{ position }
	{
		lower_left_corner = vec3(-size_x / 2., -size_y / 2.f, -1);
		horizontal = vec3(size_x, 0, 0);
		vertical = vec3(0, size_y, 0);
	}
	__host__ __device__ ray getRay(float x, float y) const {
		return ray(position, lower_left_corner + x * horizontal + y * vertical);
	};
};