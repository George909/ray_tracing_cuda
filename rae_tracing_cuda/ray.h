#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vec3.h"

class ray {
private:
	vec3 position;
	vec3 direction;

public:
	__host__ __device__ ray() = default;
	__host__ __device__ ray(vec3 position, vec3 direction)
		: position{ position } {
		this->direction = direction.normalized();
	};
	
	__host__ __device__ vec3 pos() const {
		return this->position;
	};
	__host__ __device__ vec3 dir() const {
		return this->direction;
	};
	__host__ __device__ vec3 r(float t) const {
		return position + t * direction;
	}
};