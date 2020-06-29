#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vec3.h"
#include "ray.h"
class camera {
protected:
	vec3 position; // &

	float aspect_ratio;
	float alpha;
	float height;
	float width;
public:

	__host__ __device__ camera(vec3 position, float size_x, float size_y)
		: position { position }
		, alpha {45.f}
		, height {size_y}
		, width { size_x}  
	{
		this->aspect_ratio = width / height;
	}
	__host__ __device__ ray getRay(float x, float y) const {

		float coef = std::tan(alpha / 360 * 3.14f);
		float ccs_x = (2 * x - 1) * this->aspect_ratio * coef;
		float ccs_y = (1 - 2 * y) * coef;
		vec3 dir = {ccs_x, ccs_y, -1};
		dir = dir.normalized();

		return ray(position, dir, 0);
	};
};