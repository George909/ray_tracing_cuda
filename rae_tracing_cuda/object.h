#pragma once
#include <optional>

#include "vec3.h"
#include "ray.h"

class object {
public:
	__host__ __device__ virtual bool intersection(ray& ray, float& t) const = 0;
};
