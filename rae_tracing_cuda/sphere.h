#pragma once
#include "vec3.h"
#include "ray.h"

class sphere {
protected:
	vec3 position;
	float radius;
public:
	__host__ __device__ sphere() = default;
	__host__ __device__ sphere(vec3 position, float radius)
		: position{ position }, radius{ radius } {};
	__host__ __device__ vec3 pos() const {
		return this->position;
	};
	__host__ __device__ float r() const {
		return radius;
	};
	__host__ __device__ bool intersection(ray& r, float& t) const {
		vec3 rs = r.pos() - position;
		float b = vec3::dot(rs, r.dir());
		float c = vec3::dot(rs, rs) - radius * radius;
		float discriminant = b*b - c;
		if ( discriminant > 0) {
			float sqrtDisc = sqrt(discriminant);
			float t1 = b + sqrtDisc;
			float t2 = b - sqrtDisc;
			t = (t1 < t2) ? t1 : t2;
			return true;
		}
		return false;
	};

	__host__ __device__ bool getBool() {
		return true;
	};
};