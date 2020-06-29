#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "vec3.h"

class ray {
private:
	vec3 _pos;
	vec3 _dir;
	float _t;

public:
	__host__ __device__ ray() = default;
	__host__ __device__ ray(vec3 position, vec3 direction, float t)
		: _pos{ position }
		, _t{ t } {
		this->_dir = direction.normalized();
	};
	
	__host__ __device__ void set_pos(vec3 &pos) { _pos = pos; };
	__host__ __device__ void set_dir(vec3 &dir) { _dir = dir; };
	__host__ __device__ void set_t(float &t) { _t = t; };

	__host__ __device__ vec3 pos() const {	return _pos; };
	__host__ __device__ vec3 dir() const { return _dir; };
	__host__ __device__ float t() const { return _t; };
	
	__host__ __device__ vec3 r(float &t) const { return _pos + t * _dir; }
	__host__ __device__ vec3 r() const { return _pos + _t * _dir; }
};