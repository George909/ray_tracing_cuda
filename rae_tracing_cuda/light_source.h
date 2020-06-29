#pragma once

#include "vec3.h"

class light_source {
protected:
	vec3 _pos;
	float _i;
public:
	__host__ __device__ light_source() = default;
	__host__ __device__ light_source(vec3 pos, float i)
		: _pos{ pos }, _i{ i } {};
	
	__host__ __device__ vec3 pos() const { return this->_pos; }
	__host__ __device__ float i() const { return this->_i; }
	
	__host__ __device__ ~light_source() = default;
};