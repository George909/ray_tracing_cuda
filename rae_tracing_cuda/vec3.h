#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cmath>

class vec3
{
private:
	float vec[3];

public:

	__host__ __device__ vec3() = default;
  __host__ __device__ vec3(float e0, float e1, float e2) { vec[0] = e0; vec[1] = e1; vec[2] = e2; }
  __host__ __device__ float x() const { return vec[0]; }
  __host__ __device__ float y() const { return vec[1]; }
  __host__ __device__ float z() const { return vec[2]; }

  __host__ __device__ const vec3& operator+() const { return *this; }
  __host__ __device__ vec3 operator-() const { return vec3(-vec[0], -vec[1], -vec[2]); }

  __host__ __device__ vec3 operator + (const vec3& vec) {
    return vec3(this->x() + vec.x(), this->y() + vec.y(), this->z() + vec.z());
  };
  __host__ __device__ vec3 operator - (const vec3& vec) {
    return vec3(this->x() - vec.x(), this->y() - vec.y(), this->z() - vec.z());
  };

  friend __host__ __device__ vec3 operator + (const vec3& vec1, const vec3& vec2) {
    return vec3(vec1.x() + vec2.x(), vec1.y() + vec2.y(), vec1.z() + vec2.z());
  };

  __host__ __device__ vec3 operator / (float t) {
    return vec3(vec[0] / t, vec[1] / t, vec[2] / t);
  };

  friend __host__ __device__ vec3 operator / (const vec3& vec, float t) {
    return vec3(vec.x() / t, vec.y() / t, vec.z() / t);
  };

  friend __host__ __device__ vec3 operator * (float t, const vec3& vec) {
    return vec3(vec.x() * t, vec.y() * t, vec.z() * t);
  };

  __host__ __device__ static float dot(const vec3& vec1, const vec3& vec2){
    return vec1.x() * vec2.x() + vec1.y() * vec2.y() + vec1.z() * vec2.z();
  };

  __host__ __device__ vec3 normalized() {
    float l = this->length();
    return  vec3(this->x()/l, this->y() / l, this->z() / l);
  };

  __host__ __device__ float length() const { 
    return std::sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]); 
  }
  __host__ __device__ float squared_length() const { 
    return vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]; 
  }
};

