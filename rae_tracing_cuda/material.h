#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace material {

	enum class material_type {METAL, MIRROR, EARTH};

	class material {
	private:
		material_type type;
		float ambient;
		float diffuse;
		float specular;
		float reflection;
		int spec;

		__host__ __device__ void init(material_type t) {
			switch (t)
			{
			case material_type::EARTH:
				this->ambient = 0.9f;
				this->diffuse = 0.1f;
				this->specular = 0.1f;
				this->reflection = 0.0f;
				this->spec = 16;
				break;
			case material_type::MIRROR:
				this->ambient = 0.f;
				this->diffuse = 0.f;
				this->specular = 10.f;
				this->reflection = 0.8f;
				this->spec = 64;
				break;
			case material_type::METAL:
				this->ambient = 0.35f;
				this->diffuse = 0.3f;
				this->specular = 0.8f;
				this->reflection = 0.1f;
				this->spec = 128;
				break;
			default:
				break;
			}
		}

	public:
		__host__ __device__ material()
			: type{ material_type::METAL }
		{
			init(this->type);
		}
		__host__ __device__ material(material_type t)
			: type{ t }
		{
			init(t);
		}

		__host__ __device__ void set_ambient(float a) { this->ambient = a; };
		__host__ __device__ void set_diffuse(float d) { this->diffuse = d; };
		__host__ __device__ void set_specular(float s) { this->specular = s; };
		__host__ __device__ void set_reflection(float r) { this->reflection= r; };
		__host__ __device__ void set_spec(int s) { this->spec = s; };

		__host__ __device__ float get_ambient() const { return this->ambient; };
		__host__ __device__ float get_diffuse() const { return this->diffuse; };
		__host__ __device__ float get_specular() const { return this->specular; };
		__host__ __device__ float get_reflection() const { return this->reflection; };
		__host__ __device__ int get_spec() const { return this->spec; };

		__host__ __device__ ~material() = default;
	};
}