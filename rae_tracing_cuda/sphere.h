#pragma once
#include "vec3.h"
#include "ray.h"
#include "material.h"

class sphere {
protected:
	vec3 _pos;
	float _r;
	material::material _m;
	vec3 _color;	

public:
	__host__ __device__ sphere() = default;
	__host__ __device__ sphere(vec3 pos, float r, vec3 color, float refraction_index, float spec)
		: _pos{ pos }
		, _r{ r }
		, _m{ material::material()}
	  , _color{ color } {
	};
	__host__ __device__ vec3 pos() const { return this->_pos; };
	__host__ __device__ float r() const { return _r; };
	__host__ __device__ vec3 color() const { return this->_color; };
	__host__ __device__ material::material material() const { return this->_m; };


	__host__ __device__ void set_pos(vec3 pos) { _pos = pos; };
	__host__ __device__ void set_r(float r) { _r = r; };
	__host__ __device__ void set_color(vec3 color) {_color = color; };
	__host__ __device__ void set_material(material::material_type t) { _m = material::material(t); };

	__host__ __device__ bool intersection(ray& r, float& t) const {	
		vec3 rs = _pos - r.pos();
		float b = vec3::dot(rs, r.dir());
		float c = vec3::dot(rs, rs) - _r * _r;
		float discriminant = b*b - c;
		if( discriminant > 0) {
			float sqrt_disc = sqrt(discriminant);
			float t1 = b + sqrt_disc;
			float t2 = b - sqrt_disc;
			t = t2;
			if (t < 0) t = t1;
			if (t < 0) return false;
			return true;
		}
		return false;
	};
	__host__ __device__ vec3 normal(vec3 point) const {
		return vec3(point.x() - _pos.x(), point.y() - _pos.y(), point.z() - _pos.z()).normalized();
	}
	
	__host__ __device__ ray reflectedRay(ray& r0) const {
		vec3 n = this->normal(r0.r());
		vec3 dir = r0.dir() - 2.f * vec3::dot(r0.dir(), n) * n;
		vec3 pos = vec3::dot(dir, n) < 0 ? r0.r() - 1e-3 * n : r0.r() + 1e-3 * n;
 		return ray(pos, dir, 0);
	}

	__host__ __device__ vec3 reflectedDir(vec3 &dir, vec3 &point) const {
		vec3 n = this->normal(point);
		vec3 direction = dir - 2.f * vec3::dot(dir, n) * n;
		return direction;
	}

};