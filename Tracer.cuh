#ifndef Cuda_Tracer_CU
#define Cuda_Tracer_CU

#include <iostream>
#include <math.h>
#include <limits>
#include <vector>
#include "cuda_runtime.h"
#include "curand_kernel.h"
#include <curand.h>
#include "GLFW/glfw3.h"

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

static void* fixed_cudaMalloc(size_t len);

__host__ __device__ float clamp(float x, float a, float b);


struct vec3 {
    float x, y, z;
    __host__ __device__ vec3(float x, float y, float z);
    __host__ __device__ vec3();
    __host__ __device__ vec3 operator+(const vec3 other)const;
    __host__ __device__ void operator+=(const vec3 other) { this->x += other.x; this->y += other.y; this->z += other.z; }
    __host__ __device__ vec3 operator-(const vec3 other) const;
    __host__ __device__ void operator-=(const vec3 other);
    __host__ __device__ vec3 operator*(const float scale) const;
    __host__ __device__ void operator*=(const float scale);
    __host__ __device__ float operator*(const vec3 other) const;  // dot product
    __host__ __device__ float magnitudeSquared() const;
    __host__ __device__ float magnitude() const;
    __host__ __device__ vec3 normalized() const;
    __host__ __device__ static float project(const vec3 w, const vec3 v);
    __device__ static vec3 randomInUnitSphere(curandState* state);
};

struct CudaColor {
    float r, g, b;
    int samples;
    __host__ __device__ CudaColor(float r, float g, float b);
    __host__ __device__ CudaColor(float r, float g, float b, int samples);
    __host__ __device__ CudaColor operator+(const CudaColor other) const;
    __host__ __device__ void operator+=(const CudaColor other);
    __host__ __device__ CudaColor operator-(const CudaColor other) const;
    __host__ __device__ void operator-=(const CudaColor other);
    __host__ __device__ CudaColor operator*(const float scale) const;
    __host__ __device__ bool operator==(const CudaColor other);
    __host__ __device__ CudaColor output();
    __host__ __device__ float4 floatOutput();
};

struct Ray {
    vec3 origin, direction;
    __host__ __device__ Ray(vec3 origin, vec3 direction);
    __device__ vec3 at(float distance) const;
};

struct Material {
    CudaColor color;
    __host__ __device__ Material();
    __host__ __device__ Material(CudaColor color);
};

struct HitRecord {
    float distance;
    vec3 position, normal;
    Material hitMaterial;
    __device__ HitRecord();
};

class Hittable {
public:
    vec3 location;
    Material material;
    __host__ __device__ Hittable(vec3 location, Material material);
    __device__ virtual void checkRay(const Ray& ray, HitRecord& record) = 0;
};

class CudaSphere{  // TODO should be derived ^^
public:
    vec3 location;
    float radius;
    Material material;
    __host__ __device__ CudaSphere(vec3 location, float radius);
    __host__ __device__ CudaSphere();
    __device__ void checkRay(Ray ray, HitRecord* record);
};

class World {
public:
    CudaSphere* deviceSpheres;
    size_t* deviceNumSpheres;
    CudaColor* backgroundColor;
    __host__ World(size_t numSpheres, CudaSphere* spheresToCopy, CudaColor* backgroundColorIn);
    __host__ World();
    __host__ void setSpheres(CudaSphere* spheresToCopy, size_t numSpheres);
    __host__ void setBackgroundColor(CudaColor* backgroundColorIn);
    __device__ void checkRay(Ray ray, HitRecord* record);
    __device__ __host__ ~World();
};

struct Camera {
    vec3 origin;
    float zoom;
    int width, height;
    Camera(vec3 origin, float zoom, int width, int height);
};

__device__ CudaColor shade(Ray ray, World world, int bouncesRemaining, curandState* state);

__global__  void initKernel(int width, int height, curandState *states, CudaColor *colorBuffer);

__global__ void renderFrameKernel(float4 *out_data, CudaColor *colorBuffer, curandState *states,
                                  World world, Camera camera, int samples, int bounceLimit,
                                  bool clearFrame);

class CudaTracer {
    struct cudaGraphicsResource* cudaFrameBuffer;
    CudaColor* cudaColorBuffer;
    curandState* curandStates;
    //GLuint openGLPixelBuffer;
    dim3 blocks, threads;
    World world;
    Camera camera;
    int numSamples, bounceLimit, numPixels, tileWidth, tileHeight;
public:
    CudaTracer(World world, Camera camera, GLuint openGLPixelBuffer);
    CudaTracer();
    void init();
    void setWorld(World wold);
    void setCamera(Camera camera);
    Camera getCamera();
    int getWidth() {return camera.width;}
    int getHeight() {return camera.height;}
    void setGLPixelBuffer(GLuint openGLPixelBuffer);
    void setSamples(int samples);
    void renderFrame(bool clearFrame);
    ~CudaTracer();
};

#endif