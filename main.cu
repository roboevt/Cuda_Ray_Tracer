
#include <iostream>
#include <fstream>
#include <math.h>
#include <limits>
#include <vector>
#include <memory>
#include <thrust/device_vector.h>
#include "cuda_runtime.h"
#include <curand_kernel.h>
#include <curand.h>
#include "device_launch_parameters.h"
#define GL_GLEXT_PROTOTYPES
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>

__constant__ float maxDistance = 3.40282346639e+38f;

using namespace std;

__constant__ constexpr int width = 1920;
__constant__ constexpr int height = 1080;
__constant__ constexpr int samples = 100;
__constant__ constexpr int bounceLimit = 5;
__constant__ constexpr int tileHeight = 16;
__constant__ constexpr int tileWidth = 16;
__constant__ constexpr int numPixels = width * height;
__constant__ constexpr int stackSize = 1 << 10;  // 64kb

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)


void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        std::cerr << cudaGetErrorString(result);
        cudaDeviceReset();
        exit(99);
    }
}

__host__ __device__ float clamp(float x, float a, float b) {
    return max(a, min(b, x));
}

struct vec3 {
    float x, y, z;
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3 operator+(const vec3 other) const { return vec3(this->x + other.x, this->y + other.y, this->z + other.z); }
    __host__ __device__ void operator+=(const vec3 other) { this->x += other.x; this->y += other.y; this->z += other.z; }
    __host__ __device__ vec3 operator-(const vec3 other) const { return vec3(this->x - other.x, this->y - other.y, this->z - other.z); }
    __host__ __device__ void operator-=(const vec3 other) { this->x -= other.x; this->y -= other.y; this->z -= other.z; }
    __host__ __device__ vec3 operator*(const float scale) const { return vec3(this->x * scale, this->y * scale, this->z * scale); };
    __host__ __device__ void operator*=(const float scale) { this->x *= scale; this->y *= scale; this->z *= scale; };
    __host__ __device__ float operator*(const vec3 other) const { return this->x * other.x + this->y * other.y + this->z * other.z; }  // dot product
    __host__ __device__ float magnitudeSquared() const { return this->x * this->x + this->y * this->y + this->z * this->z;}
    __host__ __device__ float magnitude() const { return sqrt(this->magnitudeSquared()); }
    __host__ __device__ vec3 normalized() const { return (*this) * (1.0f / this->magnitude()); }
};

struct Color {
    float r, g, b;
    int samples;
    __host__ __device__ Color(float r, float g, float b) : r(r), g(g), b(b), samples(0) {}
    __host__ __device__ Color(float r, float g, float b, int samples) : r(r), g(g), b(b), samples(samples) {}
    __host__ __device__ Color operator+(const Color other) const { return Color(this->r + other.r, this->g + other.g, this->b + other.b, this->samples + other.samples + 1); }
    __host__ __device__ void operator+=(const Color other) { this->r += other.r; this->g += other.g; this->b += other.b; this->samples += other.samples + 1; }
    __host__ __device__ Color operator-(const Color other) const { return Color(this->r - other.r, this->g - other.g, this->b - other.b, this->samples - other.samples + 1); }
    __host__ __device__ void operator-=(const Color other) { this->r -= other.r; this->g -= other.g; this->b -= other.b; this->samples -= other.samples + 1; }
    __host__ __device__ Color operator*(const float scale) const { return Color(this->r * scale, this->g * scale, this->b * scale, this->samples); }
    __host__ __device__ Color output() { return Color(clamp(this->r / this->samples, 0.0f, 1.0f), clamp(this->g / this->samples, 0.0f, 1.0f), clamp(this->b / this->samples, 0.0f, 1.0f)); }
    __host__ __device__ float4 floatOutput() {
        Color outputCol = this->output();
        return make_float4(outputCol.r, outputCol.g, outputCol.b, 1.0f);
    }
};

__device__ vec3 randomOnUnitSphere(curandState* state) {
    vec3 temp = vec3(curand_uniform(state), curand_uniform(state), curand_uniform(state));
    while (temp.magnitudeSquared() >= 1) {
        float newX = (curand_uniform(state) * 2.0f) - 1.0f;
        float newY = (curand_uniform(state) * 2.0f) - 1.0f;
        float newZ = (curand_uniform(state) * 2.0f) - 1.0f;
        temp = vec3(newX, newY, newZ);
    }
    return temp.normalized();
}

struct Ray {
    vec3 origin, direction;
    __host__ __device__ Ray(vec3 origin, vec3 direction) : origin(origin), direction(direction) {}
    __device__ vec3 at(float distance) const { return this->origin + (this->direction.normalized() * distance); }
};

struct Material {
    Color color;
    __host__ __device__ Material() : color(Color(0, 0, 0)) {}
    __host__ __device__ Material(Color color) : color(color) {}
};

struct HitRecord {
    float distance;
    vec3 position, normal;
    Material hitMaterial;
    __device__ HitRecord() : distance(maxDistance), position(vec3(0, 0, 0)), normal(vec3(0, 0, 0)), hitMaterial(Material()) {}
};

class Hittable {
public:
    vec3 location;
    Material material;
    __host__ __device__ Hittable(vec3 location, Material material) : location(location), material(material) {}
    __device__ virtual void checkRay(const Ray& ray, HitRecord& record) = 0;
};

class Sphere{  // TODO should be derived ^^
public:
    vec3 location;
    float radius;
    Material material;
    __host__ __device__ Sphere(vec3 location, float radius) : location(location), material(material), radius(radius) {}
    __device__ void checkRay(const Ray& ray, HitRecord& record) {
        vec3 oc = ray.origin - this->location;
        float a = ray.direction * ray.direction;
        float b = oc * ray.direction;
        float c = oc * oc - this->radius * this->radius;
        float discriminant = b * b - a * c;
        if (discriminant > 0) {
            float temp = (-b - sqrt(discriminant)) / a;
            if (temp > 0 && temp < record.distance) {
                record.distance = temp;
                record.position = ray.at(temp);
                record.normal = record.position - this->location;
                record.hitMaterial = this->material;
            }
            temp = (-b + sqrt(discriminant)) / a;
            if (temp > 0 && temp < record.distance) {
                record.distance = temp;
                record.position = ray.at(temp);
                record.normal = record.position - this->location;
                record.hitMaterial = this->material;
            }
        }
    }
};

static void *fixed_cudaMalloc(size_t len)
{
    void *p;
    if (cudaMalloc(&p, len) == cudaSuccess) return p;
    printf("Failure allocating\n");
    return 0;
}

class World {
public:
    Sphere* deviceSpheres;
    int* deviceNumSpheres;
    Color* backgroundColor;
    __host__ World(int numSpheres, Sphere* spheresToCopy, Color* backgroundColorIn) {
        deviceSpheres = (Sphere*)fixed_cudaMalloc(numSpheres * sizeof(Sphere));
        checkCudaErrors(cudaMemcpy((void*)deviceSpheres, (void*)spheresToCopy, numSpheres * sizeof(Sphere), cudaMemcpyHostToDevice));
        deviceNumSpheres = (int*) fixed_cudaMalloc(sizeof(int));
        checkCudaErrors(cudaMemcpy((void*)deviceNumSpheres, (void*) &numSpheres, sizeof(int), cudaMemcpyHostToDevice));
        backgroundColor = (Color*)fixed_cudaMalloc(sizeof(Color));
        checkCudaErrors(cudaMemcpy((void*)backgroundColor, (void*)backgroundColorIn, sizeof(Color), cudaMemcpyHostToDevice));
    }
    __device__ void checkRay(const Ray& ray, HitRecord& record) {
        for(int i = 0; i < *deviceNumSpheres; i++) {
            deviceSpheres[i].checkRay(ray, record);
        }
    }
    __device__ __host__ ~World() {
        //cudaFree(deviceSpheres);
    }
};

class Camera {
public:
    vec3 origin;
    float zoom;
    Camera(vec3 origin, float zoom) : origin(origin), zoom(zoom) {};
};

__device__ Color shade(const Ray& ray, World world, int bouncesRemaining, curandState* state) {
    HitRecord record = HitRecord();
    world.checkRay(ray, record);
    if (bouncesRemaining <= 1) {
        if (record.distance < maxDistance) {
            return record.hitMaterial.color;
        } else {
            return *world.backgroundColor;
        }
    }
    if (record.distance < maxDistance) {
        Color hitColor = record.hitMaterial.color;
        if(hitColor.r > 1.0f || hitColor.g > 1.0f || hitColor.b > 1.0f) {  // hit emissive material
            return hitColor;
        }
        Ray newRay = Ray(record.position, record.normal.normalized() + randomOnUnitSphere(state));
        return shade(newRay, world, --bouncesRemaining, state) * 0.5f;
    }  // hit nothing
    return *world.backgroundColor;
}

__global__ void render(float4* out_data, Color* colorBuffer, curandState* states, World world, Camera camera, int samples, int bounceLimit) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= width || y >= height) return;  // Out of bounds of image, no point rendering
    int pixelIndex = y * width + x;
    //if(pixelIndex == 0) {
        //world.deviceSpheres[3].location.x-=.1f;
    //}

    Ray cameraRay = Ray(camera.origin, vec3((x-width/2)/static_cast<float>(width), (y-height/2)/ static_cast<float>(width), camera.zoom));
    for (int s = 0; s < samples; s++) {
        colorBuffer[pixelIndex] += shade(cameraRay, world, bounceLimit, &states[pixelIndex]);  // Could add check for background and break
    }
    out_data[pixelIndex] = colorBuffer[pixelIndex].floatOutput();  // Could add check for background and break
}

__global__ void kernelInit(int width, int height, curandState* state, Color* colorBuffer) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    int pixelIndex = y * width + x;
    curand_init(pixelIndex, pixelIndex, 0, &state[pixelIndex]);
    colorBuffer[pixelIndex] = Color(0,0,0,0);
}

//https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html
//https://github.com/phrb/intro-cuda/blob/master/src/cuda-samples/2_Graphics/bindlessTexture/bindlessTexture.cpp
class Window {
    GLFWwindow* window;
    GLuint openGLPixelBuffer;
    struct cudaGraphicsResource* cudaFrameBuffer;
    Color* cudaColorBuffer;
    curandState* curandStates;
    dim3 blocks, threads;
    World world;
public:
    Window(World& world) : world(world){
        glfwInit();
        window = glfwCreateWindow(width, height, "test", NULL, NULL);
        glfwMakeContextCurrent(window);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glGenBuffersARB(1, &openGLPixelBuffer);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, openGLPixelBuffer);
        glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, width * height * sizeof(float4), 0, GL_STREAM_DRAW_ARB);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaFrameBuffer, openGLPixelBuffer, cudaGraphicsMapFlagsWriteDiscard));

        checkCudaErrors(cudaMallocManaged(&curandStates, numPixels * sizeof(curandState)));
        blocks = dim3(width / tileWidth + 1, height / tileHeight + 1);
        threads = dim3(tileWidth, tileHeight);
        cudaColorBuffer = (Color*)fixed_cudaMalloc(width * height * sizeof(Color));
        kernelInit <<<blocks, threads >>> (width, height, curandStates, cudaColorBuffer);
    }
    int renderFrame(Camera camera) {
        if(glfwWindowShouldClose(window)) return 1;
        float4* frameBuffer;
        checkCudaErrors(cudaGraphicsMapResources(1, &cudaFrameBuffer, 0));
        size_t numBytes;
        checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&frameBuffer, &numBytes, cudaFrameBuffer));
        //kernelInit <<<blocks, threads >>> (width, height, curandStates, cudaColorBuffer);
        render<<<blocks, threads>>>(frameBuffer, cudaColorBuffer, curandStates, world, camera, samples, bounceLimit);
        checkCudaErrors(cudaDeviceSynchronize());
        checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaFrameBuffer, 0));

        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);
        glRasterPos2i(-1,-1);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, openGLPixelBuffer);
        glDrawPixels(width, height, GL_RGBA, GL_FLOAT, 0);
        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        glfwSwapBuffers(window);
        glfwPollEvents();
        return 0;
    }
    ~Window() {
        cudaFree(curandStates);
        cudaGraphicsUnregisterResource(cudaFrameBuffer);
        glDeleteBuffersARB(1, &openGLPixelBuffer);
        cudaFree(cudaColorBuffer);
        cudaDeviceReset();  // best practice and helps with profiling
    }
};

int main(void)
{
    Camera camera = Camera(vec3(0, .75f, 0), 1.0f);

    Sphere sphere = Sphere(vec3(1.0f, 1.0f, 8.0f), 1.0f);
    sphere.material.color = Color(1, 1, 1);
    Sphere sphere2 = Sphere(vec3(0, -500.0f, 5.0f), 500.0f);
    sphere2.material.color = Color(.5f, .5f, .5f);
    Sphere sphere3 = Sphere(vec3(-1.0f, 1.0f, 8.0f), 1.0f);
    sphere3.material.color = Color(1, 1, 1);
    Sphere sphere4 = Sphere(vec3(-2.0f, 0.5f, 5.0f), 0.5f);
    sphere4.material.color = Color(60, 30, 50);
    Sphere sphere5 = Sphere(vec3(0.0f, 0.2f, 8.5f), 0.2f);
    sphere5.material.color = Color(20, 10, 30);
    Sphere spheres[] = {sphere, sphere2, sphere3, sphere4, sphere5};
    Color backgroundColor(0.0f,0.0f,0.0f);
    World world(5, spheres, &backgroundColor);

    Window window(world);

    while(!window.renderFrame(camera)) {}

    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    return 0;
}