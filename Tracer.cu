#include "cuda_runtime.h"
#include "curand.h"
#include <cuda_gl_interop.h>
#include "device_launch_parameters.h"

#include "Tracer.cuh"



void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        std::cerr << cudaGetErrorString(result);
        cudaDeviceReset();
        exit(99);
    }
}

static void* fixed_cudaMalloc(size_t len) {
    void *p;
    if (cudaMalloc(&p, len) == cudaSuccess) return p;
    printf("Failure allocating\n");
    return 0;
}

__host__ __device__ float clamp(float x, float a, float b) {
    return max(a, min(b, x));
}

__host__ __device__ vec3::vec3(float x, float y, float z) : x(x), y(y), z(z) {}
__host__ __device__ vec3::vec3() : x(0), y(0), z(0) {}
__host__ __device__ vec3 vec3::operator+(const vec3 other) const { return vec3(this->x + other.x, this->y + other.y, this->z + other.z); }
__host__ __device__ vec3 vec3::operator-(const vec3 other) const { return vec3(this->x - other.x, this->y - other.y, this->z - other.z); }
__host__ __device__ void vec3::operator-=(const vec3 other) { this->x -= other.x; this->y -= other.y; this->z -= other.z; }
__host__ __device__ vec3 vec3::operator*(const float scale) const { return vec3(this->x * scale, this->y * scale, this->z * scale); };
__host__ __device__ void vec3::operator*=(const float scale) { this->x *= scale; this->y *= scale; this->z *= scale; };
__host__ __device__ float vec3::operator*(const vec3 other) const { return this->x * other.x + this->y * other.y + this->z * other.z; }  // dot product
__host__ __device__ vec3 vec3::cross(const vec3 other) const { return vec3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x); }
__host__ __device__ float vec3::magnitudeSquared() const { return (this->x * this->x) + (this->y * this->y) + (this->z * this->z);}
__host__ __device__ float vec3::magnitude() const { return sqrt(this->magnitudeSquared()); }
__host__ __device__ vec3 vec3::normalized() const { float mag = this->magnitude(); return vec3(x/mag,y/mag,z/mag);}
__host__ __device__ float vec3::project(const vec3 w, const vec3 v) {return (w * v) / v.magnitudeSquared();}
__device__  vec3 vec3::randomInUnitSphere(curandState* state) {
    vec3 temp = vec3(curand_uniform(state) * 2.0f - 1.0f, curand_uniform(state) * 2.0f - 1.0f, curand_uniform(state) * 2.0f - 1.0f);
    while (temp.magnitudeSquared() >= 1) {
        float newX = (curand_uniform(state) * 2.0f) - 1.0f;
        float newY = (curand_uniform(state) * 2.0f) - 1.0f;
        float newZ = (curand_uniform(state) * 2.0f) - 1.0f;
        temp = vec3(newX, newY, newZ);
    }
    return temp;
}

__host__ __device__ CudaColor::CudaColor(float r, float g, float b) : r(r), g(g), b(b), samples(1) {}
__host__ __device__ CudaColor::CudaColor(float r, float g, float b, int samples) : r(r), g(g), b(b), samples(samples) {}
__host__ __device__ CudaColor  CudaColor::operator+(const CudaColor other) const { return CudaColor(this->r + other.r, this->g + other.g, this->b + other.b, this->samples + other.samples); }
__host__ __device__ void CudaColor::operator+=(const CudaColor other) { this->r += other.r; this->g += other.g; this->b += other.b; this->samples += other.samples + 1; }
__host__ __device__ CudaColor CudaColor::operator-(const CudaColor other) const { return CudaColor(this->r - other.r, this->g - other.g, this->b - other.b, this->samples - other.samples); }
__host__ __device__ void CudaColor::operator-=(const CudaColor other) { this->r -= other.r; this->g -= other.g; this->b -= other.b; this->samples -= other.samples; }
__host__ __device__ CudaColor CudaColor::operator*(const float scale) const { return CudaColor(this->r * scale, this->g * scale, this->b * scale, this->samples); }
__host__ __device__ void CudaColor::operator*=(const float scale) { *this = *this * scale; }
__host__ __device__ CudaColor CudaColor::operator*(const CudaColor other) const {return CudaColor(this->r * other.r, this->g * other.g, this->b * other.b, this->samples + other.samples);}
__host__ __device__ bool CudaColor::operator==(const CudaColor other) {return this->r == other.r & this->g == other.g & this->b == other.b;}
__host__ __device__ void CudaColor::sample() {this->samples++;}
__host__ __device__ void CudaColor::sample(int samples) {this->samples += samples;}
__host__ __device__ CudaColor CudaColor::output() { return CudaColor(clamp(this->r / this->samples, 0.0f, 1.0f), clamp(this->g / this->samples, 0.0f, 1.0f), clamp(this->b / this->samples, 0.0f, 1.0f)); }
//__host__ __device__ CudaColor CudaColor::output() { return CudaColor(clamp(this->r, 0.0f, 1.0f), clamp(this->g, 0.0f, 1.0f), clamp(this->b, 0.0f, 1.0f)); }
__host__ __device__ float4 CudaColor::floatOutput() {
    CudaColor outputCol = this->output();
    return make_float4(outputCol.r, outputCol.g, outputCol.b, 1.0f);
}

__host__ __device__ Ray::Ray(vec3 origin, vec3 direction) : origin(origin), direction(direction) {}
__device__ vec3 Ray::at(float distance) const { return this->origin + (this->direction.normalized() * distance); }

__host__ __device__ Material::Material() : color(CudaColor(0, 0, 0)) {}
__host__ __device__ Material::Material(CudaColor color) : color(color) {}

__device__ HitRecord::HitRecord() : distance(INFINITY), position(vec3(0, 0, 0)), normal(vec3(0, 0, 0)), hitMaterial(Material()) {}

__host__ __device__ Hittable::Hittable(vec3 location, Material material) : location(location), material(material) {}

__host__ __device__ CudaSphere::CudaSphere(vec3 location, float radius) : location(location), material(material), radius(radius) {}
__host__ __device__ CudaSphere::CudaSphere() {}
__device__ void CudaSphere::checkRay(Ray ray, HitRecord* record) {
    vec3 toSphere = this->location - ray.origin;
    float distanceToNearestPoint = toSphere * ray.direction.normalized();
    if (distanceToNearestPoint < 0) return; // sphere is behind ray
    float distanceToCenter = toSphere.magnitude();
    float distanceCenterToRay = sqrt((distanceToCenter * distanceToCenter)
            - (distanceToNearestPoint * distanceToNearestPoint));
    if (distanceCenterToRay > this->radius) return;  // ray misses sphere
    float distanceBackToCollision = sqrt(this->radius * this->radius -
                                         distanceCenterToRay *
                                         distanceCenterToRay);
    float distanceToCollision = distanceToNearestPoint - distanceBackToCollision > 0 ?
            distanceToNearestPoint - distanceBackToCollision : distanceToNearestPoint + distanceBackToCollision;
    if (distanceToCollision > record->distance) return;  // not closest object
    record->distance = distanceToCollision;
    record->hitMaterial = this->material;
    record->position = ray.at(record->distance);
    record->normal = (record->position - this->location).normalized();
    return;
}

__host__ World::World(size_t numSpheres, CudaSphere* spheresToCopy, CudaColor* backgroundColorIn) {
    deviceSpheres = (CudaSphere*)fixed_cudaMalloc(numSpheres * sizeof(CudaSphere));
    checkCudaErrors(cudaMemcpy((void*)deviceSpheres, (void*)spheresToCopy, numSpheres * sizeof(CudaSphere), cudaMemcpyHostToDevice));
    deviceNumSpheres = (size_t*) fixed_cudaMalloc(sizeof(size_t));
    checkCudaErrors(cudaMemcpy((void*)deviceNumSpheres, (void*) &numSpheres, sizeof(size_t), cudaMemcpyHostToDevice));
    backgroundColor = (CudaColor*)fixed_cudaMalloc(sizeof(CudaColor));
    checkCudaErrors(cudaMemcpy((void*)backgroundColor, (void*)backgroundColorIn, sizeof(CudaColor), cudaMemcpyHostToDevice));
}

__host__ World::World() {}  // TODO may need to do somthing...

__host__ void World::setSpheres(CudaSphere *spheresToCopy, size_t numSpheres) {
    deviceSpheres = (CudaSphere*)fixed_cudaMalloc(numSpheres * sizeof(CudaSphere));
    checkCudaErrors(cudaMemcpy((void*)deviceSpheres, (void*)spheresToCopy, numSpheres * sizeof(CudaSphere), cudaMemcpyHostToDevice));
    deviceNumSpheres = (size_t*) fixed_cudaMalloc(sizeof(size_t));
    checkCudaErrors(cudaMemcpy((void*)deviceNumSpheres, (void*) &numSpheres, sizeof(size_t), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void World::setBackgroundColor(CudaColor* backgroundColorIn) {
    backgroundColor = (CudaColor*)fixed_cudaMalloc(sizeof(CudaColor));
    checkCudaErrors(cudaMemcpy((void*)this->backgroundColor, (void*)backgroundColorIn, sizeof(CudaColor), cudaMemcpyDefault));
}

__device__ void World::checkRay(Ray ray, HitRecord* record) const {
    for(int i = 0; i < *deviceNumSpheres; i++) {
        deviceSpheres[i].checkRay(ray, record);
    }
}

__device__ __host__ World::~World() {  // TODO free device memory
    //cudaFree(deviceSpheres);
}

Camera::Camera(Ray ray, float zoom, int width, int height) : ray(ray), zoom(zoom), width(width), height(height) {};

vec3 Camera::getForward() { return ray.direction.normalized(); }

vec3 Camera::getSide() { return ray.direction.cross(vec3(0,1,0)).normalized(); }

__device__ CudaColor shade(Ray ray, const World world, int bouncesRemaining, curandState* state) {
    CudaColor pixelColor(0,0,0,0);
    float depthMultiplier = 1.0f;  // every further bounce contributes less to the final color
    for(int i = 0; i < bouncesRemaining; i++) {
        depthMultiplier *= 0.5f;
        HitRecord record;
        world.checkRay(ray, &record);
        if(record.distance == INFINITY) {  // hit nothing
            pixelColor += (*(world.backgroundColor) * depthMultiplier);
            break;
        } else {  // hit something
            CudaColor hitColor = record.hitMaterial.color;
            if(hitColor.r > 1.0f || hitColor.g > 1.0f || hitColor.b > 1.0f) {  // hit emissive material
                pixelColor += hitColor * depthMultiplier;
                break;
            }
            // "recursive" bounce
            pixelColor += hitColor * depthMultiplier;
            ray = Ray(record.position, (record.normal.normalized() +
                                        vec3::randomInUnitSphere(state)).normalized());

        }

    }
    return pixelColor;

    HitRecord record = HitRecord();
    world.checkRay(ray, &record);
    if (bouncesRemaining <= 1) {
        if (record.distance < INFINITY) {
            return record.hitMaterial.color;
        } else {
            return *world.backgroundColor;
        }
    }
    if (record.distance < INFINITY) {
        CudaColor hitColor = record.hitMaterial.color;
        if(hitColor.r > 1.0f || hitColor.g > 1.0f || hitColor.b > 1.0f) {  // hit emissive material
            return hitColor;
        }
        Ray newRay = Ray(record.position, (record.normal.normalized() +
                                           vec3::randomInUnitSphere(state)).normalized());
        return shade(newRay, world, --bouncesRemaining, state) * 0.5f;
    }  // hit nothing
    return *world.backgroundColor;
}

__global__  void initKernel(int width, int height, curandState *states, CudaColor *colorBuffer) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= width) || (y >= height)) return;
    int pixelIndex = y * width + x;
    curand_init(pixelIndex, pixelIndex, 0, &states[pixelIndex]);
    colorBuffer[pixelIndex] = CudaColor(0, 0, 0, 0);
}

__global__ void renderFrameKernel(float4 *out_data, CudaColor *colorBuffer, curandState *states,
                                  const World world, const Camera camera, int samples, int bounceLimit,
                                  bool clearFrame) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x >= camera.width || y >= camera.height)
        return;  // Out of bounds of image, no point rendering
    int pixelIndex = y * camera.width + x;
    if (clearFrame) colorBuffer[pixelIndex] = CudaColor(0, 0, 0, 0);
    //if(pixelIndex == 0) world.deviceSpheres[4].location.x-=.1f;
    Ray cameraRay = Ray(camera.ray.origin, vec3((x - camera.width / 2) /
                                            static_cast<float>(camera.width),
                                            (y - camera.height / 2) /
                                            static_cast<float>(camera.width),
                                            camera.zoom));

    for (int s = 0; s < samples; s++) {
        colorBuffer[pixelIndex] += shade(cameraRay, world, bounceLimit,
                                         &states[pixelIndex]);
        //colorBuffer[pixelIndex].sample();
    }
    //colorBuffer[pixelIndex] *= (1.0f/static_cast<float>(samples));
    out_data[pixelIndex] = colorBuffer[pixelIndex].floatOutput();
}

CudaTracer::CudaTracer(World world, Camera camera, GLuint openGLPixelBuffer) : world(world), camera(camera), numPixels(camera.width * camera.height), tileWidth(16), tileHeight(16), bounceLimit(5) {  // TODO magic tileWidth/height
    checkCudaErrors(cudaGraphicsGLRegisterBuffer(&cudaFrameBuffer, openGLPixelBuffer, cudaGraphicsMapFlagsWriteDiscard));
    init();
}

CudaTracer::CudaTracer() : camera(Ray(vec3(0,0,0), vec3(0,0,1.0f)), 1.0f, 1920, 1080), tileWidth(16), tileHeight(16), bounceLimit(5) {}

void CudaTracer::setWorld(World wold) {this->world = world;}
void CudaTracer::setCamera(Camera camera) { this->camera = camera; this->numPixels = camera.width * camera.height;}
//void CudaTracer::setGLPixelBuffer(GLuint openGLPixelBuffer) {this->openGLPixelBuffer = openGLPixelBuffer;}
void CudaTracer::setSamples(int samples) {this->numSamples = samples;}

void CudaTracer::init() {
    checkCudaErrors(cudaMallocManaged(&this->curandStates, numPixels * sizeof(curandState)));
    blocks = dim3(camera.width / tileWidth + 1, camera.height / tileHeight + 1);
    threadsPerBlock = dim3(tileWidth, tileHeight);
    this->cudaColorBuffer = (CudaColor*)fixed_cudaMalloc(camera.width * camera.height * sizeof(CudaColor));
    initKernel <<<blocks, threadsPerBlock >>> (camera.width, camera.height, this->curandStates, this->cudaColorBuffer);
    checkCudaErrors(cudaDeviceSynchronize());
}

void CudaTracer::renderFrame(bool clearFrame) {
    float4* frameBuffer;
    size_t numBytes;
    checkCudaErrors(cudaGraphicsMapResources(1, &cudaFrameBuffer, 0));
    checkCudaErrors(cudaGraphicsResourceGetMappedPointer((void**)&frameBuffer, &numBytes, cudaFrameBuffer));
    renderFrameKernel<<<blocks,threadsPerBlock>>>(frameBuffer, cudaColorBuffer, curandStates, world, camera, numSamples, this->bounceLimit, clearFrame);
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaGraphicsUnmapResources(1, &cudaFrameBuffer, 0));
}

CudaTracer::~CudaTracer() {
    //cudaFree(curandStates);
    //cudaGraphicsUnregisterResource(cudaFrameBuffer);
    //cudaDeviceReset();  // best practice + helps with profiling
}
