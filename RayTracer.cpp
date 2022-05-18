//
// Created by roboevt on 5/6/22.
//

#include "RayTracer.h"
#include "Tracer.cuh"

namespace {
    CudaTracer tracer;
    CudaColor backgroundColor(10.10f,10.10f,10.10f);
}

Color::Color(float r, float g, float b) : r(r), g(g), b(b) {}

Sphere::Sphere(float x, float y, float z, float r) : x(x), y(y), z(z), r(r) {}

CudaSphere toCudaSphere(Sphere sphere) {
    CudaSphere cudaSphere(vec3(sphere.x,sphere.y,sphere.z),sphere.r);
    cudaSphere.material.color = CudaColor(sphere.color.r, sphere.color.g, sphere.color.b);
    return cudaSphere;
}

Window::Window(int width, int height, std::string name) : width(width), height(height), name(name) {
    glfwInit();
    window = glfwCreateWindow(this->width, this->height, name.c_str(), NULL, NULL);
    glfwMakeContextCurrent(window);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glGenBuffersARB(1, &openGLPixelBuffer);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, openGLPixelBuffer);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, this->width * this->height * sizeof(float4), 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
}

GLuint Window::getGLBuffer() {return openGLPixelBuffer;}

int Window::displayFrame() {
    if (glfwWindowShouldClose(window)) return 1;
    glClear(GL_COLOR_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(-1, -1);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, openGLPixelBuffer);
    glDrawPixels(width, height, GL_RGBA, GL_FLOAT, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    glfwSwapBuffers(window);
    glfwPollEvents();
    return 0;
}

void Window::setKeyCallback(GLFWkeyfun callback) {
    glfwSetKeyCallback(window, callback);
}

Window::~Window() {
    glDeleteBuffersARB(1, &openGLPixelBuffer);
}

RayTracer::RayTracer(Window *window){
    RayTracer::window = window;
}

void RayTracer::addSphere(Sphere sphere) {spheres.push_back(sphere); }
void RayTracer::addSpheres(Sphere* spheres, int numSpheres) {
    std::copy(spheres, spheres + numSpheres, std::back_inserter(this->spheres));
}

void RayTracer::setSamples(int samples) {this->samples = samples; }

void RayTracer::update(std::chrono::duration<float> timestep) {
    vec3 camDirection(0,0,0);
    if(glfwGetKey(window->window, GLFW_KEY_W) == GLFW_PRESS) {
        camDirection += tracer.camera.getForward(); }
    if(glfwGetKey(window->window, GLFW_KEY_S) == GLFW_PRESS) {
        camDirection += tracer.camera.getForward() * -1.0f; }
    if(glfwGetKey(window->window, GLFW_KEY_A) == GLFW_PRESS) {
        camDirection += tracer.camera.getSide(); }
    if(glfwGetKey(window->window, GLFW_KEY_D) == GLFW_PRESS) {
        camDirection += tracer.camera.getSide() * -1.0f; }
    if(glfwGetKey(window->window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {
        camDirection += vec3(0,1.0f,0); }
    if(glfwGetKey(window->window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS) {
        camDirection += vec3(0,1.0f,0); }
    if(glfwGetKey(window->window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) {
        camDirection += vec3(0,-1.0f,0); }
    if(glfwGetKey(window->window, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS) {
        camDirection += vec3(0,-1.0f,0); }
    if(glfwGetKey(window->window, GLFW_KEY_R) == GLFW_PRESS) {
        tracer.camera.zoom += 1.0f * timestep.count(); }
    if(glfwGetKey(window->window, GLFW_KEY_F) == GLFW_PRESS) {
        tracer.camera.zoom -= 1.0f * timestep.count(); }
    tracer.camera.ray.origin += camDirection * timestep.count();
    std::cout << timestep.count() << "s\n";
}

void RayTracer::renderFrame(bool clearFrame) {
    tracer.renderFrame(clearFrame);
}

void RayTracer::loop() {
    using namespace std::chrono;
    time_point<steady_clock> previousFrame = steady_clock::now();
    renderFrame(true);
    while(!window->displayFrame()) {
        time_point<steady_clock> currentFrame = steady_clock::now();
        duration<float> frameTime = currentFrame - previousFrame;
        previousFrame = currentFrame;
        update(frameTime);
        renderFrame(true);
    }
}

void RayTracer::init() {
    size_t numSpheres = this->spheres.size();
    CudaSphere* cudaSpheres;
    cudaSpheres = new CudaSphere[numSpheres];
    for(int i = 0; i < numSpheres; i++) {
        cudaSpheres[i] = toCudaSphere(this->spheres[i]);
    }
    World world(numSpheres, cudaSpheres, &backgroundColor);
    tracer.setWorld(world);
    Camera cudaCam(Ray(vec3(0, 0.5f, 0),vec3(0,0,1.0f)), 1.0f, this->window->width, this->window->height);
    tracer = CudaTracer(world, cudaCam, this->window->getGLBuffer());
    tracer.setSamples(this->samples);
}