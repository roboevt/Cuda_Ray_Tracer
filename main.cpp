#include "RayTracer.h"

constexpr int width = 1920;
constexpr int height = 1080;

int main(void)
{
    Window window(width, height, "Cuda Ray Tracer");

    Sphere sphere = Sphere(-1.0f, 1.0f, 8.0f, 1.0f);
    sphere.color = Color(1, 1, 1);
    Sphere sphere2 = Sphere(1.0f, 1.0f, 8.0f, 1.0f);
    sphere2.color = Color(.5f, .5f, .5f);
    Sphere sphere3 = Sphere(0.0f, -100.0f, 8.0f, 136.0f);
    sphere3.color = Color(1, 1, 1);
    Sphere sphere4 = Sphere(0.0f, 5.5f, 4.0f, 0.5f);
    sphere4.color = Color(60, 80, 90);
    Sphere sphere5 = Sphere(1.5f, 0.2f, 5.5f, 0.2f);
    sphere5.color = Color(40, 15, 10);

    RayTracer rayTracer(&window);
    Sphere spheres[] = {sphere, sphere2,  sphere3,sphere4, sphere4, sphere5};
    rayTracer.addSpheres(spheres, 6);
    rayTracer.setSamples(25);
    rayTracer.init();
    window.setKeyCallback(RayTracer::keyPressCallback);
    printf("Done initialization\n");
    while(1) {
        rayTracer.renderFrame(true);
        if(window.displayFrame()) break;
    }

    return 0;
}