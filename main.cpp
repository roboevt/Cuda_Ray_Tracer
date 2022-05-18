#include "RayTracer.h"

constexpr int width = 640;
constexpr int height = 480;

int main(void)
{
    Window window(width, height, "Cuda Ray Tracer");

    Sphere sphere = Sphere(-1.0f, 1.0f, 8.0f, 1.0f);
    sphere.color = Color(.5f, .5f, 1);
    Sphere sphere2 = Sphere(1.0f, 1.0f, 8.0f, 1.0f);
    sphere2.color = Color(1.0f, 1.0f, 1.0f);
    Sphere sphere3 = Sphere(0.0f, -100.0f, 8.0f, 100.0f);
    sphere3.color = Color(1, 1, 1);
    Sphere sphere4 = Sphere(0.0f, 2.0f, 8.0f, 0.5f);
    sphere4.color = Color(1, .2F, .3f);
    Sphere sphere5 = Sphere(1.25f, 0.2f, 6.5f, 0.2f);
    sphere5.color = Color(100, 50, 40);
    //sphere5.color = Color(0, 1, 1);

    RayTracer rayTracer(&window);
    Sphere spheres[] = {sphere, sphere2,  sphere3,sphere4, sphere4, sphere5};
    rayTracer.addSpheres(spheres, 6);
    rayTracer.setSamples(500);
    rayTracer.init();
    printf("Done initialization\n");

    rayTracer.loop();

    return 0;
}