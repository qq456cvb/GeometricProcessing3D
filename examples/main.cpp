#include "io/objreader.h"
#include "algorithm/ppf.h"
#include <memory>
#include <armadillo>

int main(int argc, char *argv[]) {
    auto reader = std::make_shared<ObjReader>();
    // auto mesh = reader->read_mesh("../examples/data/bunny.obj");
    // auto dist = mesh->geodesic(std::vector{4});

    auto pc = reader->read_cloud("../examples/data/model_chair.pcd");
    auto scene_ptr = reader->read_cloud("../examples/data/scene_chair.pcd");
    // auto scene = *scene_ptr;
    auto scene = *pc;

    arma::fmat33 r = {
        {-0.78502175, -0.61736402, -0.05101488},
        {-0.13302366,  0.2484333 , -0.95947152},
        { 0.60501699, -0.74641982, -0.27714957}
    };
    arma::fvec3 t = { 0.70593162, -0.3494284,  -0.24574194 };
    for (size_t i = 0; i < scene.verts.size(); i++)
    {
        arma::fvec pc((float *)(&scene.verts[i]), 3, false, true);
        pc = r * pc + t;
        scene.verts[i][0] = pc(0);
        scene.verts[i][1] = pc(1);
        scene.verts[i][2] = pc(2);

        arma::fvec normal((float *)(&scene.normals[i]), 3, false, true);
        normal = r * normal;
        scene.normals[i][0] = normal(0);
        scene.normals[i][1] = normal(1);
        scene.normals[i][2] = normal(2);
    }
    

    auto ppf = std::make_shared<PPF>(0.01f, 12.0f / 180.0f * static_cast<float>(M_PI), 0.05f, 24.0f / 180.0f * static_cast<float>(M_PI));
    ppf->setup_model(*pc);
    ppf->detect(scene);
    return 0;
}