#include "io/objreader.h"
#include "algorithm/ppf.h"
#include <memory>

int main(int argc, char *argv[]) {
    auto reader = std::make_shared<ObjReader>();
    // auto mesh = reader->read_mesh("../examples/data/bunny.obj");
    // auto dist = mesh->geodesic(std::vector{4});

    auto pc = reader->read_cloud("../examples/data/model_chair.pcd");

    auto ppf = std::make_shared<PPF>(0.01f, 12.0f / 180.0f * static_cast<float>(M_PI));
    ppf->setup_model(*pc);
    return 0;
}