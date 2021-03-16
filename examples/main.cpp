#include "io/objreader.h"
#include "algorithm/ppf.h"
#include <memory>

int main(int argc, char *argv[]) {
    auto reader = std::make_shared<ObjReader>();
    // auto mesh = reader->read_mesh("../examples/data/bunny.obj");
    // auto dist = mesh->geodesic(std::vector{4});

    auto pc = reader->read_cloud("../examples/data/chair.pcd");

    auto ppf = std::make_shared<PPF>();
    ppf->setup_model(*pc);
    return 0;
}