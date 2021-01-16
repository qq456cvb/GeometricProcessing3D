#include "io/objreader.h"
#include <memory>

int main(int argc, char *argv[]) {
    auto reader = std::make_shared<ObjReader>();
    auto mesh = reader->read_mesh("../examples/data/bunny.obj");
    auto dist = mesh->geodesic(std::vector{4});

    return 0;
}