#include "io/objreader.h"
#include <memory>

int main(int argc, char *argv[]) {
    auto reader = std::make_shared<ObjReader>();
    auto mesh = reader->read_mesh("../examples/data/test.obj");

    return 0;
}