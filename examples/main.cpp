#include "io/objreader.h"
#include <memory>

int main(int argc, char *argv[]) {
    auto reader = std::make_shared<ObjReader>("../examples/data/test.obj");

    return 0;
}