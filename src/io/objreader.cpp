#include "io/objreader.h"
#include <sstream>
#include <fstream>
#include <memory>
#include <algorithm>
#include <type_traits>
#include <iostream>
#include <assert.h>


auto get_raw_buffer(const char *fn, bool binary = false) -> std::istringstream {
    std::ifstream ifs;

    if (binary) ifs.open(fn, std::ios::in | std::ios::binary);
    else ifs.open(fn, std::ios::in);

    // seek to end
    ifs.seekg(0, std::ios_base::end);
    auto length = ifs.tellg();

    // return to start
    ifs.seekg(0, std::ios_base::beg);
    std::shared_ptr<char[]> buf(new char[length], std::default_delete<char[]>());

    ifs.read(buf.get(), length);
    ifs.close();
    return std::istringstream(buf.get());
}

// http://stackoverflow.com/questions/6089231/getting-std-ifstream-to-handle-lf-cr-and-crlf
inline void safe_get_line(std::istream &is, std::string &t) {
    t.clear();

    std::istream::sentry se(is, true);
    std::streambuf *sb = is.rdbuf();

    if (se) {
        while (true) {
            int c = sb->sbumpc();
            switch (c) {
                case '\n':
                    return;
                case '\r':
                    if (sb->sgetc() == '\n') sb->sbumpc();
                    return;
                case EOF:
                    // Also handle the case when the last line has no line ending
                    if (t.empty()) is.setstate(std::ios::eofbit);
                    return;
                default:
                    t += static_cast<char>(c);
            }
        }
    }
}

template <typename T>
static auto parse_single_value(const char *&str) {
    static char *end;
    if constexpr (std::is_floating_point<T>::value) {
        float res = static_cast<float>(std::strtod(str, &end));
        str = end;
        return res;
    } else if constexpr (std::is_integral<T>::value) {
        int res = static_cast<int>(std::strtol(str, &end, 10));
        str = end;
        return res;
    } else {
        static_assert(std::is_arithmetic<T>::value,"unknown type");
        return;
    }
}

template <typename T>
static inline std::array<T, 3> parse_triplet(const char *&str) {
    auto v1 = parse_single_value<T>(str);
    if (*str != '/') return {v1, 0, 0};
    str++;
    if (*str == '/') {
        str++;
        return {v1, 0, parse_single_value<T>(str)};
    }
    auto v2 = parse_single_value<T>(str);
    if (*str != '/') {
        return {v1, v2, 0};
    }
    str++;
    return {v1, v2, parse_single_value<T>(str)};
}

template <typename T>
static inline std::array<T, 3> parse_triplet_direct(const char *&str) {
    auto v1 = parse_single_value<T>(str);
    auto v2 = parse_single_value<T>(str);
    auto v3 = parse_single_value<T>(str);
    return {v1, v2, v3};
}


ObjReader::ObjReader()
{
}

ObjReader::~ObjReader()
{
}

std::shared_ptr<TriangleMesh> ObjReader::read_mesh(const char *fn)
{
    auto stream = get_raw_buffer(fn);
    std::string line_buf;
    
    std::vector<xyz> verts;
    std::vector<face_idx_t> face_idxs;
    while (stream.peek() != EOF)
    {
        safe_get_line(stream, line_buf);
        line_buf.erase(line_buf.begin(), std::find_if(line_buf.begin(), line_buf.end(), [](char ch) {
            return ch != ' ' && ch != '\t';
        }));

        const char *line = line_buf.c_str();
        if (line[0] == 'v' && line[1] == ' ') {
            line++;
            auto xyz = parse_triplet_direct<float>(line);
            verts.push_back(xyz);
            // printf("x: %f, y: %f, z: %f\n", xyz[0], xyz[1], xyz[2]);
        } else if (line[0] == 'f') {
            line++;
            auto f1 = parse_triplet<int>(line);
            auto f2 = parse_triplet<int>(line);
            auto f3 = parse_triplet<int>(line);
            face_idx_t f;
            f.vs = {f1[0] - 1, f2[0] - 1, f3[0] - 1};
            f.ns = {f1[1] - 1, f2[1] - 1, f3[1] - 1};
            f.ts = {f1[2] - 1, f2[2] - 1, f3[2] - 1};
            face_idxs.push_back(f);
        }
    }

    return std::make_shared<TriangleMesh>(std::move(verts), std::move(face_idxs));
}

std::shared_ptr<PointCloud> ObjReader::read_cloud(const char *fn)
{
    auto stream = get_raw_buffer(fn);
    std::string line_buf;
    
    std::vector<xyz> verts;
    std::vector<face_idx_t> face_idxs;
    bool data_start = false;
    while (stream.peek() != EOF)
    {
        safe_get_line(stream, line_buf);
        line_buf.erase(line_buf.begin(), std::find_if(line_buf.begin(), line_buf.end(), [](char ch) {
            return ch != ' ' && ch != '\t';
        }));

        if (!data_start) {
            if (line_buf.substr(0, 10) != "DATA ascii") continue;
            data_start = true;
            continue;
        }
        const char *line = line_buf.c_str();
        auto xyz = parse_triplet_direct<float>(line);
        verts.push_back(xyz);
    }

    return std::make_shared<PointCloud>(std::move(verts));
}
