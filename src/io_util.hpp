#pragma once

#include <algorithm>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

struct MappedFile {
    const char* data = nullptr;
    size_t size = 0;

    MappedFile() = default;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    MappedFile(MappedFile&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }
    MappedFile& operator=(MappedFile&& other) noexcept {
        if (this != &other) {
            unmap();
            data = other.data;
            size = other.size;
            other.data = nullptr;
            other.size = 0;
        }
        return *this;
    }
    ~MappedFile() { unmap(); }

    void unmap() noexcept {
        if (data && size > 0) {
            ::munmap(const_cast<char*>(data), size);
        }
        data = nullptr;
        size = 0;
    }

    static MappedFile open_readonly(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("failed to open: " + path);
        }
        struct stat st {};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("failed to fstat: " + path);
        }
        MappedFile mf;
        mf.size = static_cast<size_t>(st.st_size);
        if (mf.size == 0) {
            ::close(fd);
            return mf;
        }
        void* p = ::mmap(nullptr, mf.size, PROT_READ, MAP_PRIVATE, fd, 0);
        ::close(fd);
        if (p == MAP_FAILED) {
            throw std::runtime_error("failed to mmap: " + path);
        }
        ::madvise(p, mf.size, MADV_SEQUENTIAL);
        mf.data = static_cast<const char*>(p);
        return mf;
    }
};

inline std::vector<std::string> list_csv_files(const std::string& dir) {
    std::vector<std::string> paths;
    DIR* d = ::opendir(dir.c_str());
    if (!d) return paths;
    struct dirent* ent;
    while ((ent = ::readdir(d)) != nullptr) {
        const char* name = ent->d_name;
        const size_t nlen = std::strlen(name);
        if (nlen <= 4 || std::memcmp(name + nlen - 4, ".csv", 4) != 0) continue;
        std::string full = dir;
        if (!full.empty() && full.back() != '/') full.push_back('/');
        full.append(name, nlen);
        struct stat st {};
        if (::stat(full.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
            paths.push_back(std::move(full));
        }
    }
    ::closedir(d);
    std::sort(paths.begin(), paths.end());
    return paths;
}

struct FileSlice {
    const char* base = nullptr;
    size_t start = 0;
    size_t end = 0;
};

inline std::vector<FileSlice> build_slices(const std::vector<MappedFile>& files,
                                           size_t chunk_bytes = size_t(4) << 20) {
    std::vector<FileSlice> slices;
    for (const MappedFile& mf : files) {
        if (mf.size == 0) continue;
        const char* nl = static_cast<const char*>(std::memchr(mf.data, '\n', mf.size));
        if (!nl) continue;  // header only, no trailing newline -> 0 data rows
        size_t pos = static_cast<size_t>(nl - mf.data) + 1;
        while (pos < mf.size) {
            size_t target = std::min(mf.size, pos + chunk_bytes);
            if (target < mf.size) {
                const char* end_nl = static_cast<const char*>(
                    std::memchr(mf.data + target, '\n', mf.size - target));
                target = end_nl ? static_cast<size_t>(end_nl - mf.data) + 1 : mf.size;
            }
            slices.push_back(FileSlice{mf.data, pos, target});
            pos = target;
        }
    }
    return slices;
}

inline std::vector<MappedFile> map_all_csvs(const std::string& dir) {
    auto paths = list_csv_files(dir);
    std::vector<MappedFile> files;
    files.reserve(paths.size());
    for (const auto& p : paths) {
        files.push_back(MappedFile::open_readonly(p));
    }
    return files;
}
