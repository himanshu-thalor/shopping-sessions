#pragma once

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

struct PreadFile {
    int fd = -1;
    size_t size = 0;

    PreadFile() = default;
    PreadFile(const PreadFile&) = delete;
    PreadFile& operator=(const PreadFile&) = delete;
    PreadFile(PreadFile&& other) noexcept : fd(other.fd), size(other.size) {
        other.fd = -1;
        other.size = 0;
    }
    PreadFile& operator=(PreadFile&& other) noexcept {
        if (this != &other) {
            close_fd();
            fd = other.fd;
            size = other.size;
            other.fd = -1;
            other.size = 0;
        }
        return *this;
    }
    ~PreadFile() { close_fd(); }

    void close_fd() noexcept {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        size = 0;
    }

    static PreadFile open_readonly(const std::string& path) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("failed to open: " + path);
        }
        struct stat st {};
        if (::fstat(fd, &st) != 0) {
            ::close(fd);
            throw std::runtime_error("failed to fstat: " + path);
        }
        PreadFile pf;
        pf.fd = fd;
        pf.size = static_cast<size_t>(st.st_size);
        if (pf.size > 0) {
            ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
        }
        return pf;
    }
};

struct PreadChunk {
    int fd = -1;
    size_t file_size = 0;
    size_t start = 0;
    size_t end = 0;
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

inline std::vector<PreadFile> open_all_csvs(const std::string& dir) {
    auto paths = list_csv_files(dir);
    std::vector<PreadFile> files;
    files.reserve(paths.size());
    for (const auto& p : paths) {
        files.push_back(PreadFile::open_readonly(p));
    }
    return files;
}

inline std::vector<PreadChunk> build_pread_chunks(const std::vector<PreadFile>& files,
                                                  size_t chunk_bytes = size_t(2) << 20) {
    std::vector<PreadChunk> chunks;
    for (const PreadFile& f : files) {
        if (f.size == 0) continue;
        for (size_t pos = 0; pos < f.size; pos += chunk_bytes) {
            const size_t end = std::min(f.size, pos + chunk_bytes);
            chunks.push_back(PreadChunk{f.fd, f.size, pos, end});
        }
    }
    return chunks;
}

inline void pread_exact(int fd, char* dst, size_t len, size_t offset) {
    size_t got = 0;
    while (got < len) {
        ssize_t n = ::pread(fd, dst + got, len - got, static_cast<off_t>(offset + got));
        if (n < 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error("pread failed");
        }
        if (n == 0) break;
        got += static_cast<size_t>(n);
    }
}

inline void read_chunk_into(const PreadChunk& c, std::vector<char>& buf, size_t& data_start,
                            size_t& limit_in_buf, size_t& buf_len) {
    if (c.file_size == 0 || c.end <= c.start) {
        data_start = 0;
        limit_in_buf = 0;
        buf_len = 0;
        return;
    }
    const size_t read_from = (c.start == 0) ? 0 : (c.start - 1);
    const size_t limit = c.end - read_from;
    static constexpr size_t kTailWindow = 65536;
    size_t read_to = std::min(c.file_size, c.end + kTailWindow);
    size_t len = read_to - read_from;

    buf.resize(len + 64);
    pread_exact(c.fd, buf.data(), len, read_from);

    if (c.end < c.file_size) {
        const size_t tail_start = limit - 1;
        while (std::memchr(buf.data() + tail_start, '\n', len - tail_start) == nullptr &&
               read_to < c.file_size) {
            const size_t next_to = std::min(c.file_size, read_to + kTailWindow);
            const size_t add = next_to - read_to;
            buf.resize(len + add + 64);
            pread_exact(c.fd, buf.data() + len, add, read_to);
            len += add;
            read_to = next_to;
        }
    }

    std::memset(buf.data() + len, 0, 64);
    buf_len = len;

    const char* first_nl = static_cast<const char*>(std::memchr(buf.data(), '\n', len));
    if (!first_nl) {
        data_start = 0;
        limit_in_buf = 0;
        return;
    }
    data_start = static_cast<size_t>(first_nl - buf.data()) + 1;
    limit_in_buf = limit;
}

inline std::vector<char> read_whole_file(const std::string& path) {
    PreadFile pf = PreadFile::open_readonly(path);
    std::vector<char> data(pf.size);
    if (pf.size > 0) {
        pread_exact(pf.fd, data.data(), pf.size, 0);
    }
    return data;
}
