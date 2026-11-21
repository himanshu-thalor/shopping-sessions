#pragma once

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fcntl.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "thread_pool.hpp"

struct PreadFile {
    int fd = -1;
    size_t size = 0;
    std::string path;

    PreadFile() = default;
    PreadFile(const PreadFile&) = delete;
    PreadFile& operator=(const PreadFile&) = delete;

    PreadFile(PreadFile&& o) noexcept : fd(o.fd), size(o.size), path(std::move(o.path)) {
        o.fd = -1;
        o.size = 0;
    }

    PreadFile& operator=(PreadFile&& o) noexcept {
        if (this != &o) {
            close();
            fd = o.fd;
            size = o.size;
            path = std::move(o.path);
            o.fd = -1;
            o.size = 0;
        }
        return *this;
    }

    ~PreadFile() { close(); }

    void close() noexcept {
        if (fd >= 0) {
            ::close(fd);
            fd = -1;
        }
        size = 0;
    }
};

struct PreadChunk {
    uint32_t file_idx = 0;
    size_t raw_start = 0;
    size_t raw_end = 0;
};

struct PreadView {
    const char* data = nullptr;
    size_t pos = 0;
    size_t limit = 0;
    size_t len = 0;
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
        if (ent->d_type == DT_REG) {
            paths.push_back(std::move(full));
        } else if (ent->d_type == DT_UNKNOWN || ent->d_type == DT_LNK) {
            struct stat st {};
            if (::stat(full.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
                paths.push_back(std::move(full));
            }
        }
    }
    ::closedir(d);
    std::sort(paths.begin(), paths.end());
    return paths;
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

inline PreadFile open_one_pread_file(const std::string& path) {
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
    pf.path = path;
    return pf;
}

inline std::vector<PreadFile> open_all_pread_files(const std::string& dir, ThreadPool& pool,
                                                   size_t* out_total_bytes = nullptr) {
    auto paths = list_csv_files(dir);
    std::vector<PreadFile> files(paths.size());
    if (paths.empty()) {
        if (out_total_bytes) *out_total_bytes = 0;
        return files;
    }

    if (paths.size() >= 8 && pool.size() > 1) {
        std::atomic<size_t> next_idx{0};
        std::atomic<size_t> total{0};
        std::mutex err_mu;
        std::string err_msg;
        pool.run([&](size_t) {
            size_t local_total = 0;
            for (;;) {
                const size_t i = next_idx.fetch_add(1, std::memory_order_relaxed);
                if (i >= paths.size()) break;
                try {
                    files[i] = open_one_pread_file(paths[i]);
                    local_total += files[i].size;
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lk(err_mu);
                    if (err_msg.empty()) err_msg = e.what();
                }
            }
            total.fetch_add(local_total, std::memory_order_relaxed);
        });
        if (!err_msg.empty()) throw std::runtime_error(err_msg);
        if (out_total_bytes) *out_total_bytes = total.load(std::memory_order_relaxed);
    } else {
        size_t total = 0;
        for (size_t i = 0; i < paths.size(); ++i) {
            files[i] = open_one_pread_file(paths[i]);
            total += files[i].size;
        }
        if (out_total_bytes) *out_total_bytes = total;
    }
    return files;
}

inline std::vector<PreadChunk> build_pread_chunks(const std::vector<PreadFile>& files,
                                                  size_t chunk_bytes = size_t(256) << 10) {
    std::vector<PreadChunk> chunks;
    chunks.reserve(files.size() * 8);
    for (uint32_t fi = 0; fi < static_cast<uint32_t>(files.size()); ++fi) {
        const size_t sz = files[fi].size;
        for (size_t pos = 0; pos < sz; pos += chunk_bytes) {
            chunks.push_back(PreadChunk{fi, pos, std::min(sz, pos + chunk_bytes)});
        }
    }
    return chunks;
}

inline bool read_chunk_into(const PreadFile& f, const PreadChunk& c, std::vector<char>& buf,
                            PreadView& out) {
    static constexpr size_t kTailWindow = 4096;
    static constexpr size_t kPadBytes = 192;
    if (f.size == 0 || c.raw_start >= f.size) return false;

    const size_t read_base = (c.raw_start > 0) ? (c.raw_start - 1) : 0;
    const size_t stop_pos = c.raw_end - read_base;
    size_t read_end = std::min(f.size, c.raw_end + kTailWindow);
    size_t len = read_end - read_base;

    buf.resize(len + kPadBytes);
    pread_exact(f.fd, buf.data(), len, read_base);

    auto ensure_len = [&](size_t need_end) {
        if (need_end <= read_end) return;
        const size_t new_end = std::min(f.size, need_end);
        const size_t old_len = len;
        len = new_end - read_base;
        buf.resize(len + kPadBytes);
        pread_exact(f.fd, buf.data() + old_len, len - old_len, read_base + old_len);
        read_end = new_end;
    };

    size_t pos = 0;
    if (c.raw_start == 0) {
        size_t search_from = 0;
        const char* nl = nullptr;
        for (;;) {
            nl = static_cast<const char*>(
                std::memchr(buf.data() + search_from, '\n', len - search_from));
            if (nl || read_end >= f.size) break;
            search_from = len;
            ensure_len(read_end + 65536);
        }
        if (!nl) return false;
        pos = static_cast<size_t>(nl - buf.data()) + 1;
        if (pos >= stop_pos || pos >= len) return false;
    } else {
        const size_t scan_lim = std::min(len, stop_pos);
        const char* nl = static_cast<const char*>(std::memchr(buf.data(), '\n', scan_lim));
        if (!nl) return false;
        pos = static_cast<size_t>(nl - buf.data()) + 1;
        if (pos >= stop_pos) return false;
    }

    if (read_end < f.size) {
        size_t check_from = stop_pos - 1;
        while (!std::memchr(buf.data() + check_from, '\n', len - check_from)) {
            if (read_end >= f.size) break;
            check_from = len;
            ensure_len(read_end + 65536);
        }
    }

    std::memset(buf.data() + len, 0, kPadBytes);
    out = PreadView{buf.data(), pos, stop_pos, len};
    return true;
}

inline std::vector<char> read_whole_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("failed to open: " + path);
    struct stat st {};
    if (::fstat(fd, &st) != 0) {
        ::close(fd);
        throw std::runtime_error("failed to fstat: " + path);
    }
    const size_t sz = static_cast<size_t>(st.st_size);
    std::vector<char> data(sz);
    if (sz > 0) pread_exact(fd, data.data(), sz, 0);
    ::close(fd);
    return data;
}
