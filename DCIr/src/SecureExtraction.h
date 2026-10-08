#pragma once

#include "GraphException.h"
#include <filesystem>
#include <vector>
#include <string>
#include <random>
#include <cerrno>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <aclapi.h>
#else
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace DC::Ir::detail {

inline void extractionError(const char* message) {
	throw GraphException(GraphException::ErrorType::Other, "DcgArchive", message);
}

// Handles, not pre-open pathname checks, enforce the extraction boundary.
class SecureExtraction {
#ifdef _WIN32
	using Handle = HANDLE;
	static inline Handle invalid = INVALID_HANDLE_VALUE;
	static void close(Handle h) { if (h != invalid) CloseHandle(h); }
#else
	using Handle = int;
	static constexpr Handle invalid = -1;
	static void close(Handle h) { if (h != invalid) ::close(h); }
#endif
	Handle root = invalid;
public:
	std::filesystem::path path;
	SecureExtraction() {
		const auto base = std::filesystem::temp_directory_path();
		std::random_device random;
#ifdef _WIN32
		HANDLE token = nullptr;
		if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &token)) extractionError("cannot query current user token");
		DWORD size = 0;
		GetTokenInformation(token, TokenUser, nullptr, 0, &size);
		std::vector<unsigned char> user(size);
		const bool gotUser = GetTokenInformation(token, TokenUser, user.data(), size, &size) != FALSE;
		CloseHandle(token);
		if (!gotUser) extractionError("cannot query current user SID");
		EXPLICIT_ACCESSW access{};
		access.grfAccessPermissions = FILE_ALL_ACCESS;
		access.grfAccessMode = SET_ACCESS;
		access.grfInheritance = SUB_CONTAINERS_AND_OBJECTS_INHERIT;
		access.Trustee.TrusteeForm = TRUSTEE_IS_SID;
		access.Trustee.TrusteeType = TRUSTEE_IS_USER;
		access.Trustee.ptstrName = reinterpret_cast<LPWSTR>(reinterpret_cast<TOKEN_USER*>(user.data())->User.Sid);
		PACL acl = nullptr;
		if (SetEntriesInAclW(1, &access, nullptr, &acl) != ERROR_SUCCESS) extractionError("cannot create private DACL");
		SECURITY_DESCRIPTOR descriptor{};
		const bool secured = InitializeSecurityDescriptor(&descriptor, SECURITY_DESCRIPTOR_REVISION)
			&& SetSecurityDescriptorDacl(&descriptor, TRUE, acl, FALSE)
			&& SetSecurityDescriptorControl(&descriptor, SE_DACL_PROTECTED, SE_DACL_PROTECTED);
		if (!secured) { LocalFree(acl); extractionError("cannot protect private DACL"); }
		SECURITY_ATTRIBUTES attributes{sizeof(attributes), &descriptor, FALSE};
#endif
		bool made = false;
		for (int attempt = 0; attempt != 32; ++attempt) {
			path = base / ("dcg_private_" + std::to_string(random()) + "_" + std::to_string(random()));
#ifdef _WIN32
			if (CreateDirectoryW(path.c_str(), &attributes)) { made = true; break; }
			if (GetLastError() != ERROR_ALREADY_EXISTS) break;
#else
			if (::mkdir(path.c_str(), 0700) == 0) { made = true; break; }
			if (errno != EEXIST) break;
#endif
		}
#ifdef _WIN32
		LocalFree(acl);
#endif
		if (!made) extractionError("cannot atomically create private extraction directory");
#ifdef _WIN32
		root = openDirectory(path);
#else
		root = ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
#endif
		if (root == invalid) { std::error_code ec; std::filesystem::remove(path, ec); extractionError("cannot pin private extraction directory"); }
	}
	~SecureExtraction() { close(root); }
	SecureExtraction(const SecureExtraction&) = delete;
	SecureExtraction& operator=(const SecureExtraction&) = delete;
#ifdef _WIN32
	static Handle openDirectory(const std::filesystem::path& p) {
		// No FILE_SHARE_DELETE: every opened ancestor remains pinned while used.
		Handle h = CreateFileW(p.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE,
			nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
		if (h == invalid) return h;
		BY_HANDLE_FILE_INFORMATION info{};
		if (!GetFileInformationByHandle(h, &info) || !(info.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
			|| (info.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT)) { close(h); return invalid; }
		return h;
	}
#endif
	class Output {
		friend class SecureExtraction;
		Handle file = invalid;
		std::vector<Handle> ancestors;
	public:
		Output() = default;
		Output(const Output&) = delete;
		Output& operator=(const Output&) = delete;
		~Output() { close(file); for (auto h : ancestors) close(h); }
		void write(const char* data, std::size_t size) {
			while (size) {
#ifdef _WIN32
				DWORD n = 0;
				if (!WriteFile(file, data, static_cast<DWORD>(size), &n, nullptr) || !n) extractionError("write error to extraction target");
#else
				const auto n = ::write(file, data, size);
				if (n < 0 && errno == EINTR) continue;
				if (n <= 0) extractionError("write error to extraction target");
#endif
				data += n; size -= n;
			}
		}
	};
	void createOutput(const std::filesystem::path& relative, Output& output) {
		auto currentPath = path;
		Handle current = root;
		std::vector<std::filesystem::path> components;
		for (const auto& part : relative.lexically_normal()) if (part != "." && !part.empty()) components.push_back(part);
		if (components.empty()) extractionError("empty extraction target");
		for (std::size_t i = 0; i + 1 < components.size(); ++i) {
#ifdef _WIN32
			currentPath /= components[i];
			if (!CreateDirectoryW(currentPath.c_str(), nullptr) && GetLastError() != ERROR_ALREADY_EXISTS) extractionError("cannot create extraction ancestor");
			Handle next = openDirectory(currentPath);
#else
			if (::mkdirat(current, components[i].c_str(), 0700) != 0 && errno != EEXIST) extractionError("cannot create extraction ancestor");
			Handle next = ::openat(current, components[i].c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
#endif
			if (next == invalid) extractionError("symlink/reparse or inaccessible extraction ancestor");
			output.ancestors.push_back(next); current = next;
		}
#ifdef _WIN32
		currentPath /= components.back();
		output.file = CreateFileW(currentPath.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_NEW,
			FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
#else
		output.file = ::openat(current, components.back().c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
#endif
		// Exclusive creation also refuses existing symlinks, reparse points and hard links.
		if (output.file == invalid) extractionError("cannot exclusively create extraction target");
	}
};
} // namespace DC::Ir::detail
