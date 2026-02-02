#pragma once
#include <filesystem>

#include <functional>

#include "Tensor.h"

namespace DC {
	class InferBase;
	class Infer {
	public:
		using Task = std::unordered_map<std::string, Tensor>;
		using Response = std::unordered_map<std::string, Tensor>;
		using ConstructEngine = std::function<Infer* (std::vector<std::byte>, size_t)>;

		enum class ErrorCode {
			SUCCESS,
			MISSING_TENSOR,
			ERROR_TENSOR
		};

		virtual ~Infer() = default;
		virtual Response Run(Task& inputs) = 0;

		static std::unique_ptr<Infer> Create(
			const std::vector<std::byte>& modelData,
			size_t maxParallelCount = 1,
			const std::string& engine = ""
		);

		static std::unique_ptr<Infer> Create(
			const std::filesystem::path& modelPath,
			size_t maxParallelCount = 1,
			const std::string& engine = ""
		);

		static void addEngine(
			const std::string& name, 
			ConstructEngine factory
		);

		static std::unordered_set<std::string> getEngines();
	};
}