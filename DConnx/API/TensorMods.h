#pragma once
#include <vector>
#include <deque>
#include <map>
#include <unordered_set>
#include <string>
#include <memory>
#include <span>
#include <DCtype.h>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <mutex>

namespace DC {
	// 张量元数据
	struct TensorMeta {
		enum class TensorType {
			Float,
			Int,
			Uint,
			Bool,
			Char,
			Data,
			Void
		};

		enum class DataState {
			Overflow,
			TypeError,
			Empty,
			Ready
		};

		enum class MismatchPolicy {
			Auto,
			Truncate,
			Bitcast,
			Convert,
			Throw
		};
		
	public:
		TensorMeta();

		static void ensureTypeMap();

		std::string name = "";
		size_t typeSize = 0;
		// 规则形状（RuleShape）。若为空，则不进行形状检查。
		// 约定：某一维为 -1 表示该维度为动态维度，check() 时跳过该维度的比较。
		std::vector<int64_t> shape = {};
		TensorType type = TensorType::Void;

		// 检查形状是否符合规则
		bool check(const std::vector<int64_t>& currentShape) const;

	private:
		static void setTypeMap();
	};

	

	// 实际储存张量数据的类
	// 数据块一维平摊储存
	class TensorData {
	public:
		using DataDimSets = std::vector<std::unordered_set<uint64_t>>;
		using DataBlock = std::vector<char>;
		using DataMap = std::map<std::vector<int64_t>, DataBlock>;

		TensorData(); 
		TensorData(
			const std::vector<int64_t>& shape,
			size_t typeSize,
			std::vector<char>&& denseBytes
		);
		TensorData(
			const std::vector<int64_t>& shape,
			std::vector<char>&& data
		);

		bool cacheValid() const;

		std::span<const char> dataSpan() const;

		std::span<char> dataSpanMut();

		template<typename T>
		std::span<const T> dataSpanAs() const;

		template<typename T>
		std::span<T> dataSpanAsMut();

		size_t size() const;

		DataDimSets getDataDimSets() const;

		// 设置类型字节数
		// 这会重新解释张量形状
		void setTypeSize(size_t typeSize);

		// 写入数据
		// 如果写入索引会改变维度数量，则清空旧数据，当作全新张量写入
		template<typename T>
		bool write(std::vector<int64_t> path, const std::vector<T>& data);

		// 按规则 typeSize 写入 bytes（不改变张量规则；仅影响解释方式）
		bool writeBytes(std::vector<int64_t> path, size_t ruleTypeSize, const std::vector<char>& bytes);

		// vector<T> 作为 bytes 来源，按位拷贝并按 ruleTypeSize 解释。
		template<typename T>
		bool writeBitcast(std::vector<int64_t> path, size_t ruleTypeSize, const std::vector<T>& data);

		// 读取数据
		template<typename T>
		std::vector<T> read(std::vector<int64_t> path) const;

		template<typename T>
		std::span<const T> readSpan(const std::vector<int64_t>& path) const;

		template<typename T>
		std::span<const T> readSpan(const std::vector<int64_t>& path);

		template<typename T>
		bool writeElement(const std::vector<int64_t>& fullPath, const T& value);

		template<typename T>
		T readElement(const std::vector<int64_t>& fullPath) const;

		// 清空数据
		void clear();

		bool isScalar() const;
		void setScalar(bool scalar);

		// 获取当前动态形状
		// 即稠密形状（最大索引 + 1）+ 数据块大小（元素数量）
		// 注意：在 setDenseBytes() 的“稠密直通模式”下，不会构建稀疏块映射，形状直接来自 _flattenedCacheShape。
		std::vector<int64_t> getCurrentShape() const;

		// 直接设置为稠密（连续）数据。
		// 适用于外部已是稠密张量的场景（例如推理输出接收），避免数据块登记/拼装开销。
		// shape 为张量形状（元素维度），typeSize 为单元素字节数。
		void setDenseBytes(const std::vector<int64_t>& shape, size_t typeSize, std::vector<char>&& bytes);

		// 显式进入可编辑模式：若当前为稠密直通模式，则会将稠密 bytes 物化为稀疏块映射。
		void ensureEditable();

		// 密集直通模式下的快速标量写入。
		// 在展平的稠密字节缓冲区中更新单个元素，而不具体化稀疏块。
		// path 是完整的元素索引路径（与 Tensor::View::path() 的语义相同）：
		// - 如果它与形状维度匹配，则最后一个索引被视为最后一个维度块内的元素索引。
		// - 否则，它指的是一个完整的块路径（此处不支持）。
		bool writeScalarDenseBytes(const std::vector<int64_t>& path, size_t ruleTypeSize, const void* scalarBytes, size_t scalarByteCount);

	private:
		class CacheScope {
		public:
			enum class Mode {
				DenseEdit,
				SparseEdit
			};
			enum class DensePolicy {
				EnsureCache,
				RequireCache
			};

			CacheScope(TensorData& owner, Mode mode, DensePolicy policy = DensePolicy::EnsureCache);

			~CacheScope();

			bool active() const;

		private:
			TensorData* _owner = nullptr;
			Mode _mode;
			bool _active = true;
		};

		void syncDenseCacheMeta(const std::vector<int64_t>& denseShape);

		void invalidateDenseCache();

		void ensureDenseCache();

		void ensureSparseEditable();

		DataMap _data;
		DataDimSets _dataDimSets;
		size_t _size = 0;
		size_t _typeSize = 0; // 类型字节数
		size_t _dataSize = 0; // 单个数据块的大小
		bool _isScalar = false;

		std::vector<char> _dataCache;
		std::vector<int64_t> _shapeCache;
		bool _cacheValid = false;
		bool _sparseValid = true;

		// 强制类型转换
		// 将输入的数据块转换为 char 类型
		template<typename T>
		DataBlock toCharVector(const std::vector<T>& data);

		template<>
		DataBlock toCharVector<bool>(const std::vector<bool>& data);

		// 从数据块中恢复指定类型的数据
		template<typename T>
		std::vector<T> fromCharVector(const DataBlock& block) const;

		template<>
		std::vector<bool> fromCharVector<bool>(const DataBlock& block) const;

		// 考虑稀疏张量，计算总大小
		// 取 DataDimSets 中每个维度的最大值，与数据块大小相乘
		size_t calculateTotalSize() const;

		// Offset helpers (bytes)
		// - blockPath: rank-1 indices, pointing to a full last-dimension block
		// - elementPath: rank indices, pointing to a single element
		size_t calculateBlockOffsetBytes(const std::vector<int64_t>& blockPath, const std::vector<int64_t>& denseShape) const;
		size_t calculateElementOffsetBytes(const std::vector<int64_t>& elementPath, const std::vector<int64_t>& denseShape) const;

		// 获取稠密形状
		std::vector<int64_t> getDenseShape() const;

		void buildFlattenedCache();

		void materializeFromDense();
	};

	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	// Template Implementations: TensorData
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	template<typename T>
	std::span<const T> TensorData::dataSpanAs() const {
		if (!_cacheValid) {
			return {};
		}
		if (sizeof(T) == 0 || (_typeSize != sizeof(T))) {
			return {};
		}
		if ((_dataCache.size() % sizeof(T)) != 0) {
			return {};
		}
		return std::span<const T>(reinterpret_cast<const T*>(_dataCache.data()), _dataCache.size() / sizeof(T));
	}

	template<typename T>
	std::span<T> TensorData::dataSpanAsMut() {
		dataSpanMut();
		if (sizeof(T) == 0 || (_typeSize != sizeof(T))) {
			return {};
		}
		if ((_dataCache.size() % sizeof(T)) != 0) {
			return {};
		}
		return std::span<T>(reinterpret_cast<T*>(_dataCache.data()), _dataCache.size() / sizeof(T));
	}

	template<typename T>
	bool TensorData::write(std::vector<int64_t> path, const std::vector<T>& data) {
		CacheScope scope(*this, CacheScope::Mode::SparseEdit);
		if (path.size() != _dataDimSets.size()) {
			clear();
			_dataDimSets.resize(path.size());
		}

		for (int index = 0; index < path.size();++index) {

			if (path[index] < 0) return false;
			_dataDimSets[index].insert(path[index]);
		}

		{
			const auto it = _data.find(path);
			if (it != _data.end()) {
				_size -= it->second.size();
			}
			_data[path] = toCharVector(data);
			_size += _data[path].size();
			if (_dataSize < _data[path].size()) {
				_dataSize = _data[path].size();
			}
		}

		return true;
	}

	template<typename T>
	bool TensorData::writeBitcast(std::vector<int64_t> path, size_t ruleTypeSize, const std::vector<T>& data) {
		if (ruleTypeSize == 0) {
			throw std::invalid_argument("TensorData::writeBitcast: ruleTypeSize must be > 0");
		}
		const size_t totalBytes = data.size() * sizeof(T);
		if (totalBytes % ruleTypeSize != 0) {
			throw std::invalid_argument("TensorData::writeBitcast: byte size is not divisible by ruleTypeSize");
		}
		auto bytes = toCharVector(data);
		return writeBytes(std::move(path), ruleTypeSize, bytes);
	}

	// Note: TensorData::read is declared as const; no non-const overload.


	template<typename T>
	std::span<const T> TensorData::readSpan(const std::vector<int64_t>& path) const {
		if (sizeof(T) == 0 || _typeSize != sizeof(T)) {
			throw std::invalid_argument("TensorData::readSpan: type size mismatch");
		}
		// Contract: missing blocks/elements read as zero-filled.
		// Const-read does not mutate/cache-rebuild. Require dense cache to be valid.
		if (!_cacheValid) {
			throw std::runtime_error("TensorData::readSpan: dense cache is not available in const context");
		}
		const auto& denseShape = _shapeCache;
		if (path.size() > denseShape.size()) {
			throw std::out_of_range("TensorData::readSpan: path rank exceeds tensor rank");
		}

		size_t elementCount = 0;
		size_t offsetBytes = 0;
		if (path.size() == denseShape.size()) {
			elementCount = 1;
			offsetBytes = calculateElementOffsetBytes(path, denseShape);
		}
		else if (path.size() + 1 == denseShape.size()) {
			elementCount = static_cast<size_t>(denseShape.back());
			offsetBytes = calculateBlockOffsetBytes(path, denseShape);
		}
		else {
			throw std::out_of_range("TensorData::readSpan: unsupported slice form");
		}

		const size_t totalBytes = elementCount * _typeSize;
		if (offsetBytes + totalBytes > _dataCache.size()) {
			throw std::out_of_range("TensorData::readSpan: offset out of range");
		}

		return std::span<const T>(
			reinterpret_cast<const T*>(_dataCache.data() + offsetBytes),
			elementCount);
	}

	template<typename T>
	std::span<const T> TensorData::readSpan(const std::vector<int64_t>& path) {
		if (sizeof(T) == 0 || _typeSize != sizeof(T)) {
			throw std::invalid_argument("TensorData::readSpan: type size mismatch");
		}
		// Contract: missing blocks/elements read as zero-filled.
		// Non-const read may rebuild dense cache.
		ensureDenseCache();
		const auto& denseShape = _shapeCache;
		if (path.size() > denseShape.size()) {
			throw std::out_of_range("TensorData::readSpan: path rank exceeds tensor rank");
		}

		size_t elementCount = 0;
		size_t offsetBytes = 0;
		if (path.size() == denseShape.size()) {
			elementCount = 1;
			offsetBytes = calculateElementOffsetBytes(path, denseShape);
		}
		else if (path.size() + 1 == denseShape.size()) {
			elementCount = static_cast<size_t>(denseShape.back());
			offsetBytes = calculateBlockOffsetBytes(path, denseShape);
		}
		else {
			throw std::out_of_range("TensorData::readSpan: unsupported slice form");
		}

		const size_t totalBytes = elementCount * _typeSize;
		if (offsetBytes + totalBytes > _dataCache.size()) {
			throw std::out_of_range("TensorData::readSpan: offset out of range");
		}

		return std::span<const T>(
			reinterpret_cast<const T*>(_dataCache.data() + offsetBytes),
			elementCount);
	}

	template<typename T>
	bool TensorData::writeElement(const std::vector<int64_t>& fullPath, const T& value) {
		// 类型大小检查
		if (sizeof(T) != _typeSize) return false;
		if (_dataSize == 0) return false;  // 块大小未定义

		// 确保可编辑（稠密 → 稀疏）
		ensureEditable();

		auto currentShape = getCurrentShape();
		if (fullPath.size() != currentShape.size()) return false;

		// 标量特殊处理
		if (fullPath.empty()) {
			// 标量：直接操作 _data[{}] 块
			auto it = _data.find({});
			if (it == _data.end()) {
				DataBlock newBlock(_typeSize, 0);
				std::memcpy(newBlock.data(), &value, _typeSize);
				_data[{}] = std::move(newBlock);
				_size += _typeSize;
			}
			else {
				DataBlock& block = it->second;
				if (block.size() != _typeSize) return false;
				std::memcpy(block.data(), &value, _typeSize);
			}
			return true;
		}

		// 分离块路径和元素索引
		std::vector<int64_t> blockPath(fullPath.begin(), fullPath.end() - 1);
		int64_t elemIdx = fullPath.back();

		// 验证元素索引范围
		int64_t lastDimSize = currentShape.back();
		if (elemIdx < 0 || elemIdx >= lastDimSize) return false;

		// 更新维度集合
		for (size_t i = 0; i < blockPath.size(); ++i) {
			if (blockPath[i] < 0) return false;
			_dataDimSets[i].insert(static_cast<uint64_t>(blockPath[i]));
		}

		// 查找或创建块
		auto it = _data.find(blockPath);
		if (it == _data.end()) {
			DataBlock newBlock(_dataSize, 0);
			size_t offset = static_cast<size_t>(elemIdx) * _typeSize;
			std::memcpy(newBlock.data() + offset, &value, _typeSize);
			_data[blockPath] = std::move(newBlock);
			_size += _dataSize;
		}
		else {
			DataBlock& block = it->second;
			if (block.size() > _dataSize) {
				throw std::runtime_error("TensorData::writeElement: block size exceeds current rule block size");
			}
			if (block.size() < _dataSize) {
				const auto oldSize = block.size();
				block.resize(_dataSize, 0);
				_size += (_dataSize - oldSize);
			}
			size_t offset = static_cast<size_t>(elemIdx) * _typeSize;
			if (offset + _typeSize > block.size()) return false;
			std::memcpy(block.data() + offset, &value, _typeSize);
		}

		invalidateDenseCache();
		return true;
	}

	template<typename T>
	T TensorData::readElement(const std::vector<int64_t>& fullPath) const {
        if (sizeof(T) != _typeSize)
            throw std::invalid_argument("Type size mismatch");

        // Sparse-backed read: locate the block and reconstruct element from block bytes
        if (_sparseValid) {
            // Scalar special case
            if (fullPath.empty()) {
                auto it = _data.find({});
                if (it == _data.end()) throw std::out_of_range("Element not found");
                const DataBlock& block = it->second;
                if (block.size() < _typeSize) throw std::out_of_range("Block size too small");
                T value;
                std::memcpy(&value, block.data(), _typeSize);
                return value;
            }

            // Split block path and element index
            std::vector<int64_t> blockPath(fullPath.begin(), fullPath.end() - 1);
            int64_t elemIdx = fullPath.back();

            const auto it = _data.find(blockPath);
            if (it == _data.end()) throw std::out_of_range("Block not found");
            const DataBlock& block = it->second;

            const size_t offset = static_cast<size_t>(elemIdx) * _typeSize;
            if (offset + _typeSize > block.size()) throw std::out_of_range("Element offset out of range");
            T value;
            std::memcpy(&value, block.data() + offset, _typeSize);
            return value;
        }

        // Dense cache-backed read
        if (!_cacheValid) {
            throw std::runtime_error("TensorData::readElement: dense cache is not available in const context");
        }
        const auto& shape = _shapeCache;
        if (fullPath.size() != shape.size()) {
            throw std::out_of_range("Path rank mismatch");
        }
        size_t offset = calculateElementOffsetBytes(fullPath, shape);
        if (offset + _typeSize > _dataCache.size()) {
            throw std::out_of_range("Offset out of range");
        }
        T value;
        std::memcpy(&value, _dataCache.data() + offset, _typeSize);
        return value;
	}

	template<typename T>
	TensorData::DataBlock TensorData::toCharVector(const std::vector<T>& data) {
		DataBlock charData(data.size() * sizeof(T));
		std::memcpy(charData.data(), data.data(), charData.size());
		return charData;
	}

	template<>
	inline TensorData::DataBlock TensorData::toCharVector<bool>(const std::vector<bool>& data) {
		DataBlock charData;
		charData.reserve(data.size());
		for (bool b : data) {
			charData.push_back(static_cast<char>(b));
		}
		return charData;
	}

	template<typename T>
	std::vector<T> TensorData::fromCharVector(const DataBlock& block) const {
		std::vector<T> data(block.size() / sizeof(T));
		std::memcpy(data.data(), block.data(), block.size());
		return data;
	}

	template<>
	inline std::vector<bool> TensorData::fromCharVector<bool>(const DataBlock& block) const {
		std::vector<bool> data;
		data.reserve(block.size());
		for (char c : block) {
			data.push_back(c != 0);
		}
		return data;
	}
}