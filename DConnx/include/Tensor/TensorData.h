#pragma once
#include <map>
#include <unordered_set>
#include <string>
#include <memory>
#include <span>
#include <algorithm>
#include <type_traits>
#include <stdexcept>

#include "DCtype.h"

namespace DC
{
	// Todo：无拷贝移动写入

    // TensorData: 存储张量的底层数据容器
    // - 支持两种内部表示：稀疏块视图（_dataMain / _dataDimSets，称为 "view"）
    //   与连续稠密缓存（_dataCache，称为 "cache"）。
    // - view 表示为：按除最后一维外的坐标索引到一个字节块（DataBlock），
    //   每个块内按 _typeSize 解释元素；最后一维为块内元素数量（元素字节数 = _typeSize）。
    // - cache 表示为：完整连续的字节缓冲区，按稠密形状和 _typeSize 存放所有元素，
    //   未写入的元素以 0 填充。类在需要时会在两种表示之间物化（materialize）或构建缓存（flatten）。
	class TensorData {
	public:
		using Shape = std::vector<size_t>;
		using DataBlock = std::vector<std::byte>;
		using DataCatalog = std::vector<std::unordered_set<int64_t>>;
		using DataMap = std::map<Shape, DataBlock>;

		TensorData();
		TensorData(
			const Shape& shape,
			size_t typeSize,
			DataBlock&& denseBytes
		);
		TensorData(
			const Shape& shape,
			DataBlock&& data
		);

		bool hasView() const { return (_validFlags & FlagView) != 0; }
		bool hasCache() const { return (_validFlags & FlagCache) != 0; }

        // 获取当前数据的字节视图（优先返回稠密 cache，如果 cache 不存在则尝试从 view 构建 cache）。
        // 返回值：连续字节视图，表示稠密缓冲区的全部内容；若无数据则返回空的 span。
        // 备注：返回的是按字节的视图，不保证对任何对齐或类型重解释是安全的（调用方应使用 data<T>() 进行类型检查）。
        std::span<const std::byte> data() const;

        // 以类型 T 解释并返回稠密数据的视图。
        // 要求：T 为 trivially_copyable，且 _typeSize % sizeof(T) == 0。
        // 行为：若当前无稠密 cache，会尝试构建；若仍无数据则抛出或返回空（取决于 _typeSize 状态）。
        // 返回值：按 T 的元素数构造的 const span；元素数 = _dataCache.size() / sizeof(T)。
        template<typename T>
        std::span<const T> data() const;

        // 返回当前张量使用的总字节数（等同于稠密形状的元素数量 * _typeSize）。
        // 如果未设置 _typeSize 或无数据，返回 0。
        size_t size() const;

		// 设置类型字节数
		void setTypeSize(size_t typeSize);

        // 写入一个完整的块（block）到稀疏视图（view）。
        // 参数：
        //  - path: 块路径，长度等于张量秩 - 1（即不包含最后一维）；对于 0-D 标量写入，path 可为空且应使用 write(element) 重载。
        //  - data: 要写入的元素（按元素类型 T），其元素数量应等于最后一维的元素数（可小于或等于当前块大小，超出部分将扩展并用 0 填充）。
        // 返回：写入成功返回 true。若写入导致维度数量变化（path 长度与现有不同），会重置旧数据并以新维度初始化。
        template<typename T>
        bool write(const Shape& path, std::span<const T> data);

        // vector overload: 将 vector 拷贝为 span 后委托给上面的 write。
        template<typename T>
        bool write(const Shape& path, const std::vector<T>& data);

		bool write(const Shape& path, const std::vector<bool>& data);

        // 写入单个元素（按坐标全路径）。
        // 参数：
        //  - fullPath: 完整坐标路径，长度等于张量秩；最后一个元素为块内索引（element index）。
        //    传入空路径表示对 0-D 标量写入（会将张量重置为单元素标量，并保留/采用当前的 _typeSize）。
        //  - value: 要写入的值，类型为 T。若 sizeof(T) < _typeSize，则仅拷贝 sizeof(T) 字节并将其余字节清零；
        //    如果 sizeof(T) 不整除 _typeSize 的约束将由 validateAndSetTypeSize 检查（允许 _typeSize 为已有值且为 sizeof(T) 的倍数）。
        // 返回：写入成功返回 true。
        template<typename T>
        bool write(const Shape& fullPath, const T& value);

        // 读取稠密表示下的子范围或元素视图。
        // 参数：
        //  - path: 若长度 == rank，则表示读取单个元素（返回可能包含多个 T，取决于 _typeSize / sizeof(T)）；
        //    若长度 < rank，则表示按该前缀读取一个子张量（返回值包含后续维度展开的元素数 * (_typeSize / sizeof(T))）。
        // 返回：按 T 类型解释的 const span；若无数据则返回空 span。若请求越界或类型尺寸不匹配则抛出异常。
        template<typename T>
        std::span<const T> read(const Shape& path) const;

        // 读取单个元素的值（按类型 T）。若位置无数据，返回 T{}。
        template<typename T>
        T readElement(const Shape& fullPath) const;

        // 直接写入稠密缓存（_dataCache）中的区域。
        // 要求：当前对象必须处于 cache 模式（hasCache() == true），否则写入会失败或抛出。
        // 参数 path 的形式仅支持两种：
        //  - full element path（rank == shape.size()）：写入单元素（element count == 1）
        //  - block path（rank == shape.size() - 1）：写入整块（元素数 == shape.back()）
        // data 的字节大小必须精确匹配目标区域的字节数（element_count * _typeSize），否则抛出。
        template<typename T>
        bool writeCache(const Shape& path, const std::span<const T>& data);

        // vector overload for writeCache
        template<typename T>
        bool writeCache(const Shape& path, const std::vector<T>& data);

		template<typename T>
		bool writeCacheElement(const Shape& fullPath, const T& value);

		size_t typeSize() const { return _typeSize; }
		void clear();
		bool valid() const { return hasCache() || hasView(); }
		bool empty() const { return valid() && _dataCache.empty() && _dataMain.empty(); }
		bool isScalar() const { return _isScalar; }
		void setScalar(bool scalar = true) { _isScalar = scalar; }

		// 获取当前动态形状
		// 即稠密形状（最大索引 + 1）+ 数据块大小（元素数量）
		Shape getCurrentShape() const;

		// 直接设置为稠密（连续）数据。
		// 适用于外部已是稠密张量的场景（例如推理输出接收），避免数据块登记/拼装开销。
		// shape 为张量形状（元素维度），typeSize 为单元素字节数。
		void loadData(const Shape& shape, size_t typeSize, DataBlock&& bytes);

		// 显式进入可编辑模式：若当前为稠密直通模式，则会将稠密 bytes 物化为稀疏块映射。
		void editMode();

		// 取出数据
		DataBlock getData();

		template<typename T>
		TensorData& expand(const Shape& targetShape, const T& fillData);

		TensorData& crop(const Shape& targetShape);

	private:
        // 更新 _shapeCache / _dataSize / _size 等缓存元信息以匹配给定的稠密形状。
        // 参数 denseShape: 当前稠密表示的形状（最后一维为块内元素数）。
        // 影响：修改 _shapeCache、_dataSize、_size。
        void syncDenseCacheMeta(const Shape& denseShape);

        // 确保稠密缓存存在：若当前没有 cache 但有 view，则调用 buildFlattenedCache() 构建稠密缓存。
        // 行为：可能会改变 _dataCache、_shapeCache，并设置 FlagCache。此方法在 const 情况下通过 mutable 或 const_cast 被调用。
        void ensureCache();

        // 确保稀疏视图已物化（materialized）：若当前为 cache 模式且没有 view，则根据 cache 调用 materializeFromDense() 并设置 FlagView。
        // 影响：可能会填充 _dataMain、_dataCatalog，并设置 FlagView。
        void ensureView();

		DataMap _dataMain;
		DataCatalog _dataCatalog;
		size_t _typeSize; // 类型字节数
		size_t _dataSize; // 单个数据块的大小
		bool _isScalar;

		DataBlock _dataCache;
		Shape _shapeCache;
		static constexpr uint8_t FlagView = 0x1;
		static constexpr uint8_t FlagCache = 0x2;
		uint8_t _validFlags;

		
		void setViewFlag() { _validFlags |= FlagView; }
		void setCacheFlag() { _validFlags |= FlagCache; }
		template<typename T>
		DataBlock deposit(std::span<const T> data);

		DataBlock deposit(const std::vector<bool>& data);



		// Offset helpers (bytes)
		// - blockPath: rank-1 indices, pointing to a full last-dimension block
		// - elementPath: rank indices, pointing to a single element
		size_t blockOffset(const Shape& blockPath, const Shape& denseShape) const;
		size_t elementOffset(const Shape& elementPath, const Shape& denseShape) const;

		// 获取稠密形状
		Shape getDenseShape() const;

		// 构建稠密缓存：根据当前稀疏数据块映射和维度集合，构建一个完整的连续字节缓冲区（dataCache）表示稠密张量。未覆盖的元素填充为零。
		void buildCache();

		// 从稠密缓存物化为稀疏块映射：根据当前稠密缓存和形状，重建稀疏数据块映射（data）和维度集合（dataDimSets）。这会清空现有的稀疏结构，并将所有元素视为存在于一个完整的块中。
		void buildView();

        // 校验 _typeSize。
        // 参数 expectedSize: 期望的单个元素字节数（通常为 sizeof(T)）。
        bool checkType(size_t expectedSize, const std::string& callerName) const;

        // 校验写入路径并根据 path 更新 _dataCatalog（维度索引集合）。
        // 参数 path: 块路径（rank = tensor_rank - 1）。
        // 行为：若 path 长度与当前 _dataCatalog 不同，则会清空现有数据并按新的 path 长度重置维度集合；
        //       否则仅将 path 中的每个索引插入对应的维度集合。对负索引抛出 out_of_range。
        void updateCatalog(const Shape& path, const std::string& callerName);

        // 将给定的 DataBlock 提交到稀疏视图（_dataMain），并更新相关元信息。
        // 参数 path: 块路径；block: 要移动进来的字节块（右值引用）。
        // 行为：更新 _dataSize（若 block 更大），将 block 放入 _dataMain[path]，重新计算 _size，并清除稠密缓存（invalidate cache）。
        void commitData(const Shape& path, DataBlock&& block);

        // 计算稠密形状的元素总数（所有维度的乘积）。对于空形状（标量）返回 1。
        static size_t denseElementCount(const Shape& shape);
		
		void clearCache();

		void clearView();

        // 计算并返回稠密缓存中对应 path 的可写区域（按字节）。
        // 参数 path: 与 writeCache 的 path 语义一致（element path 或 block path）。
        // 返回：指向 _dataCache 中目标区域的可变 span；若 path 与 _shapeCache 不匹配或超出范围则抛出异常。
        std::span<std::byte> calcWriteRegion(const Shape& path);

        // 直接用 rawBytes 覆盖稠密缓存的目标区域（按字节）。
        // 参数:
        //  - path: 目标区域（参见 calcWriteRegion）。
        //  - rawBytes: 要写入的字节数据（右值引用，大小必须等于目标区域大小）。
        //  - typeSize: 提供的元素字节数，用于 validateAndSetTypeSize 校验。
        //  - apiName: 报错信息中使用的调用者名称。
        // 返回：写入成功返回 true；当不处于 cache 模式时返回 false（不会抛出）。
        bool writeCacheRaw(
            const Shape& path,
            DataBlock&& rawBytes,
            size_t typeSize,
            const char* apiName
        );

        // 辅助函数：将任意可转换为 DataBlock 的范围（Range）首先 deposit 为 DataBlock，然后调用 writeCacheRaw。
        // Range 可以是 std::span<const T>、std::vector<T> 或 std::vector<bool> 等。此函数负责调用 deposit 并传递右值到 writeCacheRaw。
        template<class Range>
        bool writeCacheByDeposit(
            const Shape& path,
            Range&& r,
            size_t typeSize,
            const char* apiName
        );
	};

	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	// Template Implementations: TensorData
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	template<typename T>
	std::span<const T> TensorData::data() const {
		static_assert(std::is_trivially_copyable_v<T>, "TensorData::data requires trivially copyable type");
		// Ensure dense cache is available for const reads: allow logical-const cache build
		if (!hasCache()) {
			const_cast<TensorData*>(this)->ensureCache();
		}
		
		checkType(sizeof(T), "TensorData::data");
		return std::span<const T>(reinterpret_cast<const T*>(_dataCache.data()), _dataCache.size() / sizeof(T));
	}

	template<typename T>
	bool TensorData::write(const Shape& path, std::span<const T> data) {
		static_assert(std::is_trivially_copyable_v<T>, "TensorData::write requires trivially copyable element type");

		if (!checkType(sizeof(T), "TensorData::write")) {
			setTypeSize(sizeof(T));
		}

		updateCatalog(path, "TensorData::write");
		ensureView();

		commitData(path, deposit(data));
		setViewFlag();
		return true;
	}

	inline bool TensorData::write(const Shape& path, const std::vector<bool>& data) {
		if (!checkType(sizeof(bool), "TensorData::write")) {
			setTypeSize(sizeof(bool));
		}

		updateCatalog(path, "TensorData::write");
		ensureView();

		commitData(path, deposit(data));
		return true;
	}

	template<typename T>
	bool TensorData::write(const Shape& path, const std::vector<T>& data) {
		return write(path, std::span<const T>(data.data(), data.size()));
	}

	template<typename T>
	bool TensorData::write(const Shape& fullPath, const T& value) {
        // Allow writing when stored element size is a multiple of incoming type size.
        // If _typeSize is not initialized, adopt sizeof(T). Otherwise require divisibility.
		if (!checkType(sizeof(T), "TensorData::write(element)")) {
			setTypeSize(sizeof(T));
		}

        // 0-D 标量处理
        if (fullPath.empty()) {
            // preserve validated _typeSize across clear()
            clear();
			setScalar(true);
			setTypeSize(sizeof(T));
			_dataSize = typeSize();

            DataBlock block(typeSize(), std::byte());
            // Copy only sizeof(T) bytes; zero the remaining bytes in the element slot if any.
            std::memcpy(block.data(), &value, sizeof(T));
            if (sizeof(T) < typeSize()) {
                std::memset(block.data() + sizeof(T), 0, typeSize() - sizeof(T));
            }
            commitData({}, std::move(block));

			setViewFlag();
            return true;
        }

		// 拆分出 blockPath
		Shape blockPath(fullPath.begin(), fullPath.end() - 1);
		size_t elementIndex = static_cast<size_t>(fullPath.back());

		// 复用路径准备和物化逻辑
		updateCatalog(blockPath, "TensorData::write(element)");
		ensureView();

		size_t targetBlockSize = (elementIndex + 1) * typeSize();
		// Guard against multiplication overflow when computing target block byte size
		if (typeSize() != 0) {
			size_t maxElems = std::numeric_limits<size_t>::max() / typeSize();
			if (elementIndex > maxElems) {
				throw std::out_of_range("TensorData::write(element): index too large");
			}
		}
		targetBlockSize = (elementIndex + 1) * typeSize();
		auto it = _dataMain.find(blockPath);
		DataBlock block;
		if (it == _dataMain.end()) {
			block.assign(targetBlockSize, std::byte());
		}
		else {
			block = it->second; // copy existing
			if (block.size() < targetBlockSize) block.resize(targetBlockSize, std::byte());
		}
        // Copy only sizeof(T) bytes into the element slot; zero remainder to avoid stale data.
        size_t elementOffset = elementIndex * typeSize();
        std::memcpy(block.data() + elementOffset, &value, sizeof(T));
        if (sizeof(T) < typeSize()) {
            std::memset(block.data() + elementOffset + sizeof(T), 0, typeSize() - sizeof(T));
        }
		commitData(blockPath, std::move(block));

		setScalar(false);
		clearCache();
		setViewFlag();

		return true;
	}

	template<typename T>
	std::span<const T> TensorData::read(const Shape& path) const {
		static_assert(std::is_trivially_copyable_v<T>, "TensorData::read requires trivially copyable type");

		if (!checkType(sizeof(T), "TensorData::read")) {
			throw std::runtime_error("TensorData::read: typeSize must be > 0 and a multiple of sizeof(T)");
		}

		const size_t ratio = typeSize() / sizeof(T);

		if (!hasCache()) {
			const_cast<TensorData*>(this)->ensureCache();
			if (!hasCache()) {
				return std::span<const T>(); // 无数据可读
			}
		}

		const auto& denseShape = _shapeCache;
		if (path.size() > denseShape.size()) {
			throw std::out_of_range("TensorData::readSpan: path rank exceeds tensor rank");
		}

		if (path.size() == denseShape.size()) {
			// 单元素：返回元素对应字节在新类型下的视图（可能为多个 T）
			for (size_t i = 0; i < path.size(); ++i) if (path[i] >= denseShape[i]) throw std::out_of_range("TensorData::readSpan: element index out of range");
			size_t offsetBytes = elementOffset(path, denseShape);
			return std::span<const T>(reinterpret_cast<const T*>(_dataCache.data() + offsetBytes), ratio);
		}

		// 前缀情况：补 0 或直接计算后缀乘积
		for (size_t i = 0; i < path.size(); ++i) if (path[i] >= denseShape[i]) throw std::out_of_range("TensorData::readSpan: element index out of range");
		size_t elementCount = 1;
		for (size_t i = path.size(); i < denseShape.size(); ++i) elementCount *= static_cast<size_t>(denseShape[i]);
		// 在新类型下的元素数量需要乘以 ratio
		size_t tElementCount = elementCount * ratio;

		Shape fullPath = path;
		fullPath.resize(denseShape.size(), 0);
		size_t offsetBytes = elementOffset(fullPath, denseShape);

		const size_t totalBytes = elementCount * typeSize();
		if (offsetBytes + totalBytes > _dataCache.size()) throw std::out_of_range("TensorData::readSpan: out of range");
		return std::span<const T>(reinterpret_cast<const T*>(_dataCache.data() + offsetBytes), tElementCount);
	}

	template<typename T>
	T TensorData::readElement(const Shape& fullPath) const {
		auto span = read<T>(fullPath);
		return span.empty() ? T{} : span[0];
	}

	template<typename T>
	bool TensorData::writeCache(const Shape& path, const std::span<const T>& data) {
		static_assert(std::is_trivially_copyable_v<T>,
			"TensorData::writeCache requires trivially copyable element type");
		return writeCacheByDeposit(path, data, sizeof(T), "TensorData::writeCache");
	}

	template<>
	inline bool TensorData::writeCache(const Shape& path, const std::vector<bool>& data) {
		return writeCacheByDeposit(path, data, sizeof(bool), "TensorData::writeCache");
	}

	template<typename T>
	bool TensorData::writeCache(const Shape& path, const std::vector<T>& data) {
		return writeCache(path, std::span<const T>(data.data(), data.size()));
	}

	template<typename T>
	bool TensorData::writeCacheElement(const Shape& fullPath, const T& value) {
		return writeCache(fullPath, std::span<const T>(&value, 1));
	}


	template<typename T>
	TensorData::DataBlock TensorData::deposit(std::span<const T> data) {
		static_assert(std::is_trivially_copyable_v<T>, "TensorData::deposit requires trivially copyable type");
		DataBlock charData(data.size() * sizeof(T));
		if (!data.empty()) {
			std::memcpy(charData.data(), data.data(), charData.size());
		}
		return charData;
	}

	inline TensorData::DataBlock TensorData::deposit(const std::vector<bool>& data) {
		DataBlock charData;
		charData.reserve(data.size());
		for (bool b : data) {
			charData.push_back(static_cast<std::byte>(b));
		}
		return charData;
	}


	template<class Range>
	bool TensorData::writeCacheByDeposit(const Shape& path, Range&& r, size_t typeSize, const char* apiName) {
		return writeCacheRaw(path, deposit(std::forward<Range>(r)), typeSize, apiName);
	}


	template<typename T>
	TensorData& TensorData::expand(const Shape& targetShape, const T& fillData) {
		static_assert(std::is_trivially_copyable_v<T>, "expand requires trivially copyable T");
		if (!checkType(sizeof(T), "TensorData::expand")) setTypeSize(sizeof(T));
		ensureView();

		auto current = getCurrentShape();
		if (current == targetShape) return *this;
		if (targetShape.empty()) { /* handle scalar case */ }
		// 增加这一行，防止非法收缩
		for (size_t i = 0; i < current.size(); ++i) {
			if (targetShape[i] < current[i]) {
				throw std::invalid_argument("TensorData::expand: target shape must be greater than or equal to current shape in each dimension");
			}
		}

		size_t rank = targetShape.size();
		size_t blockRank = (rank >= 1) ? rank - 1 : 0;
		size_t blockLen = targetShape.back();

		// iterate over all block paths (multi-index loop)
		Shape blockPath(blockRank, 0);
		bool done = (blockRank == 0); // zero-dim block space handled separately
		while (!done) {
			if (_dataMain.find(blockPath) == _dataMain.end()) {
				updateCatalog(blockPath, "TensorData::expand");
				std::vector<T> vals(blockLen, fillData);
				auto bytes = deposit(std::span<const T>(vals.data(), vals.size()));
				commitData(blockPath, std::move(bytes));
			}
			// increment blockPath lexicographically with carry
			for (size_t i = 0; i < blockRank; ++i) {
				if (++blockPath[i] < targetShape[i]) break;
				blockPath[i] = 0;
				if (i + 1 == blockRank) done = true;
			}
		}

		setViewFlag();
		clearCache(); // commitData may already clear cache, ensure consistency
		return *this;
	}
}