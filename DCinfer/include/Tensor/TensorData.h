#pragma once
#include <atomic>
#include <map>
#include <unordered_set>
#include <string>
#include <vector>
#include <cstddef>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <algorithm>
#include <type_traits>
#include <stdexcept>

#include "DCtype.h"
#include "TensorException.h"

namespace DC {

// 存储张量的底层数据容器：两种内部表示——稀疏块视图（view：按除最后一维外的
// 坐标索引到字节块，块内按 _typeSize 解释元素）与连续稠密缓存（cache：按稠密
// 形状存放全部元素，未写入处补零）；按需在两种表示间物化。
class TensorData {
public:
	using Shape = std::vector<size_t>;
	using DataBlock = std::vector<std::byte>;
	using DataCatalog = std::vector<std::unordered_set<int64_t>>;
	using DataMap = std::map<Shape, DataBlock>;

	TensorData();
	TensorData(const Shape& shape, size_t typeSize, DataBlock&& denseBytes);
	TensorData(const Shape& shape, DataBlock&& data);

	/// @brief 拷贝构造：深拷贝数据，产出非冻结副本。
	TensorData(const TensorData& other);
	/// @brief 拷贝赋值：深拷贝数据，目标变为非冻结副本。
	TensorData& operator=(const TensorData& other);
	/// @brief 移动构造：接管资源（冻结状态随身份转移）。
	TensorData(TensorData&& other) noexcept;
	/// @brief 移动赋值：接管资源（冻结状态随身份转移）。
	TensorData& operator=(TensorData&& other) noexcept;

	/// @brief 冻结：预物化稠密缓存并置冻结位；冻结后写路径抛 TensorException(Frozen)，只读不再触发惰性物化。
	void freeze();

	/// @brief 是否已冻结。
	bool isFrozen() const {
		return _frozen;
	}

	bool hasView() const {
		return (_validFlags.load(std::memory_order_acquire) & FlagView) != 0;
	}
	bool hasCache() const {
		return (_validFlags.load(std::memory_order_acquire) & FlagCache) != 0;
	}

	// 获取稠密字节视图（cache 缺失时从 view 构建）；类型解释安全性由 data<T>() 保证。
	std::span<const std::byte> data() const;

	// 以类型 T 解释稠密数据（要求 trivially_copyable 且 typeSize % sizeof(T) == 0）。
	template <typename T>
	std::span<const T> data() const;

	// 当前张量总字节数（稠密形状元素数 * _typeSize）；无数据返回 0。
	size_t size() const;

	void setTypeSize(size_t typeSize);

	// 写入完整块到稀疏视图：path 长度 = 秩 - 1（0-D 标量用 write(element) 重载）。
	template <typename T>
	bool write(const Shape& path, std::span<const T> data);

	template <typename T>
	bool write(const Shape& path, const std::vector<T>& data);

	bool write(const Shape& path, const std::vector<bool>& data);

	// 写入单个元素（fullPath 长度 = 秩，最后一项为块内索引；空路径 = 0-D 标量写入）。
	template <typename T>
	bool write(const Shape& fullPath, const T& value);

	// 读取子范围或元素视图（path 长度 == 秩读单元素；< 秩按前缀读子张量）。
	template <typename T>
	std::span<const T> read(const Shape& path) const;

	// 读取单个元素；无数据返回 T{}。
	template <typename T>
	T readElement(const Shape& fullPath) const;

	// 写入稠密缓存区域（须处于 cache 模式；path 为元素路径或块路径，字节数须精确匹配）。
	template <typename T>
	bool writeCache(const Shape& path, const std::span<const T>& data);

	template <typename T>
	bool writeCache(const Shape& path, const std::vector<T>& data);

	template <typename T>
	bool writeCacheElement(const Shape& fullPath, const T& value);

	size_t typeSize() const {
		return _typeSize;
	}
	void clear();
	bool valid() const {
		return hasCache() || hasView();
	}
	bool empty() const {
		return valid() && _dataCache.empty() && _dataMain.empty();
	}
	bool isScalar() const {
		return _isScalar;
	}
	void setScalar(bool scalar = true) {
		_isScalar = scalar;
	}

	// 当前动态形状：稠密形状（最大索引 + 1）+ 块内元素数。
	Shape getCurrentShape() const;

	// 直接装入稠密数据（外部已是稠密张量时避免逐块登记开销）。
	void loadData(const Shape& shape, size_t typeSize, DataBlock&& bytes);

	// 进入可编辑模式：稠密直通数据物化为稀疏块映射。
	void editMode();

	DataBlock getData();

	template <typename T>
	TensorData& expand(const Shape& targetShape, const T& fillData);

	TensorData& crop(const Shape& targetShape);

private:
	/// @brief 冻结门校验；仅覆盖载荷写路径（对象级赋值不在其列）。
	void _ensureMutable(const char* api) const;

	// 按稠密形状更新 _shapeCache / _dataSize / _size。
	void syncDenseCacheMeta(const Shape& denseShape);

	// 确保稠密缓存存在（view → cache）；const 路径经 const_cast 调用。
	void ensureCache();

	// 确保稀疏视图已物化（cache → view）。
	void ensureView();

	DataMap _dataMain;
	DataCatalog _dataCatalog;
	size_t _typeSize;
	size_t _dataSize;
	bool _isScalar;

	DataBlock _dataCache;
	Shape _shapeCache;
	static constexpr uint8_t FlagView = 0x1;
	static constexpr uint8_t FlagCache = 0x2;
	// 有效标志位（release/acquire 配对：构建者写标志的 release 保证数据对快路径可见）。
	std::atomic<uint8_t> _validFlags{0};

	bool _frozen = false;

	// 惰性物化串行化锁（双重检查构建）。
	mutable std::mutex _lazyMutex;

	void setViewFlag() {
		_validFlags.fetch_or(FlagView, std::memory_order_release);
	}
	void setCacheFlag() {
		_validFlags.fetch_or(FlagCache, std::memory_order_release);
	}
	template <typename T>
	DataBlock deposit(std::span<const T> data);

	DataBlock deposit(const std::vector<bool>& data);

	// 字节偏移辅助：blockPath = rank-1 索引（整块）；elementPath = rank 索引（单元素）。
	size_t blockOffset(const Shape& blockPath, const Shape& denseShape) const;
	size_t elementOffset(const Shape& elementPath, const Shape& denseShape) const;

	Shape getDenseShape() const;

	// 从稀疏块映射构建完整连续字节缓冲区（未覆盖元素补零）。
	void buildCache();

	// 从稠密缓存物化为稀疏块映射（清空并重建 _dataMain/_dataCatalog，视为单个完整块）。
	void buildView();

	// 校验 _typeSize（expectedSize 通常为 sizeof(T)）。
	bool checkType(size_t expectedSize, const std::string& callerName) const;

	// 按块路径更新 _dataCatalog；path 长度变化时清空数据并重置维度集合。
	void updateCatalog(const Shape& path, const std::string& callerName);

	// 提交字节块到稀疏视图并更新元信息（清除稠密缓存）。
	void commitData(const Shape& path, DataBlock&& block);

	// 稠密形状元素总数（空形状返回 1）。
	static size_t denseElementCount(const Shape& shape);

	void clearCache();

	void clearView();

	// 稠密缓存中 path 对应区域的可写字节 span（越界抛异常）。
	std::span<std::byte> calcWriteRegion(const Shape& path);

	// 用 rawBytes 覆盖稠密缓存目标区域；非 cache 模式返回 false（不抛出）。
	bool writeCacheRaw(const Shape& path, DataBlock&& rawBytes, size_t typeSize, const char* apiName);

	// 将 Range deposit 为 DataBlock 后转调 writeCacheRaw。
	template <class Range>
	bool writeCacheByDeposit(const Shape& path, Range&& r, size_t typeSize, const char* apiName);
};

template <typename T>
std::span<const T> TensorData::data() const {
	static_assert(std::is_trivially_copyable_v<T>, "TensorData::data requires trivially copyable type");
	// const 读路径允许惰性构建缓存
	if (!hasCache()) {
		const_cast<TensorData*>(this)->ensureCache();
	}

	checkType(sizeof(T), "TensorData::data");
	return std::span<const T>(reinterpret_cast<const T*>(_dataCache.data()), _dataCache.size() / sizeof(T));
}

template <typename T>
bool TensorData::write(const Shape& path, std::span<const T> data) {
	static_assert(std::is_trivially_copyable_v<T>, "TensorData::write requires trivially copyable element type");
	_ensureMutable("TensorData::write");

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
	_ensureMutable("TensorData::write");
	if (!checkType(sizeof(bool), "TensorData::write")) {
		setTypeSize(sizeof(bool));
	}

	updateCatalog(path, "TensorData::write");
	ensureView();

	commitData(path, deposit(data));
	return true;
}

template <typename T>
bool TensorData::write(const Shape& path, const std::vector<T>& data) {
	return write(path, std::span<const T>(data.data(), data.size()));
}

template <typename T>
bool TensorData::write(const Shape& fullPath, const T& value) {
	_ensureMutable("TensorData::write");
	// 未初始化时采用 sizeof(T)，否则要求整除关系
	if (!checkType(sizeof(T), "TensorData::write(element)")) {
		setTypeSize(sizeof(T));
	}

	if (fullPath.empty()) {
		clear();
		setScalar(true);
		setTypeSize(sizeof(T));
		_dataSize = typeSize();

		DataBlock block(typeSize(), std::byte());
		std::memcpy(block.data(), &value, sizeof(T));
		if (sizeof(T) < typeSize()) {
			std::memset(block.data() + sizeof(T), 0, typeSize() - sizeof(T));
		}
		commitData({}, std::move(block));

		setViewFlag();
		return true;
	}

	Shape blockPath(fullPath.begin(), fullPath.end() - 1);
	size_t elementIndex = static_cast<size_t>(fullPath.back());

	// 乘法回绕防护（先检查再改动 catalog）：elementIndex >= SIZE_MAX/typeSize 时
	// (元素数+1)*typeSize 回绕，后续 memcpy 越界写。
	if (typeSize() == 0)
		throw std::out_of_range("TensorData::write(element): typeSize not initialized");
	{
		const size_t maxElems = std::numeric_limits<size_t>::max() / typeSize();
		if (elementIndex >= maxElems)
			throw std::out_of_range("TensorData::write(element): index too large");
	}
	const size_t targetBlockSize = (elementIndex + 1) * typeSize();

	updateCatalog(blockPath, "TensorData::write(element)");
	ensureView();
	auto it = _dataMain.find(blockPath);
	DataBlock block;
	if (it == _dataMain.end()) {
		block.assign(targetBlockSize, std::byte());
	} else {
		block = it->second; // 拷贝已有块
		if (block.size() < targetBlockSize)
			block.resize(targetBlockSize, std::byte());
	}
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

template <typename T>
std::span<const T> TensorData::read(const Shape& path) const {
	static_assert(std::is_trivially_copyable_v<T>, "TensorData::read requires trivially copyable type");

	if (!checkType(sizeof(T), "TensorData::read")) {
		throw std::runtime_error("TensorData::read: typeSize must be > 0 and a multiple of sizeof(T)");
	}

	const size_t ratio = typeSize() / sizeof(T);

	if (!hasCache()) {
		const_cast<TensorData*>(this)->ensureCache();
		if (!hasCache()) {
			return std::span<const T>();
		}
	}

	const auto& denseShape = _shapeCache;
	if (path.size() > denseShape.size()) {
		throw std::out_of_range("TensorData::readSpan: path rank exceeds tensor rank");
	}

	if (path.size() == denseShape.size()) {
		for (size_t i = 0; i < path.size(); ++i)
			if (path[i] >= denseShape[i])
				throw std::out_of_range("TensorData::readSpan: element index out of range");
		size_t offsetBytes = elementOffset(path, denseShape);
		// 纵深防御：声明与数据不一致的脏数据在此显式暴露，而非静默越界读
		const size_t spanBytes = ratio * sizeof(T);
		if (offsetBytes + spanBytes > _dataCache.size())
			throw std::out_of_range("TensorData::readSpan: element view exceeds dense cache size");
		return std::span<const T>(reinterpret_cast<const T*>(_dataCache.data() + offsetBytes), ratio);
	}

	for (size_t i = 0; i < path.size(); ++i)
		if (path[i] >= denseShape[i])
			throw std::out_of_range("TensorData::readSpan: element index out of range");
	size_t elementCount = 1;
	for (size_t i = path.size(); i < denseShape.size(); ++i)
		elementCount *= static_cast<size_t>(denseShape[i]);
	size_t tElementCount = elementCount * ratio;

	Shape fullPath = path;
	fullPath.resize(denseShape.size(), 0);
	size_t offsetBytes = elementOffset(fullPath, denseShape);

	const size_t totalBytes = elementCount * typeSize();
	if (offsetBytes + totalBytes > _dataCache.size())
		throw std::out_of_range("TensorData::readSpan: out of range");
	return std::span<const T>(reinterpret_cast<const T*>(_dataCache.data() + offsetBytes), tElementCount);
}

template <typename T>
T TensorData::readElement(const Shape& fullPath) const {
	auto span = read<T>(fullPath);
	return span.empty() ? T{} : span[0];
}

template <typename T>
bool TensorData::writeCache(const Shape& path, const std::span<const T>& data) {
	static_assert(std::is_trivially_copyable_v<T>, "TensorData::writeCache requires trivially copyable element type");
	return writeCacheByDeposit(path, data, sizeof(T), "TensorData::writeCache");
}

template <>
inline bool TensorData::writeCache(const Shape& path, const std::vector<bool>& data) {
	return writeCacheByDeposit(path, data, sizeof(bool), "TensorData::writeCache");
}

template <typename T>
bool TensorData::writeCache(const Shape& path, const std::vector<T>& data) {
	return writeCache(path, std::span<const T>(data.data(), data.size()));
}

template <typename T>
bool TensorData::writeCacheElement(const Shape& fullPath, const T& value) {
	return writeCache(fullPath, std::span<const T>(&value, 1));
}

template <typename T>
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

template <class Range>
bool TensorData::writeCacheByDeposit(const Shape& path, Range&& r, size_t typeSize, const char* apiName) {
	return writeCacheRaw(path, deposit(std::forward<Range>(r)), typeSize, apiName);
}

template <typename T>
TensorData& TensorData::expand(const Shape& targetShape, const T& fillData) {
	static_assert(std::is_trivially_copyable_v<T>, "expand requires trivially copyable T");
	_ensureMutable("TensorData::expand");
	if (!checkType(sizeof(T), "TensorData::expand"))
		setTypeSize(sizeof(T));
	ensureView();

	auto current = getCurrentShape();
	if (current == targetShape)
		return *this;
	// 目标秩必须与当前一致
	if (targetShape.size() != current.size()) {
		throw std::invalid_argument(
			"TensorData::expand: rank mismatch (target rank must equal current rank)");
	}
	for (size_t i = 0; i < current.size(); ++i) {
		if (targetShape[i] < current[i]) {
			throw std::invalid_argument(
				"TensorData::expand: target shape must be greater than or equal to current shape in each dimension");
		}
	}

	size_t rank = targetShape.size();
	size_t blockRank = (rank >= 1) ? rank - 1 : 0;
	size_t blockLen = targetShape.back();
	const size_t blockBytes = blockLen * typeSize();

	// fill pattern：单元素 deposit（与 write(element) 槽位语义一致）
	auto pattern = deposit(std::span<const T>(&fillData, 1));

	// 块一致性：缺失块整块填充；已有块扩容且仅新区域填充（旧值保留）。
	auto ensureBlock = [&](const Shape& path) {
		auto it = _dataMain.find(path);
		if (it == _dataMain.end()) {
			updateCatalog(path, "TensorData::expand");
			std::vector<T> vals(blockLen, fillData);
			commitData(path, deposit(std::span<const T>(vals.data(), vals.size())));
		} else if (it->second.size() < blockBytes) {
			// 已有块扩容：仅对新区域按 pattern 填充
			const size_t oldBytes = it->second.size();
			DataBlock block = std::move(it->second);
			block.resize(blockBytes, std::byte(0));
			for (size_t off = oldBytes; off < blockBytes; off += pattern.size())
				std::memcpy(block.data() + off, pattern.data(), pattern.size());
			updateCatalog(path, "TensorData::expand");
			commitData(path, std::move(block));
		} else {
			// 补登记 catalog（幂等），确保形状与 targetShape 一致
			updateCatalog(path, "TensorData::expand");
		}
	};

	if (blockRank == 0) {
		// 一维目标：整块 = root path
		ensureBlock({});
	} else {
		Shape blockPath(blockRank, 0);
		bool done = false;
		while (!done) {
			ensureBlock(blockPath);
			for (size_t i = 0; i < blockRank; ++i) {
				if (++blockPath[i] < targetShape[i])
					break;
				blockPath[i] = 0;
				if (i + 1 == blockRank)
					done = true;
			}
		}
	}

	setViewFlag();
	clearCache(); // commitData 可能已清缓存，此处兜底一致性
	return *this;
}
} // namespace DC