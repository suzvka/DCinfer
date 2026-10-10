#include "TensorData.h"
#include <cassert>
#include <limits>
#include <numeric>

namespace DC {

namespace {
/// 形状乘积守卫：乘积为 0 作除数即 UB，前置拒绝；仅供双参构造的 typeSize 推导。
size_t checkedShapeProduct(size_t product) {
	if (product == 0)
		throw std::invalid_argument("TensorData: shape product must be > 0 for typeSize inference");
	return product;
}

size_t checkedShapeProduct(const TensorData::Shape& shape) {
	size_t product = 1;
	for (const auto d : shape) {
		if (d == 0)
			throw std::invalid_argument("TensorData: shape dimensions must be > 0");
		if (product > std::numeric_limits<size_t>::max() / d)
			throw std::invalid_argument("TensorData: shape element count overflows");
		product *= d;
	}
	return checkedShapeProduct(product);
}
} // namespace

TensorData::TensorData()
	: _dataSize(0), _dataMain({}), _dataCatalog({}), _dataCache({}), _shapeCache({}), _validFlags(0), _isScalar(false),
	  _typeSize(0) {}

TensorData::TensorData(const TensorData& other)
	: _dataMain(other._dataMain), _dataCatalog(other._dataCatalog), _typeSize(other._typeSize),
	  _dataSize(other._dataSize), _isScalar(other._isScalar), _dataCache(other._dataCache),
	  _shapeCache(other._shapeCache), _validFlags(other._validFlags.load(std::memory_order_acquire)) {
	// _frozen 默认 false：拷贝为独立可变副本
}

TensorData& TensorData::operator=(const TensorData& other) {
	if (this != &other) {
		_dataMain = other._dataMain;
		_dataCatalog = other._dataCatalog;
		_typeSize = other._typeSize;
		_dataSize = other._dataSize;
		_isScalar = other._isScalar;
		_dataCache = other._dataCache;
		_shapeCache = other._shapeCache;
		_validFlags.store(other._validFlags.load(std::memory_order_acquire), std::memory_order_release);
		_frozen = false; // 拷贝为独立可变副本
	}
	return *this;
}

TensorData::TensorData(TensorData&& other) noexcept
	: _dataMain(std::move(other._dataMain)), _dataCatalog(std::move(other._dataCatalog)),
	  _typeSize(other._typeSize), _dataSize(other._dataSize), _isScalar(other._isScalar),
	  _dataCache(std::move(other._dataCache)), _shapeCache(std::move(other._shapeCache)),
	  _validFlags(other._validFlags.load(std::memory_order_acquire)), _frozen(other._frozen) {
	// 移动即身份转移：冻结状态随行
}

TensorData& TensorData::operator=(TensorData&& other) noexcept {
	if (this != &other) {
		_dataMain = std::move(other._dataMain);
		_dataCatalog = std::move(other._dataCatalog);
		_typeSize = other._typeSize;
		_dataSize = other._dataSize;
		_isScalar = other._isScalar;
		_dataCache = std::move(other._dataCache);
		_shapeCache = std::move(other._shapeCache);
		_validFlags.store(other._validFlags.load(std::memory_order_acquire), std::memory_order_release);
		_frozen = other._frozen; // 移动即身份转移：冻结状态随行
	}
	return *this;
}

void TensorData::_ensureMutable(const char* api) const {
	if (_frozen) {
		throw TensorException(TensorException::ErrorType::Frozen, api,
							  "tensor is frozen (published to a shared connection, read-only); "
							  "clone before mutating");
	}
}

void TensorData::freeze() {
	std::lock_guard lk(_lazyMutex);
	if (_frozen) {
		return;
	}
	if (!hasCache() && hasView()) {
		buildCache(); // 预物化：冻结后只读路径零惰性物化，并发共享的前提
	}
	_frozen = true;
}

TensorData::TensorData(const Shape& shape, size_t typeSize, DataBlock&& denseBytes)
	: _dataSize(0), _dataMain({}), _dataCatalog({}), _dataCache({}), _shapeCache({}), _validFlags(0), _isScalar(false),
	  _typeSize(0) {
	_isScalar = shape.empty();

	if (typeSize == 0) {
		throw std::invalid_argument("TensorData: typeSize must be > 0");
	}

	size_t elementCount = 1;
	for (auto d : shape) {
		if (d == 0) {
			// 零维即 0 元素张量：空文本或空集合的合法表示，无分配无溢出风险；
			// 空载荷走 metadata-only lazy 语义，带载荷走下方 mismatch 拒绝。
			elementCount = 0;
			break;
		}
		if (elementCount > std::numeric_limits<size_t>::max() / d)
			throw std::invalid_argument("TensorData: shape element count overflows");
		elementCount *= d;
	}
	if (elementCount > std::numeric_limits<size_t>::max() / typeSize)
		throw std::invalid_argument("TensorData: required byte size overflows");
	// 空块即 metadata-only 构造：校验形状后保持惰性与无载荷语义。
	if (denseBytes.empty()) {
		setTypeSize(typeSize);
		return;
	}
	const size_t expectedBytes = elementCount * typeSize;
	if (denseBytes.size() != expectedBytes) {
		throw std::invalid_argument("TensorData: denseBytes.size() does not match shape product");
	}

	loadData(shape, typeSize, std::move(denseBytes));
}

TensorData::TensorData(const Shape& shape, DataBlock&& data)
	: TensorData(shape,
				 (!shape.empty())
					 ? (data.size() / checkedShapeProduct(shape))
					 : data.size(),
				 std::move(data)) {}

std::span<const std::byte> TensorData::data() const {
	if (!hasCache()) {
		if (empty()) {
			return {};
		}
		const_cast<TensorData*>(this)->ensureCache();
	}
	return std::span<const std::byte>(_dataCache.data(), _dataCache.size());
}

size_t TensorData::size() const {
	if (hasCache())
		return _dataCache.size();
	return denseElementCount(getDenseShape()) * typeSize();
}

void TensorData::setTypeSize(size_t typeSize) {
	_ensureMutable("TensorData::setTypeSize");
	if (typeSize == 0) {
		throw std::invalid_argument("TensorData::setTypeSize: typeSize must be > 0");
	}
	_typeSize = typeSize;
}

void TensorData::clear() {
	_ensureMutable("TensorData::clear");
	_dataMain.clear();
	_dataCatalog.clear();
	_dataSize = 0;
	_dataCache.clear();
	_shapeCache.clear();
	clearCache();
	clearView();
	setScalar(false);
}

TensorData::Shape TensorData::getCurrentShape() const {
	if (hasCache()) {
		return _shapeCache;
	}

	if (isScalar()) {
		return {};
	}

	Shape shape(_dataCatalog.size(), 0);
	for (size_t i = 0; i < _dataCatalog.size(); ++i) {
		if (!_dataCatalog[i].empty()) {
			shape[i] = (*std::max_element(_dataCatalog[i].begin(), _dataCatalog[i].end())) + 1;
		}
	}
	if (_dataSize > 0 && typeSize() > 0) {
		shape.push_back(_dataSize / typeSize()); // 最后一维即块内元素数
	}
	return shape;
}

void TensorData::loadData(const Shape& shape, size_t typeSize, DataBlock&& bytes) {
	_ensureMutable("TensorData::loadData");
	// 入口校验：shape 积 × typeSize 必须等于缓冲字节数，否则下游按 shape
	// 寻址将越界；校验先于 clear，失败时对象状态不变。载荷原样进入 cache
	// 不做解释，“不透明传递”仅适用于载荷内容，不适用于寻址元数据。
	if (typeSize == 0)
		throw std::invalid_argument("TensorData::loadData: typeSize must be > 0");
	size_t elementCount = 1;
	for (auto d : shape) {
		if (d == 0)
			throw std::invalid_argument("TensorData::loadData: shape dimensions must be > 0");
		if (elementCount > std::numeric_limits<size_t>::max() / d)
			throw std::invalid_argument("TensorData::loadData: shape element count overflows");
		elementCount *= static_cast<size_t>(d);
	}
	if (elementCount > std::numeric_limits<size_t>::max() / typeSize)
		throw std::invalid_argument("TensorData::loadData: required byte size overflows");
	const size_t expectedBytes = elementCount * typeSize;
	if (bytes.size() != expectedBytes) {
		throw std::invalid_argument("TensorData::loadData: bytes.size() ("
									+ std::to_string(bytes.size())
									+ ") does not match shape product * typeSize ("
									+ std::to_string(expectedBytes) + ")");
	}

	clear();
	setTypeSize(typeSize);
	_dataCache = std::move(bytes);
	_shapeCache = shape;
	setCacheFlag();
	clearView();
	setScalar(shape.empty());

	if (!shape.empty() && typeSize > 0) {
		_dataSize = shape.back() * typeSize;
	}
}

void TensorData::editMode() {
	_ensureMutable("TensorData::editMode");
	ensureView();
	if (hasCache()) {
		clearCache();
		setViewFlag();
		return;
	}
	if (!hasView()) {
		setViewFlag();
		return;
	}
}

void TensorData::syncDenseCacheMeta(const Shape& denseShape) {
	_shapeCache = denseShape;
	_dataSize = 0;
	if (!denseShape.empty() && typeSize() > 0) {
		_dataSize = static_cast<size_t>(denseShape.back()) * typeSize();
	}
}

void TensorData::ensureCache() {
	if (hasCache()) {
		return; // 快路径：已物化
	}
	std::lock_guard lk(_lazyMutex); // 双重检查：惰性物化唯一路径
	if (!hasCache() && hasView()) {
		buildCache();
	}
}

void DC::TensorData::ensureView() {
	if (hasView()) {
		setViewFlag();
		return; // 快路径：已物化
	}
	std::lock_guard lk(_lazyMutex); // 双重检查：惰性物化唯一路径
	if (hasCache() && !hasView()) {
		buildView();
	}

	if (hasView()) {
		setViewFlag();
	}
}

size_t TensorData::blockOffset(const Shape& blockPath, const Shape& denseShape) const {
	if (denseShape.empty()) {
		if (!blockPath.empty()) {
			throw std::out_of_range("TensorData::calculateBlockOffsetBytes: blockPath rank mismatch");
		}
		return 0;
	}

	if (blockPath.size() + 1 != denseShape.size()) {
		throw std::out_of_range("TensorData::calculateBlockOffsetBytes: blockPath rank mismatch");
	}

	size_t elementOffset = 0;
	size_t multiplier = static_cast<size_t>(denseShape.back());
	for (size_t k = blockPath.size(); k-- > 0;) {
		elementOffset += static_cast<size_t>(blockPath[k]) * multiplier;
		multiplier *= static_cast<size_t>(denseShape[k]);
	}
	return elementOffset * typeSize();
}

size_t TensorData::elementOffset(const Shape& elementPath, const Shape& denseShape) const {
	if (denseShape.empty()) {
		if (!elementPath.empty()) {
			throw std::out_of_range("TensorData::calculateElementOffsetBytes: elementPath rank mismatch");
		}
		return 0;
	}

	if (elementPath.size() != denseShape.size()) {
		throw std::out_of_range("TensorData::calculateElementOffsetBytes: elementPath rank mismatch");
	}

	const Shape blockPath(elementPath.begin(), elementPath.end() - 1);
	const size_t blockBase = blockOffset(blockPath, denseShape);
	return blockBase + static_cast<size_t>(elementPath.back()) * typeSize();
}

TensorData::Shape TensorData::getDenseShape() const {
	if (isScalar()) {
		return {};
	}
	Shape shape(_dataCatalog.size());
	for (size_t i = 0; i < _dataCatalog.size(); ++i) {
		if (_dataCatalog[i].empty()) {
			shape[i] = 0;
		} else {
			shape[i] = *std::max_element(_dataCatalog[i].begin(), _dataCatalog[i].end()) + 1;
		}
	}
	if (_dataSize > 0) {
		shape.push_back(_dataSize / typeSize());
	}
	return shape;
}

void TensorData::buildCache() {
	const auto denseShape = getDenseShape();
	const size_t totalBytes = denseElementCount(denseShape) * typeSize();
	_dataCache.assign(totalBytes, std::byte(0));
	for (const auto& [path, block] : _dataMain) {
		const size_t offset = blockOffset(path, denseShape);
		// 越界或超长块等脏数据跳过容错；debug 构建经断言暴露不一致
		const size_t copyBytes = std::min(block.size(), _dataSize);
		assert(block.size() <= _dataSize
			   && "TensorData::buildCache: block exceeds _dataSize (silently truncated)");
		assert(offset + copyBytes <= _dataCache.size()
			   && "TensorData::buildCache: block out of dense cache range (catalog/data inconsistent)");
		if (offset + copyBytes <= _dataCache.size()) {
			std::memcpy(_dataCache.data() + offset, block.data(), copyBytes);
		}
	}
	syncDenseCacheMeta(denseShape);

	setCacheFlag();
}

void TensorData::buildView() {
	if (!hasCache()) {
		return;
	}

	if (_shapeCache.empty() && !isScalar()) {
		return;
	}

	clearView();

	if (isScalar()) {
		_dataMain[{}] = _dataCache;
		_dataSize = _dataCache.size();
		setViewFlag(); // 视图已从缓存重建，与多维路径一致
		return;
	}

	_dataSize = _shapeCache.back() * typeSize();
	if (_dataSize == 0) {
		return;
	}

	const size_t pathDims = (_shapeCache.size() >= 2) ? (_shapeCache.size() - 1) : 0;
	_dataCatalog.resize(pathDims);
	for (size_t i = 0; i < pathDims; ++i) {
		for (size_t idx = 0; idx < _shapeCache[i]; ++idx) {
			_dataCatalog[i].insert(idx);
		}
	}

	if (pathDims == 0) {
		_dataMain[{}] = _dataCache;
		setViewFlag(); // 视图已从缓存重建，与多维路径一致
		return;
	}

	Shape path(pathDims, 0);
	size_t offsetBytes = 0;
	while (true) {
		if (offsetBytes + _dataSize > _dataCache.size()) {
			break;
		}
		DataBlock block(_dataCache.begin() + offsetBytes, _dataCache.begin() + offsetBytes + _dataSize);
		_dataMain[path] = std::move(block);
		offsetBytes += _dataSize;

		std::ptrdiff_t dim = static_cast<std::ptrdiff_t>(pathDims) - 1;
		while (dim >= 0) {
			path[dim]++;
			if (path[dim] < _shapeCache[dim]) {
				break;
			}
			path[dim] = 0;
			--dim;
		}
	}

	setViewFlag();
}

bool TensorData::checkType(size_t expectedSize, const std::string& callerName) const {
	if (typeSize() == 0) {
		return false;
	}

	else if (typeSize() % expectedSize != 0) {
		throw std::invalid_argument(std::string(callerName) + ": type size mismatch");
	}

	return true;
}

void TensorData::updateCatalog(const Shape& path, const std::string& callerName) {
	if (path.size() != _dataCatalog.size()) {
		size_t preservedTypeSize = typeSize();
		clear();
		setTypeSize(preservedTypeSize);
		_dataCatalog.resize(path.size());
	}
	for (size_t index = 0; index < path.size(); ++index) {
		_dataCatalog[index].insert(path[index]);
	}
}

void TensorData::commitData(const Shape& path, DataBlock&& block) {
	if (_dataSize < block.size()) {
		_dataSize = block.size();
	}
	_dataMain[path] = std::move(block);
	clearCache();
}

size_t TensorData::denseElementCount(const Shape& shape) {
	if (shape.empty())
		return 1;
	size_t n = 1;
	for (auto d : shape)
		n *= static_cast<size_t>(d);
	return n;
}

void TensorData::clearCache() {
	_validFlags.fetch_and(static_cast<uint8_t>(~FlagCache), std::memory_order_release);
	_dataCache.clear();
	_shapeCache.clear();
}

void TensorData::clearView() {
	_validFlags.fetch_and(static_cast<uint8_t>(~FlagView), std::memory_order_release);
	_dataMain.clear();
	_dataCatalog.clear();
}

std::span<std::byte> TensorData::calcWriteRegion(const Shape& path) {
	const auto& shape = _shapeCache;

	if (path.size() > shape.size())
		throw std::out_of_range("TensorData::writeCache: path rank exceeds tensor rank");

	size_t elementCount = 0;
	size_t offsetBytes = 0;

	if (path.size() == shape.size()) {
		elementCount = 1;
		offsetBytes = elementOffset(path, shape);
	} else if (path.size() + 1 == shape.size()) {
		elementCount = static_cast<size_t>(shape.back());
		offsetBytes = blockOffset(path, shape);
	} else {
		throw std::invalid_argument("TensorData::writeCache: unsupported slice form");
	}

	const size_t totalBytes = elementCount * typeSize();
	if (offsetBytes + totalBytes > _dataCache.size())
		throw std::out_of_range("TensorData::writeCache: write exceeds cache size");

	return std::span<std::byte>(_dataCache.data() + offsetBytes, totalBytes);
}

bool TensorData::writeCacheRaw(const Shape& path, DataBlock&& rawBytes, size_t typeSize, const char* apiName) {
	_ensureMutable(apiName);
	if (!checkType(typeSize, apiName)) {
		setTypeSize(typeSize);
	}

	if (!hasCache())
		return false;

	auto region = calcWriteRegion(path);

	// Allow inputs smaller than the target region by zero-padding to the full region size.
	if (rawBytes.size() != static_cast<size_t>(region.size())) {
		if (rawBytes.size() < static_cast<size_t>(region.size())) {
			DataBlock padded(static_cast<size_t>(region.size()), std::byte(0));
			if (!rawBytes.empty())
				std::memcpy(padded.data(), rawBytes.data(), rawBytes.size());
			rawBytes = std::move(padded);
		} else {
			throw std::invalid_argument("TensorData::writeCache: data size mismatch");
		}
	}

	std::memcpy(region.data(), rawBytes.data(), region.size());
	clearView();
	return true;
}

TensorData::DataBlock TensorData::getData() {
	_ensureMutable("TensorData::getData");
	if (!hasCache()) {
		ensureCache();
	}

	DataBlock dataBlock = std::move(_dataCache);
	clearCache();
	return std::move(dataBlock);
}

TensorData& TensorData::crop(const Shape& targetShape) {
	_ensureMutable("TensorData::crop");
	if (!hasCache()) {
		ensureCache();
	}
	if (!hasCache()) {
		throw std::runtime_error("TensorData::crop: no data to crop");
	}
	const auto& currentShape = _shapeCache;
	if (targetShape.size() != currentShape.size()) {
		throw std::invalid_argument("TensorData::crop: target shape rank mismatch");
	}
	for (size_t i = 0; i < targetShape.size(); ++i) {
		if (targetShape[i] > currentShape[i]) {
			throw std::invalid_argument(
				"TensorData::crop: target shape must be smaller than or equal to current shape in each dimension");
		}
	}
	size_t newElementCount = 1;
	for (auto d : targetShape) {
		newElementCount *= static_cast<size_t>(d);
	}
	const size_t newByteSize = newElementCount * typeSize();
	if (newByteSize > _dataCache.size()) {
		throw std::runtime_error("TensorData::crop: calculated byte size exceeds current cache size");
	}

	// 多维前缀裁剪，行主序：每维保留前 targetShape[i] 个坐标；旧扁平
	// resize 仅一维正确，如 {2,3} 裁到 {2,2} 应得 1,2,4,5 而非 1,2,3,4。
	// 实现：按 row-major 块坐标逐行 memcpy；一维退化为 root 单块拷贝。
	DataBlock cropped;
	cropped.resize(newByteSize);
	if (targetShape.empty()) {
		// 0-D 标量：单元素原样保留
		std::memcpy(cropped.data(), _dataCache.data(), typeSize());
	} else if (newElementCount > 0) {
		const size_t blockRank = targetShape.size() - 1;
		const size_t rowBytes = targetShape.back() * typeSize();
		Shape blockPath(blockRank, 0);
		while (true) {
			const size_t srcOffset = blockOffset(blockPath, currentShape);
			const size_t dstOffset = blockOffset(blockPath, targetShape);
			std::memcpy(cropped.data() + dstOffset, _dataCache.data() + srcOffset, rowBytes);
			// 字典序递增，进位且最内维最先；最高位溢出即遍历完成
			bool carry = true;
			for (size_t i = blockRank; i-- > 0;) {
				if (++blockPath[i] < targetShape[i]) {
					carry = false;
					break;
				}
				blockPath[i] = 0;
			}
			if (carry)
				break;
		}
	}

	_dataCache = std::move(cropped);
	syncDenseCacheMeta(targetShape);
	// 稀疏视图指向旧形状：一并失效，后续经惰性重建
	clearView();
	return *this;
}
} // namespace DC