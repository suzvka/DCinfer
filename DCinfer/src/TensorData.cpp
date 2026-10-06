#include "TensorData.h"
#include <cassert>
#include <limits>
#include <numeric>

namespace DC {

namespace {
/// 形状乘积守卫（CORE-03）：shape 含 0 维（或空乘积）时乘积为 0，
/// 作为除数即未定义行为——前置拒绝并抛出明确异常。
/// 仅供双参构造的 typeSize 推导使用（委托构造的初始化列表中无法先行校验）。
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

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// TensorData implementation
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

TensorData::TensorData()
	: _dataSize(0), _dataMain({}), _dataCatalog({}), _dataCache({}), _shapeCache({}), _validFlags(0), _isScalar(false),
	  _typeSize(0) {}

// ── 拷贝/移动（自定义：内部互斥锁不可复制；拷贝产出非冻结副本）──

TensorData::TensorData(const TensorData& other)
	: _dataMain(other._dataMain), _dataCatalog(other._dataCatalog), _typeSize(other._typeSize),
	  _dataSize(other._dataSize), _isScalar(other._isScalar), _dataCache(other._dataCache),
	  _shapeCache(other._shapeCache), _validFlags(other._validFlags.load(std::memory_order_acquire)) {
	// _frozen 默认 false：拷贝产出独立的可变副本（clone 语义）
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
		_frozen = false; // 拷贝产出独立的可变副本（clone 语义）
	}
	return *this;
}

TensorData::TensorData(TensorData&& other) noexcept
	: _dataMain(std::move(other._dataMain)), _dataCatalog(std::move(other._dataCatalog)),
	  _typeSize(other._typeSize), _dataSize(other._dataSize), _isScalar(other._isScalar),
	  _dataCache(std::move(other._dataCache)), _shapeCache(std::move(other._shapeCache)),
	  _validFlags(other._validFlags.load(std::memory_order_acquire)), _frozen(other._frozen) {
	// 移动 = 载荷身份转移：冻结状态随行
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
		_frozen = other._frozen; // 移动 = 载荷身份转移：冻结状态随行
	}
	return *this;
}

// ── 冻结（共享发布）──

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
		buildCache(); // 预物化：冻结后只读路径零惰性物化（并发共享的前提）
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
		if (d == 0)
			throw std::invalid_argument("TensorData: shape dimensions must be > 0");
		if (elementCount > std::numeric_limits<size_t>::max() / d)
			throw std::invalid_argument("TensorData: shape element count overflows");
		elementCount *= d;
	}
	if (elementCount > std::numeric_limits<size_t>::max() / typeSize)
		throw std::invalid_argument("TensorData: required byte size overflows");
	// An empty block means metadata-only construction; preserve the public
	// Tensor API's lazy/no-payload semantics after validating the shape.
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
		shape.push_back(_dataSize / typeSize()); // Last dimension is block element count
	}
	return shape;
}

void TensorData::loadData(const Shape& shape, size_t typeSize, DataBlock&& bytes) {
	_ensureMutable("TensorData::loadData");
	// 入口校验（与三参构造同一不变量）：shape 元素总数 × typeSize 必须精确
	// 等于缓冲区字节数——稠密直通语义是"全部数据都在本缓冲区"，声明与
	// 实际不一致时下游按 shape 寻址将越过缓冲区。校验先于 clear：失败时
	// 对象状态不变。注意：校验对象是调用方自身声明的自洽性（元数据），
	// 字节载荷本身仍原样进入 cache，不做任何解释（"不透明传递"哲学
	// 仅适用于载荷内容，不适用于用于寻址的元数据）。
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
	// Ensure sparse view is materialized from dense cache when needed
	ensureView();
	if (hasCache()) {
		// materialization happened: invalidate cache and mark view valid
		clearCache();
		setViewFlag();
		return;
	}
	// If no view exists but we are entering edit mode,
	// create an empty view state rather than failing.
	if (!hasView()) {
		setViewFlag();
		return;
	}
}

//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
// Private helpers implementation
//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

void TensorData::syncDenseCacheMeta(const Shape& denseShape) {
	_shapeCache = denseShape;
	_dataSize = 0;
	if (!denseShape.empty() && typeSize() > 0) {
		_dataSize = static_cast<size_t>(denseShape.back()) * typeSize();
	}
}

void TensorData::ensureCache() {
	if (hasCache()) {
		return; // 快路径：已物化（含冻结发布后的全部只读访问）
	}
	std::lock_guard lk(_lazyMutex); // 双重检查：惰性物化的唯一构建路径
	if (!hasCache() && hasView()) {
		buildCache();
	}
}

void DC::TensorData::ensureView() {
	if (hasView()) {
		setViewFlag();
		return; // 快路径：已物化
	}
	std::lock_guard lk(_lazyMutex); // 双重检查：惰性物化的唯一构建路径
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
		// 越界/超长块（catalog 与数据不一致导致的脏数据）保持跳过容错，
		// debug 构建经断言暴露不一致（#8-7：不再完全无诊断）
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
		// Treat whole data as one block at root path {}
		_dataMain[{}] = _dataCache;
		_dataSize = _dataCache.size();
		setViewFlag(); // 视图已从缓存重建（#8-8：与多维路径一致，不依赖调用顺序）
		return;
	}

	// Block size: last dim count * typeSize
	_dataSize = _shapeCache.back() * typeSize();
	if (_dataSize == 0) {
		return;
	}

	const size_t pathDims = (_shapeCache.size() >= 2) ? (_shapeCache.size() - 1) : 0;
	_dataCatalog.resize(pathDims);
	for (size_t i = 0; i < pathDims; ++i) {
		// iterate in unsigned domain to match _shapeCache element type (size_t)
		for (size_t idx = 0; idx < _shapeCache[i]; ++idx) {
			// store as int64_t in the index sets
			_dataCatalog[i].insert(idx);
		}
	}

	if (pathDims == 0) {
		// Degrade to 1D: store whole block at root path {}
		_dataMain[{}] = _dataCache;
		setViewFlag(); // 视图已从缓存重建（#8-8：与多维路径一致）
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
			// Use shape bounds, not data bytes.
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

	// 多维前缀裁剪（行主序）：保留每个维度的前 targetShape[i] 个坐标所
	// 寻址的元素。旧的"扁平 resize 到新元素总数"只对一维正确——对
	// {2,3}→{2,2} 会保留 1,2,3,4（裁成 [[1,2],[3,4]]），而每维前缀
	// 语义应为 1,2,4,5（[[1,2],[4,5]]）。
	// 实现：新建目标缓冲；外层按 row-major 遍历去掉最后一维的块坐标，
	// 对每行在源/目标布局下分别计算块偏移，整行 memcpy（最后一维长度
	// 个元素）；一维退化为 root 单块拷贝，与旧扁平行为一致。
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
			// 字典序递增（带进位，最内维最先）；最高位进位溢出即遍历完成
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
	// 稀疏视图仍指向旧形状的块布局：一并失效，后续读写经惰性重建保持一致
	clearView();
	return *this;
}
} // namespace DC