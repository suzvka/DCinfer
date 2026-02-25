#include "TensorData.h"
#include <limits>
#include <numeric>

namespace DC
{
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
	// TensorData implementation
	//>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

	TensorData::TensorData() :_dataSize(0), _dataMain({}), _dataCatalog({}), _dataCache({}), _shapeCache({}), _validFlags(0), _isScalar(false), _typeSize(0) {}

	TensorData::TensorData(
		const Shape& shape,
		size_t typeSize,
		DataBlock&& denseBytes
	) : _dataSize(0), _dataMain({}), _dataCatalog({}), _dataCache({}), _shapeCache({}), _validFlags(0)
	{
		if (denseBytes.empty()) {
			setTypeSize(typeSize);
			return;
		}

		if (typeSize == 0) {
			throw std::invalid_argument("TensorData: typeSize must be > 0");
		}

		size_t elementCount = 1;
		for (auto d : shape) {
			elementCount *= static_cast<size_t>(d);
		}
		const size_t expectedBytes = elementCount * typeSize;
		if (denseBytes.size() != expectedBytes) {
			throw std::invalid_argument("TensorData: denseBytes.size() does not match shape product");
		}

		loadData(shape, typeSize, std::move(denseBytes));
	}

	TensorData::TensorData(
		const Shape& shape,
		DataBlock&& data
	) : TensorData(
		shape,
		(!shape.empty()) ? (data.size() / std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), [](size_t a, auto b){ return a * static_cast<size_t>(b); })) : data.size(),
		std::move(data)
	) {}

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
		if (hasCache()) return _dataCache.size();
		return denseElementCount(getDenseShape()) * typeSize();
	}

	void TensorData::setTypeSize(size_t typeSize) {
		if (typeSize == 0) {
			throw std::invalid_argument("TensorData::setTypeSize: typeSize must be > 0");
		}
		_typeSize = typeSize;
	}

	void TensorData::clear() {
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
 		// Ensure sparse view is materialized from dense cache when needed
 		ensureView();
		if (hasCache()) {
			// materialization happened: invalidate cache and mark view valid
			clearCache();
			setViewFlag();
 			return;
 		}
		// If no view exists but we are entering edit mode (e.g. first write to an
		// initially-empty tensor), create an empty view state rather than
		// failing. This preserves any catalog information that may have been
		// prepared earlier and allows subsequent write() to populate blocks.
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
		if (!hasCache() && hasView()) {
			buildCache();
		}
	}

	void DC::TensorData::ensureView() {
    if (hasCache() && !hasView()) {
        buildView();
    }

    // If there is no view materialized and no cache to build it from,
    // do not clear existing catalog/state. Calling clearView() here
    // would discard catalog information that may have been prepared
    // by updateCatalog() for upcoming writes. Only set the view flag
    // when a view actually exists.
    if (!hasView()) {
        return;
    }

    setViewFlag();
	}

	size_t TensorData::blockOffset(const Shape& blockPath, const Shape& denseShape) const {
		// 处理标量
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
				shape[i] = 0; // if dimension has no index, size is 0
			}
			else {
				// Shape size is max index + 1
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
			const size_t copyBytes = std::min(block.size(), _dataSize);
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
		if (shape.empty()) return 1;
		size_t n = 1;
		for (auto d : shape) n *= static_cast<size_t>(d);
		return n;
	}

	void TensorData::clearCache() {
		_validFlags &= static_cast<uint8_t>(~FlagCache);
		_dataCache.clear();
		_shapeCache.clear();
	}

	void TensorData::clearView() {
		_validFlags &= static_cast<uint8_t>(~FlagView);
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
		}
		else if (path.size() + 1 == shape.size()) {
			elementCount = static_cast<size_t>(shape.back());
			offsetBytes = blockOffset(path, shape);
		}
		else {
			throw std::invalid_argument("TensorData::writeCache: unsupported slice form");
		}

		const size_t totalBytes = elementCount * typeSize();
		if (offsetBytes + totalBytes > _dataCache.size())
			throw std::out_of_range("TensorData::writeCache: write exceeds cache size");

		return std::span<std::byte>(_dataCache.data() + offsetBytes, totalBytes);
	}

	bool TensorData::writeCacheRaw(const Shape& path, DataBlock&& rawBytes, size_t typeSize, const char* apiName) {
		if (!checkType(typeSize, apiName)) { 
			setTypeSize(typeSize);
		}

		if (!hasCache()) return false;

		auto region = calcWriteRegion(path);

		// Allow inputs smaller than the target region by zero-padding to the full region size.
		if (rawBytes.size() != static_cast<size_t>(region.size())) {
			if (rawBytes.size() < static_cast<size_t>(region.size())) {
				DataBlock padded(static_cast<size_t>(region.size()), std::byte(0));
				if (!rawBytes.empty()) std::memcpy(padded.data(), rawBytes.data(), rawBytes.size());
				rawBytes = std::move(padded);
			}
			else {
				throw std::invalid_argument("TensorData::writeCache: data size mismatch");
			}
		}

		std::memcpy(region.data(), rawBytes.data(), region.size());
		clearView();
		return true;
	}
}